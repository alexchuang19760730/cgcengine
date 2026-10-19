// =============================================================================
// L4 冷页归还 · 阶段 0 独立 Metal 探针（不碰引擎，纯小程序）
//
// 回答的问题（决定 docs/les/L4_COLD_PAGE_RECLAIM_PROBE_2026-10-18.md 能否往下走）：
//   Q1 用 ggml 同款路徑 vm_allocate(VM_FLAGS_ANYWHERE) + newBufferWithBytesNoCopy
//      (storageModeShared, deallocator=nil) 建的 shared region，GPU 仍「持有 MTLBuffer
//      引用」期間，host 對其底層頁 madvise(MADV_DONTNEED / MADV_FREE) 會不會成功、
//      該 region 的 pages_resident 是否立刻下降？（若 GPU 映射把頁 wire 住，madvise 無效。）
//   Q2 madvise 之後 host 立即讀到什麼（零頁 / 舊值）。
//   Q3 madvise 之後讓 GPU 再讀一次：hang / crash / 讀零 / 仍讀舊值。
//   Q4 host 重新全覆蓋重填後，GPU 再讀是否 bit-exact 恢復。
//   Q5 MADV_FREE（lazy，pages_reusable 升、記憶體壓力才回收）對照。
//   Q6 Metal setPurgeableState(volatile/nonVolatile) 整 buffer 粒度對照。
//
// 測量方法（v2 修正，避免虛假 PASS）：
//   - 不用全 buffer XOR（連續 pattern 會抵消成 0，無區分度）；改 GPU 讀「頭/尾 sentinel」。
//   - 不用 task_vm_info.internal（vsize 口徑，不反映 resident 歸零）；用 mach_vm_region +
//     VM_REGION_EXTENDED_INFO 的 pages_resident / pages_swapped_out / pages_reusable 精確測
//     該 region（page_size 本機 16 KiB）。
//
// 編譯：
//   clang++ -std=c++17 -fobjc-arc -O2 -framework Foundation -framework Metal \
//     docs/les/tools/l4_reclaim_probe.mm -o /tmp/l4_reclaim_probe
// 跑：
//   /tmp/l4_reclaim_probe [region_mib=128]
//
// 進階段 2 的門檻（須全部成立）：
//   DONTNEED 後 pages_resident 明顯歸零（或降）＋ host 讀頭=0 ＋ Q3 GPU 不 hang（讀零可接受）
//   ＋ Q4 重填後 host/GPU 頭尾 sentinel 完全相等（bit-exact）。
// 任一 FAIL 則階段 2 不可行，改走 purgeable 整 buffer / 文件 backing。
// 這是可行性+安全性探針，不量 t/s、不許諾收益。
// =============================================================================
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_region.h>
#include <mach/vm_statistics.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#ifndef MADV_FREE_REUSABLE
#define MADV_FREE_REUSABLE 7   // macOS libc malloc 真正歸還頁的機制：立刻計 pages_reusable
#endif
#ifndef MADV_FREE_REUSE
#define MADV_FREE_REUSE 8     // 撤銷 reusable 標記（重新使用前）
#endif

// ── 進程 phys_footprint（MiB）：系統判定記憶體壓力 / swap 的口徑，這才是冷頁歸還的主判據 ──
static int64_t phys_mib(){
    task_vm_info_data_t t{}; mach_msg_type_number_t cnt=TASK_VM_INFO_COUNT;
    task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&t,&cnt);
    return (int64_t)(t.phys_footprint/1048576);
}

// ── per-region 分頁狀態（resident / swapped / reusable，單位頁）──
struct RStat { int64_t size_mib, resident, swapped, reusable, dirty, shared_np; bool ok; };
static RStat region_stat(void* addr) {
    RStat s{0,0,0,0,0,0,false};
    mach_vm_address_t a = (mach_vm_address_t)addr;
    mach_vm_size_t sz = 0;
    vm_region_extended_info_data_t ei{};
    mach_msg_type_number_t cnt = VM_REGION_EXTENDED_INFO_COUNT;
    mach_port_t obj = MACH_PORT_NULL;
    kern_return_t kr = mach_vm_region(mach_task_self(), &a, &sz, VM_REGION_EXTENDED_INFO,
                                     (vm_region_info_t)&ei, &cnt, &obj);
    if (kr != KERN_SUCCESS) { return s; }
    s.ok = true;
    s.size_mib = (int64_t)(sz/1048576);
    s.resident = ei.pages_resident;
    s.swapped  = ei.pages_swapped_out;
    s.reusable = ei.pages_reusable;
    s.dirty    = ei.pages_dirtied;
    s.shared_np = ei.pages_shared_now_private;
    return s;
}
static void stat_line(const char* tag, const RStat& s) {
    const long ps = sysconf(_SC_PAGESIZE);
    double m = (double)ps/1048576.0;
    printf("    [%s] region %lldMiB resident=%lld(%.1fMiB) swapped=%lld reusable=%lld dirty=%lld sharedNP=%lld\n",
           tag, s.size_mib, s.resident, s.resident*m, s.swapped, s.reusable, s.dirty, s.shared_np);
}

// ── GPU 探針：單線程讀頭/尾 sentinel（運行時編譯，自包含）──
static NSString* const kKernelSrc = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void probe(device const uint* in, device uint* out, constant uint& len, uint gid [[thread_position_in_grid]]) {\n"
"    if (gid == 0) { out[0] = in[0]; out[1] = in[len-1]; }\n"
"}\n";

// GPU 讀頭/尾。返回 false=命令錯誤/hang。
static bool gpu_probe(id<MTLCommandQueue> q, id<MTLComputePipelineState> pso,
                      id<MTLBuffer> data, uint32_t len_u32, uint32_t* head, uint32_t* tail) {
    id<MTLDevice> dev = data.device;
    id<MTLBuffer> res = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    memset([res contents],0,64);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:data offset:0 atIndex:0];
    [enc setBuffer:res offset:0 atIndex:1];
    [enc setBytes:&len_u32 length:sizeof(uint32_t) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("    !! GPU status=%ld error=%s\n",(long)cb.status,
               cb.error.localizedDescription.UTF8String ?: "(null)");
        return false;
    }
    volatile uint32_t* r = (volatile uint32_t*)[res contents];
    *head = r[0]; *tail = r[1];
    return true;
}

static void fill(uint32_t* p, size_t bytes, uint32_t head, uint32_t tail) {
    size_t n = bytes/4;
    for (size_t i=0;i<n;i++) p[i] = (uint32_t)(i*2654435761u + head); // 非零、不週期抵消
    p[0] = head; p[n-1] = tail;
}

// 一個 region 完整生命週期。mode: NOMETAL / DONTNEED / MADV_FREE / FREEREUS / PURGEABLE
static int run_case(id<MTLDevice> dev, id<MTLCommandQueue> q,
                    id<MTLComputePipelineState> pso, size_t bytes, const char* mode) {
    int fails = 0;
    bool use_metal = (strcmp(mode,"NOMETAL")!=0);
    const uint32_t AH=0xAAA11111u, AT=0xAAA22222u, BH=0xBBB33333u, BT=0xBBB44444u;
    printf("\n================ CASE %s  (%.1f MiB, %s) ================\n",
           mode, bytes/1048576.0, use_metal?"GPU 持有 MTLBuffer":"無 Metal 對照");

    void* data=nullptr;
    if (vm_allocate(mach_task_self(),(vm_address_t*)&data,bytes,VM_FLAGS_ANYWHERE)!=KERN_SUCCESS){
        printf("  FAIL vm_allocate\n"); return 1; }
    fill((uint32_t*)data,bytes,AH,AT);
    id<MTLBuffer> buf=nil;
    if (use_metal) {
        buf = [dev newBufferWithBytesNoCopy:data length:bytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
        if(!buf){ printf("  FAIL NoCopy nil\n"); vm_deallocate(mach_task_self(),(vm_address_t)data,bytes); return 1; }
    }
    uint32_t len_u32=(uint32_t)(bytes/4);
    stat_line("初始", region_stat(data));

    // 基線：host / GPU 都應讀到 A 頭尾
    uint32_t hh=((uint32_t*)data)[0], ht=((uint32_t*)data)[len_u32-1];
    uint32_t gh=0,gt=0; bool gok=true;
    if (use_metal) { gok=gpu_probe(q,pso,buf,len_u32,&gh,&gt); }
    bool base_ok = (hh==AH && ht==AT) && (!use_metal || (gok && gh==AH && gt==AT));
    printf("  基線 host=(0x%08x,0x%08x) gpu=(0x%08x,0x%08x) %s\n",
           hh,ht,gh,gt, base_ok?"PASS":"FAIL");
    if(!base_ok) fails++;

    // 回收（GPU 仍持有 buf）。macOS 上：DONTNEED 對匿名頁近乎無動作（惰性 hint）；
    // MADV_FREE 去髒；MADV_FREE_REUSABLE 立即計 reusable（libc malloc 歸還路徑）。
    int adv=0;
    int64_t phys_before=phys_mib();
    if (!strcmp(mode,"DONTNEED")||!strcmp(mode,"NOMETAL")) adv=madvise(data,bytes,MADV_DONTNEED);
    else if (!strcmp(mode,"MADV_FREE")) adv=madvise(data,bytes,MADV_FREE);
    else if (!strcmp(mode,"FREEREUS")) adv=madvise(data,bytes,MADV_FREE_REUSABLE);
    else if (!strcmp(mode,"PURGEABLE")) {
        MTLPurgeableState s=[buf setPurgeableState:MTLPurgeableStateVolatile];
        printf("    setPurgeable Volatile prev=%lu\n",(unsigned long)s);
    }
    usleep(50000);
    int64_t phys_after=phys_mib();
    printf("  回收調用=%d  phys_footprint %lld→%lld（%+lldMiB）\n",
           adv, phys_before, phys_after, phys_after-phys_before);
    stat_line("回收後", region_stat(data));

    // Q2 host 讀頭
    uint32_t ah=((uint32_t*)data)[0];
    printf("  Q2 host 回收後讀頭=0x%08x %s\n", ah, ah==0?"（零頁：已丟）":"（非零：頁未丟/懶回收/重故障）");

    // Q3 GPU 再讀（危險：觀察是否 hang / 讀零 / 讀舊）
    if (use_metal) {
        uint32_t xh=0,xt=0;
        bool ok=gpu_probe(q,pso,buf,len_u32,&xh,&xt);
        printf("  Q3 GPU 回收後讀 gpu_ok=%d head=0x%08x tail=0x%08x（0=讀零/丟；A=未丟；其他=髒）\n",
               ok,xh,xt);
        if(!ok){ printf("  FAIL GPU hang/error —— 引擎內不可直接對 L4 做此動作\n"); fails++; }
    }

    // Q4 host 全覆蓋重填 B，GPU 再讀必須 bit-exact
    if (!strcmp(mode,"FREEREUS")) {
        // 重新寫回前先撤銷 reusable 標記（若頁已被回收，這裡是零頁，fill 會全覆蓋）
        int rc2=madvise(data,bytes,MADV_FREE_REUSE);
        printf("    重填前 MADV_FREE_REUSE rc=%d\n", rc2);
    }
    fill((uint32_t*)data,bytes,BH,BT);
    if (!strcmp(mode,"PURGEABLE")) {
        MTLPurgeableState s=[buf setPurgeableState:MTLPurgeableStateNonVolatile];
        printf("    setPurgeable NonVolatile prev=%lu（已重填）\n",(unsigned long)s);
    }
    stat_line("重填後", region_stat(data));
    uint32_t bh=((uint32_t*)data)[0], bt=((uint32_t*)data)[len_u32-1];
    if (use_metal) {
        uint32_t gh2=0,gt2=0; bool ok2=gpu_probe(q,pso,buf,len_u32,&gh2,&gt2);
        bool exact = ok2 && gh2==BH && gt2==BT && bh==BH && bt==BT;
        printf("  Q4 host=(0x%08x,0x%08x) gpu=(0x%08x,0x%08x) %s\n",
               bh,bt,gh2,gt2, exact?"PASS bit-exact":"FAIL");
        if(!exact) fails++;
    } else {
        printf("  Q4 host 重填後=(0x%08x,0x%08x) %s\n", bh,bt, (bh==BH&&bt==BT)?"PASS":"FAIL");
        if(!(bh==BH&&bt==BT)) fails++;
    }

    if(use_metal) buf=nil;
    vm_deallocate(mach_task_self(),(vm_address_t)data,bytes);
    printf("  CASE %s：%s（fails=%d）\n", mode, fails==0?"PASS":"FAIL", fails);
    return fails;
}

// RELEASE case：模擬「slab 動態 create/release」正解。
// evict 時先釋放 MTLBuffer（解除 GPU wire），再 madvise FREEREUS；refill 時 REUSE + 全覆蓋，
// 再用同一 host 位址重新 newBufferWithBytesNoCopy，GPU 讀必須 bit-exact。
// 若此 case reusable 升、重建後 bit-exact、無 hang，則 L4 歸還的唯一可行形態是
// 「slot/slab 必須能解除 GPU 引用（釋放 buffer）後再回收」，而非在 GPU 持有的大 buffer 上 madvise。
static int run_case_release(id<MTLDevice> dev, id<MTLCommandQueue> q,
                            id<MTLComputePipelineState> pso, size_t bytes) {
    int fails=0;
    const uint32_t AH=0xAAA11111u, AT=0xAAA22222u, BH=0xBBB33333u, BT=0xBBB44444u;
    uint32_t len_u32=(uint32_t)(bytes/4);
    printf("\n================ CASE RELEASE-THEN-REUSE (%.1f MiB, 動態 slab) ================\n", bytes/1048576.0);

    void* data=nullptr;
    vm_allocate(mach_task_self(),(vm_address_t*)&data,bytes,VM_FLAGS_ANYWHERE);
    fill((uint32_t*)data,bytes,AH,AT);
    {
        @autoreleasepool {
            id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:data length:bytes
                                                      options:MTLResourceStorageModeShared deallocator:nil];
            uint32_t gh=0,gt=0;
            bool ok=gpu_probe(q,pso,buf,len_u32,&gh,&gt);
            printf("  初始 GPU 讀=(0x%08x,0x%08x) %s\n",gh,gt,(ok&&gh==AH&&gt==AT)?"PASS":"FAIL");
            if(!(ok&&gh==AH&&gt==AT)) fails++;
            buf=nil;
        } // pool drain：強制 MTLBuffer 同步 dealloc，拆除 GPU/IOAccelerator 映射
    }
    stat_line("釋放 buffer(pool drain) 後", region_stat(data));
    int rc=madvise(data,bytes,MADV_FREE_REUSABLE);
    printf("  madvise FREEREUS rc=%d\n",rc);
    stat_line("evict：釋放buffer+reusable", region_stat(data));

    // refill：撤銷 reusable → 全覆蓋重填 → 用同位址重新包 MTLBuffer → GPU 讀
    int rc2=madvise(data,bytes,MADV_FREE_REUSE);
    printf("  madvise REUSE rc=%d\n",rc2);
    fill((uint32_t*)data,bytes,BH,BT);
    {
        @autoreleasepool {
            id<MTLBuffer> buf2 = [dev newBufferWithBytesNoCopy:data length:bytes
                                                        options:MTLResourceStorageModeShared deallocator:nil];
            if(!buf2){ printf("  FAIL 重建 NoCopy buffer=nil\n"); fails++; }
            else {
                uint32_t gh2=0,gt2=0;
                bool ok2=gpu_probe(q,pso,buf2,len_u32,&gh2,&gt2);
                bool exact=ok2&&gh2==BH&&gt2==BT;
                printf("  refill 重建後 GPU 讀=(0x%08x,0x%08x) %s\n",gh2,gt2,exact?"PASS bit-exact":"FAIL");
                if(!exact) fails++;
            }
        }
    }
    stat_line("refill 重填後", region_stat(data));
    vm_deallocate(mach_task_self(),(vm_address_t)data,bytes);
    printf("  CASE RELEASE-THEN-REUSE：%s（fails=%d）\n", fails==0?"PASS":"FAIL", fails);
    return fails;
}

int main(int argc,char**argv){
    size_t mib=(argc>1)?(size_t)atol(argv[1]):128;
    size_t bytes=mib*1024*1024;
    @autoreleasepool {
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
        if(!dev){printf("FAIL 無 Metal\n");return 2;}
        printf("device=%s recommendedMaxWorkingSet=%.0fMiB page=%ldKiB\n",
               dev.name.UTF8String, dev.recommendedMaxWorkingSetSize/1048576.0,
               sysconf(_SC_PAGESIZE)/1024);
        NSError* err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithSource:kKernelSrc options:nil error:&err];
        id<MTLFunction> fn=[lib newFunctionWithName:@"probe"];
        id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:fn error:&err];
        if(!pso){printf("FAIL PSO %s\n",err.localizedDescription.UTF8String);return 2;}
        id<MTLCommandQueue> q=[dev newCommandQueue];
        int total=0;
        total+=run_case(dev,q,pso,bytes,"NOMETAL");
        total+=run_case(dev,q,pso,bytes,"DONTNEED");
        total+=run_case(dev,q,pso,bytes,"MADV_FREE");
        total+=run_case(dev,q,pso,bytes,"FREEREUS");
        total+=run_case(dev,q,pso,bytes,"PURGEABLE");
        total+=run_case_release(dev,q,pso,bytes);
        printf("\n================ 總計 fails=%d ================\n",total);
        printf("macOS 語義：DONTNEED 對匿名頁近乎無動作（resident/dirty/head 都不變）。\n");
        printf("關鍵：GPU 持有 NoCopy buffer 時 FREEREUS 無效（reusable=0）；唯一形態見 RELEASE-THEN-REUSE。\n");
        return total==0?0:1;
    }
}
