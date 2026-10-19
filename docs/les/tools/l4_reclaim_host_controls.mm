// =============================================================================
// L4 冷页归还 · 階段 0 host 側 VM 語義對照（純小程序，自包含）
//
// 與 l4_reclaim_probe.mm 配套：本檔專門回答「macOS Mach VM / Metal shared 對各種
// 回收動作的底層反應」，用 vm_region_extended_info（resident / reusable / dirty /
// share_mode / tag）與 task phys_footprint 測量。
//
// 編譯：
//   clang++ -std=c++17 -fobjc-arc -O2 -framework Metal \
//     docs/les/tools/l4_reclaim_host_controls.mm -o /tmp/l4_host_controls
//   /tmp/l4_host_controls
//
// 已釘死的實測結論（Apple M4 / 2026-10-18，page=16KiB）：
//   T1 純 vm_allocate + MADV_DONTNEED：resident/dirty/head 全不變 → macOS 對匿名頁
//      DONTNEED 近乎無動作（與 Linux 立即歸零不同），不可作為回收手段。
//   T2 純 vm_allocate + MADV_FREE_REUSABLE(7)：pages_reusable 立即升、dirty 歸零
//      → 這是 macOS libc malloc 真正歸還頁的路徑。
//   T3 被 GPU kernel dispatch 過的 region，vm_region 口徑 share_mode=5、reusable 不升、
//      resident 不降（頁物理上仍作快取）；【但見 T4】region_stat 口徑會誤導，真正的
//      記憶體壓力口徑 phys_footprint 才是判據。
//   T4 以 phys_footprint 量化（嚴格 A/B 對照，見 adv_ctl）：
//      - 不 madvise 僅等待 300ms：phys 不變（IOAccelerator 不會自然釋放）；
//      - madvise FREEREUS：即使 MTLBuffer 仍持有、GPU 剛 dispatch 過，phys 立刻降
//        256MB（256MiB region 全額退出 footprint）；事後 GPU 再讀不 hang、重填 bit-exact。
//   結論：冷頁歸還的最小改動路徑 = 在現有固定大 Metal shared buffer 上，對冷 slot 頁範圍
//   調 MADV_FREE_REUSABLE（不釋放 buffer、不 vm_deallocate）；物理頁在壓力下直接回收、
//   不寫 swap，無壓力時保留為免費快取。vm_deallocate 是徹底拆除的兜底（非首選）。
//
// 推論（見 L4_COLD_PAGE_RECLAIM_PROBE 文檔）：現有 cgc_discard_pages 用的 MADV_DONTNEED
// 在 macOS 上無效，應改 FREEREUS，並把回收從 L3 擴展到 L4 pool_ext（按頁對齊 slab 粒度）。
// =============================================================================
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_region.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#ifndef MADV_FREE_REUSABLE
#define MADV_FREE_REUSABLE 7
#endif

static int64_t phys_mib(){
    task_vm_info_data_t t={}; mach_msg_type_number_t c=TASK_VM_INFO_COUNT;
    task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&t,&c);
    return (int64_t)(t.phys_footprint/1048576);
}
static void dump(const char* tag,void* p){
    mach_vm_address_t a=(mach_vm_address_t)p; mach_vm_size_t sz=0;
    vm_region_extended_info_data_t ei={}; mach_msg_type_number_t c=VM_REGION_EXTENDED_INFO_COUNT; mach_port_t o=0;
    kern_return_t kr=mach_vm_region(mach_task_self(),&a,&sz,VM_REGION_EXTENDED_INFO,(vm_region_info_t)&ei,&c,&o);
    long ps=getpagesize();
    if(kr!=KERN_SUCCESS){ printf("  %-34s [region gone kr=%d]\n",tag,kr); return; }
    printf("  %-34s tag=%u share=%u resident=%.0fMB reusable=%u dirty=%u head=%u\n",
        tag,ei.user_tag,ei.share_mode, ei.pages_resident*ps/1048576.0,
        ei.pages_reusable,ei.pages_dirtied,*(unsigned*)p);
}

int main(){
    size_t n=64*1024*1024;
    printf("page=%dKiB\n",getpagesize()/1024);

    // T1 純 vm_allocate + DONTNEED
    { void* p=0; vm_allocate(mach_task_self(),(vm_address_t*)&p,n,VM_FLAGS_ANYWHERE);
      memset(p,0x77,n);
      printf("\nT1 純 vm_allocate + MADV_DONTNEED\n");
      dump("fill後",p); madvise(p,n,MADV_DONTNEED); dump("DONTNEED後(應不變)",p);
      vm_deallocate(mach_task_self(),(vm_address_t)p,n); }

    // T2 純 vm_allocate + FREEREUS
    { void* p=0; vm_allocate(mach_task_self(),(vm_address_t*)&p,n,VM_FLAGS_ANYWHERE);
      memset(p,0x77,n);
      printf("\nT2 純 vm_allocate + MADV_FREE_REUSABLE\n");
      dump("fill後",p); madvise(p,n,MADV_FREE_REUSABLE); dump("FREEREUS後(reusable應升)",p);
      vm_deallocate(mach_task_self(),(vm_address_t)p,n); }

    // T3/T4 GPU dispatch 過的 region
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    printf("\nMetal device=%s recommendedMaxWorkingSet=%.0fMiB\n",
           dev.name.UTF8String, dev.recommendedMaxWorkingSetSize/1048576.0);
    NSError* e=nil;
    NSString* ks=@"#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void probe(device const uint* in,device uint* out,constant uint& len,uint gid[[thread_position_in_grid]]){if(gid==0){out[0]=in[0];out[1]=in[len-1];}}\n";
    id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[[dev newLibraryWithSource:ks options:nil error:&e] newFunctionWithName:@"probe"] error:&e];
    id<MTLCommandQueue> q=[dev newCommandQueue];
    size_t n4=256*1024*1024; uint32_t len=(uint32_t)(n4/4);
    void* p=0; vm_allocate(mach_task_self(),(vm_address_t*)&p,n4,VM_FLAGS_ANYWHERE);
    memset(p,0x77,n4);
    int64_t f_live=0, f_drain=0, f_adv=0, f_dealloc=0;
    @autoreleasepool {
        id<MTLBuffer> b=[dev newBufferWithBytesNoCopy:p length:n4 options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> r=[dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState:pso];[en setBuffer:b offset:0 atIndex:0];[en setBuffer:r offset:0 atIndex:1];[en setBytes:&len length:4 atIndex:2];
        [en dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [en endEncoding];[cb commit];[cb waitUntilCompleted];
        dump("T3 buffer存活+dispatch後",p);
        f_live=phys_mib();   // MTLBuffer 存活，物理頁計入 footprint
        b=nil;
    }
    f_drain=phys_mib();      // @autoreleasepool drain，MTLBuffer 已 dealloc
    printf("\nT3 GPU dispatch 過的 region（256MB）\n");
    dump("buffer drain 後",p);
    madvise(p,n4,MADV_FREE_REUSABLE);
    f_adv=phys_mib();
    dump("FREEREUS後(預期無效 share=5)",p);

    printf("\nT4 各動作物理歸還（phys_footprint，MiB）\n");
    printf("  buffer 存活=%lld  釋放buffer後=%lld  +FREEREUS後=%lld\n",f_live,f_drain,f_adv);
    vm_deallocate(mach_task_self(),(vm_address_t)p,n4);
    usleep(200000);
    f_dealloc=phys_mib();
    printf("  vm_deallocate 後=%lld\n",f_dealloc);
    printf("  歸還量：釋放buffer=%lld  FREEREUS=%lld  vm_dealloc=%lld（期望 vm_dealloc 吃下剩餘 ~256）\n",
           f_live-f_drain, f_drain-f_adv, f_adv-f_dealloc);
    return 0;
}
