# L4 專家池匿名頁不可回收根因判詞（常駐 server 多主題必然超訂）

> **日期**：2026-10-18　**機器**：M4 Air 16GB（統一記憶體 108.8 GB/s）
> **模型**：`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（13 GB、3.06 bpw、n_params 35.5B、L=41 路由層 40、E=256 top-8）
> **分支**：`demo/sweet-spot-windows-fix`（HEAD `58abe27ad`，commit2 六檔 staged 未提交）
> **配置**：`prod-new`、`-expert-cache 8589934592`（8 GiB）、`--ngl 99 --load-mode none`、`CGC_EXPERT_SKIP_READRAW=1`、`CGC_POOL_MADVISE=2`、`CGC_OVERLAP_FENCE=41`、MTP off
> **結論等級**：根因（有進程級 vmmap 實證 + 源碼路徑）；**不是**預熱、**不是** ubatch、**不是** compute buffer

---

## 0. 一句話

**L4 expert pool 的實體頁是 Metal shared 匿名 dirty 記憶體（`vm_allocate` + `newBufferWithBytesNoCopy:storageModeShared`），不是 gguf 檔案 mmap 頁；常駐 server 連跑 7 個主題把 8 GiB 池 5720 槽幾乎全 fill，evict 後程式碼也不對這些頁 `madvise`（歸還路徑只覆蓋 L3 malloc pool、不覆蓋 L4 metal pool），冷專家頁內核無法回收、只能 swap —— 進程獨佔髒足跡實測 ≈ 12.9 GiB（resident 8.1 + swapped 4.8），撞穿 11453 MiB Metal working set 天花板 ⇒ free 0、swap 4.1–4.5 GiB、coding 提前 EOS、全 profile tier D。**

這與 C17（20.79 t/s、free 86%、swap 零增長）不矛盾：C17 是**單 prompt 窄工作集**，常駐 replay 是 **7 profile 寬工作集劇烈 churn**，兩者的池 fill 面完全不同。（口徑註：C17 來源 `attribution=oversubscribed`，是「配置名義超訂但窄集實測 swap 零增長」的對比值，**不可引用**為乾淨基線；見 §7。）

---

## 1. 起因：commit2 baseline 三次 FAIL，逐層剝假設

為讓 commit2 六檔過 replay gate 建健康 baseline，連續三輪（重開機 swap=0）：

| 輪次 | 假設 / 修正 | 結果 | 裁定 |
|---|---|---|---|
| v1 `steady_baseline_20261018_200336` | 8 主題×3 輪多樣化預熱，池反覆 fill/evict | prefill 7–12、coding 0.6、swap 6230 MB、Writable 21.5G | FAIL |
| v2 `steady_baseline_20261018_204104` | 刪光多樣化預熱、只留 1 個活性請求、補 `FENCE=41`、前後健康門 | 仍 7 profile 全 D、free 0.06GB、swap 2310、coding 0.6/1 token | FAIL，**「多樣化預熱」假設證偽** |
| ub512 `steady_ub512_20261018_205300` | 對齊 C17 真實 `-b 2048 -ub 512`（C17 標量 5632 但 rows 是 512），驗證「大 ubatch compute buffer 多吃 2–3GB」 | peak RSS 8805 vs 8902（**僅省 97 MB**）、仍全 D、free 0、swap 4119 | FAIL，**「ubatch/compute buffer」假設證偽** |

三輪共同的不變量：**活性單請求正常（「1+1 等於 2。」7 token、啟動前 free 87%、swap 0），但 replay 7 profile 跑完必然 free 0、swap 數 GB、coding 退化。** 指向「隨主題數增長的池記憶體」，而非任何固定大小的開銷。

---

## 2. 鐵證：vmmap 進程記憶體解剖（ub512，PID 4962）

完整 dump：`Backup/steady_ub512_20261018_205300/vmmap_ub512.txt`（3.3 MB，已存證）。

Writable regions 總計：**Total ≈ 15.2G、resident 8.1G、swapped_out 4.8G**。按可回收性彙總：

| 記憶體區域 | 類型 | vsize (MB) | resident (MB) | swapped (MB) | 含義 |
|---|---|---:|---:|---:|---|
| `SM=SHM`（64 個 ~128MB 塊） | GPU 共享 | 8359 | **8236** | 0 | 當前活躍 GPU 工作集（dense + 當前專家 + KV + compute/slab） |
| **`MallocHelperZone`** | 匿名私有 | 6185 | **0** | **4552** | **曾 fill、已被 evict 的冷專家池頁**，全在 swap |
| `VM_ALLOCATE` | 匿名私有 | 416 | 0 | 402 | ggml `vm_allocate` 其餘開銷 |
| `unused` / `__TEXT` 等 | — | 3875 | 203 | 0 | 預留/程式碼 |

**進程獨佔髒足跡（不可被系統回收的匿名資料）≈ 8.24 + 4.55 + 0.40 ≈ 13.2 GiB。**

`MallocHelperZone` 內部結構（決定性）：

- `MALLOC_LARGE`：vsize 4045 MB / swapped 3593 MB，**104 個均勻的 65–68 MB 大塊**
- `MALLOC_LARGE_REUSABLE`：vsize 898 MB / swapped 889 MB
- 每個 65–68 MB 大塊 = **一層（按 kind 切分）的專家池連續片**；pool 幾何實測 8 GiB = 40 層 × 143 槽 = 5720 槽、每槽 1.4325 MiB
- 析構行印證：`pool integrity: owner-set slots=5720`（**40×143 槽全部曾被 fill**）、`slab fills: pool=148919.5 MiB disk=117336.5 MiB (non-resident share 44.1%)`、`capacity miss=21204 evict=30170 hit_pct=95.86`

> 全 src 樹 grep `malloc_zone_create` **無任何命中** ⇒ `MallocHelperZone` 不是我們程式碼顯式建立，是 Apple Metal/IOAccelerator 在包裝 shared host buffer 時使用的內部 zone。65–68MB 均勻步長 + 5720 槽全滿 + 與 gather slab 位址（`0x64xxxxxx`–`0x66xxxxxx`）緊鄰，鎖定其身份就是 **L4 expert pool 的實體存儲**。

對比 C17 窄集實測（`Backup/aff_cert_20261013/aff_off_single.json`，單 prompt p512 固定 seed；注意 attribution=oversubscribed，是名義超訂但窄集 swap 零增長的對比、非乾淨基線）：

| | C17（單 prompt 窄集） | 常駐 7 profile（寬集 churn） |
|---|---|---|
| cache resident | 6197 MiB | RSS 8.2–9.1G（峰 9087） |
| swap（launch→end） | 251 MB、**增長 0** | 2310–4119 MB（free 0） |
| 專家工作集 | 單主題、窄、evict 極少 | 7 主題並集 > 143 槽、evict 30170 |
| 冷專家頁（swap） | 幾乎不產生 | 4552 MB |
| decode | 20.79 t/s（oversubscribed，**不可引用**）、free 86%→83% | 15–24 抖動、run2/3 掉 13–15、coding 崩 |

---

## 3. 源碼根因（三處串成因果鏈）

### 3.1 L4 pool 是 Metal shared 匿名記憶體，不是檔案頁

- Metal shared buffer 的 host 存儲由 `ggml_metal_host_malloc` 分配：
  `ggml/src/ggml-metal/ggml-metal-device.m:1643`
  ```objc
  static void * ggml_metal_host_malloc(size_t n) {
      void * data = NULL;
  #if TARGET_OS_OSX
      kern_return_t err = vm_allocate((vm_map_t) mach_task_self(),
                                     (void *) &data, n, VM_FLAGS_ANYWHERE);   // ← 匿名 dirty
  ...
  ```
- 再以零拷貝包裝給 Metal：同檔 `:1699-1702`
  ```objc
  if (props_dev->use_shared_buffers && shared) {   // unified memory 上 use_shared_buffers=true（:902）
      res->buffers[0].metal = [dev->mtl_device
          newBufferWithBytesNoCopy:res->all_data length:size_aligned
          options:MTLResourceStorageModeShared deallocator:nil];
  ```
- L4 pool 用 expert tensor 的 Metal buft 分配：`src/llama.cpp:524` → `llama_expert_cache_pool_split_alloc`（`llama-expert-cache.cpp:2314`）→ `ggml_backend_buft_alloc_buffer(buft, total)`，即上面的 shared 匿名路徑。
- 啟動印 `L4 metal pool: 143 slots/layer, regions adopted from expert tensors`（`llama-expert-cache.cpp:4447-4450`）。

**`vm_allocate` 出來的匿名頁被 pread 寫入專家權重後成為 dirty：無檔案後備，內核不能丟棄，記憶體壓力下只能壓縮或 swap。**

### 3.2 evict 後的頁歸還路徑漏掉 L4 metal pool

- 唯一的頁丟棄函式 `cgc_discard_pages`（`llama-expert-cache.cpp:948`，內部 `madvise(MADV_DONTNEED)`，:960）。
- 它在 fill 路徑只對「落在 malloc pool 內」的指標生效（`:4042-4047`）：
  ```cpp
  if (cgc_pool_madvise_mode() >= 1) {
      for (size_t i = 0; i < n; ++i) {
          if (cgc_in_malloc_pool(cache, dsts[i])) {      // ← 關鍵守衛
              cgc_discard_pages(dsts[i], segs[i].bytes);
          }
      }
  }
  ```
- `cgc_in_malloc_pool`（`:965-983`）只遍歷 `cache->pool[l][k]`（**L3 Option A 的 `std::vector<uint8_t>`**）。
- 但 L4 metal pool 走的是 `pool_ext` / `pool_owned_bufs`（`pool_split_alloc`），**不在 `cache->pool` vector 裡** ⇒ `cgc_in_malloc_pool` 對 L4 池永遠回 false ⇒ **L4 冷頁 evict 後永不 madvise、永不歸還**。即使 `CGC_POOL_MADVISE=2` 開到頂也一樣（該開關只覆蓋 L3）。

> 註：`:4042` 的 madvise 本意是「pread 全覆蓋前先丟掉恰好躺在 swap 的舊頁，避免 swap-in 後又被覆蓋」的**讀取側優化**，並非「evict 即歸還」的容量側機制。L4 兩者都沒拿到。

### 3.3 budget 口徑把「匿名不可回收池」誤當「檔案可回收」

`scripts/check/budget_preflight.py`（task#37 修過的版本）與 `budget_gate.sh` 的判定：

```
_dense 555 + _pool 8192 = 8747 MiB  vs  hw.memsize 16384  ⇒ OK（不超訂）
```

四道錯（本判詞 D 項一併修）：

1. **上限錯**：拿 `hw.memsize=16384` 當可用量；真實單 GPU 進程上限是引擎自己印的 `recommendedMaxWorkingSetSize = 11453.25 MiB`（`Backup/metal_working_set.json`，peak_wired 11029）。
2. **池可回收性錯**：註釋假設「expert CPU 零副本、L4 adopt、其餘 pread 按需」≈檔案 mmap 可回收；實際 L4 池是匿名 dirty（§3.1/3.2），**冷頁不可回收**。
3. **漏算固定開銷**：KV cache（q8_0@8192 ≈ 85 MiB，僅 10 層帶 K/V）、gather slab（110+110+136=356 MiB）、prefill stream/compute graph buffer、以及系統/GPU wired 常駐競爭。
4. **不分工作集寬窄**：單 prompt 窄集（C17）靜態帳有效；多主題寬集（常駐 7 profile）池被 churn 全滿、冷頁累積，靜態帳嚴重低估。

---

## 4. 為什麼「單 prompt 健康、7 主題崩」是結構性的

- L4 池容量固定 8 GiB（5720 槽），但**物理頁的提交由「曾被 pread 寫過的層片」決定**。單主題只觸發窄專家並集，活躍工作集 resident ≈ 6.2G（C17），其餘層片是惰性未提交頁。
- 7 主題（code/zh/chat/math/reasoning/writing/translation）專家並集寬，5720 槽全部被寫過一次（owner-set=5720），層片物理頁大面積提交；之後 capacity miss 21204、evict 30170 反覆覆寫，但**被覆寫前的舊頁因為匿名 dirty + 無 madvise 而保留**（在 swap 中仍佔額度）。
- 於是「活躍 8.2G（GPU 必須持有）+ 冷池 4.55G（只在 swap）」疊到 ~13.2G，16GB 物理（扣系統、扣 wired ceiling）裝不下，coding 這類低佔比/短回應 profile 在 D 級壓力下提前吐 EOS（comp=1）。

**這不是腳本、預熱、batch 參數能修的；是「常駐多主題工作集 × 不可回收匿名池」在 16GB 上的結構性衝突。**

---

## 5. 對各條線的影響

| 條線 | 影響 |
|---|---|
| **commit2 baseline（replay gate）** | 常駐 server 跑 7 profile 在 16GB/8G 池下**結構性超訂**，無法得到健康穩態 baseline。replay gate 應回歸「質量+不退化」門的本職，穩態權威用 harness 單 prompt、且須 attribution=none（C17 雖 swap 零增長但名義 oversubscribed，其 20.79 不可作乾淨基線，需另跑 none 臂定權威速度）；不應強迫常駐 7-profile 給速度。 |
| **B：G6 縮池 8/6/4G** | 直接削減匿名池容量上限，是近期、確定的物理槓桿；見 `docs/les/G6_POOL_SHRINK_EXPECTATION_2026-10-18.md`。 |
| **C：L4 冷頁歸還** | **根治項**：讓 evict 的 metal pool 頁真正可回收（madvise/purgeable/檔案 backing），多主題不再累積冷頁；與 LES G3 preadv、「L2 單一駐留」同源，是 MoE 引擎共有 IP。見 `docs/les/L4_COLD_PAGE_RECLAIM_PROBE_2026-10-18.md`，探針先行、不許諾收益。 |
| **D：budget 口徑** | 改用 11453 working ceiling、池按匿名不可回收計、補固定開銷、分窄/寬工作集；本判詞配套修改 `budget_preflight.py` / `budget_gate.sh`。 |
| **MTP（G5）** | 與本根因無關，維持「只驗共存、n_tokens>1 降級、bit-exact」，不當加速槓桿。 |
| decode 25 / prefill 250 | 冷頁 swap 是 43.20ms 非 kernel 等待的一部分；C 若成立可釋放物理壓力，但**不承諾**直接到 25，需 G3/G6 一手數據。 |

---

## 6. 未決項（誠實邊界，需乾淨 GPU 窗口確認，勿在當前 swap 壓力下硬跑）

1. **`SM=SHM` 8.2G 的精確歸屬**：是「dense + 當前池工作集 + KV/compute」的正常活躍集，還是模型載入時的**第二份 expert Metal 駐留**（與 L4 池構成雙副本）。當前證據（位址、塊大小）偏向「活躍工作集 + GPU 共享視圖」，但須在重開機 swap=0 下，用啟動日誌的 `CGC-POOL-SPLIT`/`GATHER-SLAB` base 位址即時對照 vmmap 才能 100% 釘死。**若為第二份副本，修掉它是比縮池更大的槓桿。**
2. **L4 層片物理頁是否隨 evict 真的不釋放**：乾淨窗口下對單一進程「窄集 fill → 寬集 fill → 回到窄集」做 vmmap 時間線（resident/swapped 對 owner-set 槽數），確認冷頁只增不減。
3. **對 vm_allocate + Metal shared buffer 做 `madvise(MADV_FREE)`/`MADV_DONTNEED` 的安全性**：GPU 是否仍透過 `newBufferWithBytesNoCopy` 持有該物理頁的引用、丟棄後下一次 pread 全覆蓋是否 bit-exact（C 探針階段 0 驗證）。
4. 6G/4G 真實 `LAYER_CAPS`（本判詞用 143 等比換算 107/72，G6 重測）。

---

## 7. 證據索引

- vmmap 完整 dump：`Backup/steady_ub512_20261018_205300/vmmap_ub512.txt`；三輪結果：`Backup/steady_baseline_20261018_{200336,204104}`、`steady_ub512_20261018_205300/steady_baseline.json`
- server log（pool 幾何 / 5720 槽 / slab fills）：`Backup/cgc_logs/llama_server_20261018_204104.log`
- C17 窄集實測對比值（**非健康基線、不可引用**，attribution=oversubscribed、swap_growth=0）：`Backup/aff_cert_20261013/aff_off_single.json`（20.790546 t/s）
- Metal 上限：`Backup/metal_working_set.json`（11453.25 MiB）
- 源碼：`ggml/src/ggml-metal/ggml-metal-device.m:1643,1699-1705,902`；`ggml/src/ggml-metal/ggml-metal.cpp:202,508`；`src/llama.cpp:524`；`src/llama-expert-cache.cpp:948-983,2301-2330,4042-4047,4436-4470`
- 關聯判詞：`docs/S_LINE_FINAL_VERDICT_2026-10-07.md`、`docs/les/G3_PREADV_WORKER_PINGPONG_PATCH.md`、charter `docs/mindmap/briefs/e-les-router-prefetch-2026-10-17.md`（G6）
