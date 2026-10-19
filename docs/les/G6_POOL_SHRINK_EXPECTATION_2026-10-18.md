# G6 縮池正確預期（8 / 6 / 4 GiB）—— 冷頁定量 × Metal ceiling × 誠實 decode 區間

> **日期**：2026-10-18　**配套判詞**：`docs/L4_POOL_ANON_NONRECLAIMABLE_VERDICT_2026-10-18.md`（根因）
> **charter**：`docs/mindmap/briefs/e-les-router-prefetch-2026-10-17.md` G6
> **腳本**：`~/les_g6_baseline_run.sh`（三池 off 權威基線）、`~/les_g6_les_abba_run.sh`（六格 off/on ABBA + 門檻）
> **本文只校正預期，不改腳本門檻**；所有 decode 數字以 G6 乾淨窗口一手數據為準，本文不預設結論。

---

## 0. 兩個目標不要混（再次釘死）

| 目標 | 正確路徑 | 本文件的答案 |
|---|---|---|
| **物理常駐 < 6 GB**（省 2 GB） | 縮池 / 冷頁歸還 | 名義上**只有 4G 池**（池 4 + fixed 1.4 ≈ 5.4G）或 6G 池 + C 冷頁歸還；**6G 池名義常駐 ≈ 7.4G，不滿足「<6GB」**，它滿足的是 charter 的「峰值降 ≥1.5GB」 |
| **decode 往 25** | LES G3 preadv 藏 SSD miss + expert kernel 利用率 + b 重疊 | 縮池本身**不是加速槓桿**，它是「省記憶體 + 消除 swap 拖累」；在 swap 嚴重的寬集，縮池可能因系統變健康而 t/s 不降反升 |

---

## 1. 池幾何（實測，8G 為準等比換算）

- 8G 實測：`pool_cap_slots=143`、`slots_layer=143`、`n_layer=40`、owner-set 5720 = 40×143
- 每槽位元組：`8192 MiB / 40 / 143 = 1.4325 MiB/slot`（一個專家在某層的跨 kind 槽）

| 池 | bytes | 槽/層（換算） | 對 8G | 名義池容量 |
|---|---|---:|---:|---:|
| 8G | 8589934592 | **143**（實測） | 100% | 8192 MiB |
| 6G | 6442450944 | **≈107**（等比，G6 重測 LAYER_CAPS） | −25% | 6144 MiB |
| 4G | 4294967296 | **≈72**（等比，G6 重測） | −50% | 4096 MiB |

---

## 2. 物理預算（Metal ceiling 11453，可用門檻 10308，v3 budget 口徑）

固定非池常駐 `fixed ≈ 1396 MiB`（dense 555 + KV q8_0 85 + gather slab 356 + compute/graph/prefill 400）。

| 池 | narrow 靜態需求=fixed+池 | narrow 餘量 | wide 需求（+2048 寬集餘量） | wide 判定 |
|---|---:|---:|---:|---|
| 8G | 9588 | 720 | **11636** | **超訂 +1328**（常駐多主題實測必然 swap，D 判詞） |
| 6G | 7540 | 2768 | 9588 | 過，餘 720（**寬集最小可行**） |
| 4G | 5492 | 4816 | 7540 | 過，餘 2768（寬集舒適） |

**物理常駐（池填滿名義上限 = fixed + 池）**：8G ≈ **9.4G**、6G ≈ **7.4G**、4G ≈ **5.4G**。

**寬集獨佔髒足跡（含 swap，粗線性外推，僅供預期、G6 校準）**：
- 8G 實測 **12.9G**（resident 8.1 + swapped 4.8，D 判詞 vmmap）。
- 線性外推：6G ≈ 10.9G、4G ≈ 8.9G。
- ⚠ 這條線假設「超額與池容量線性」，未計入未決項 SM=SHM 8.2G 若含第二份副本的可能；**不能當結論，只當風險區間**。G6 每格都要實測 `resident_mib / max_wired_mib / swap_after / free`。

---

## 3. decode 預期（機制 + 區間，不預設數字、不許諾）

舊轉述曲線 **8/6/4G = 12.06/10.48/7.67 已查證為轉述且曾受 override 語法 bug（無 SERVER_ 前綴導致兩臂都 8G）污染，作廢不可引用**。G6 用下列**機制**解讀，不背誦舊數字：

### 3.1 單 prompt / 窄工作集（harness、warm-skip64、固定 seed）

- C17 8G 權威 = **20.79 t/s**（`Backup/aff_cert_20261013/aff_off_single.json`），其單主題工作集 `worst_slots=143`、`hit 96.1%`、swap 零增長 —— 換句話說**單主題已經把 8G 池用到接近滿**。
- 6G（107 槽）：單主題若觸發並集 >107 槽即開始 capacity miss，decode 大概率**小幅下降**；若該主題並集 ≤107 則與 8G 持平。預期方向：**持平到小幅下降**，是否守住 ≥0.95 是 G6 第一道實證。
- 4G（72 槽）：單主題更容易溢出，capacity miss 明顯上升，**off 臂下降概率高**；charter 只要求「4G+on 懲罰收窄 ≥50%」，不要求 4G 追平。

### 3.2 常駐多主題 / 寬工作集（replay 7 profile）

- 8G 已實證崩（swap 4.5G、free 0、coding 0.6、tier D）——這不是有效基線，是超訂樣本。
- 6G：池容量小 → 可累積的冷匿名頁更少、swap 壓力下降，**系統健康度提升可能抵銷甚至超過 miss 增加的損失**。這正是舊線索「4G miss 1.91× 但 t/s 10.28→10.60 反略升」的合理方向（但那筆數據本身作廢，只取機制不取數字）。
- 4G：同樣受益於 swap 減少，但即時補量更大，後台 IO 可能成瓶頸（charter 風險 6）。

### 3.3 LES on 的槓桿（G1 已證命中率 82%，G3 才是速度關）

- G1 離線：target≥4 層 recall mean 0.820、recall@d8=.79/d10=.86/d12=.90/d16=.94。預測只指導預取、真實路由兜底，**逐字節一致**。
- LES 對小池的意義：**把 6G/4G 縮池後新增的 capacity miss，用下一層真實 router 預測提前 preadv 進來**，讓「縮池省的記憶體」不白付速度。
- **但 G1 是離線命中率，不產生 t/s**；decode 收益只在 G3a（preadv worker 把 SSD miss 藏進當前層 compute）落地後才成立。故 G6 六格必須在 G3a 補丁 apply 且 bit-identical 自證之後跑，否則 on/off 臂只差 G2 圖內 argsort，讀不到速度。

---

## 4. 健康判定（每格必須記錄，避免第四次無效跑）

每格（off/on × 池）除 decode/prefill 外，健康閘建議：

| 指標 | 健康（可引用） | 失效（標 D / 不引用） |
|---|---|---|
| 啟動前 / 每格後 swap used | 啟動 0、每格增長 < 512 MiB | 增長 > 1 GiB 或 free 趨近 0 |
| 單格後 `vm_stat` free | 維持 > 1.5 GiB | free < 0.3 GiB |
| `max_wired_mib` | 顯著 < 11453，留 ≥800 MiB | 頂到 11453 / Metal OOM |
| resident_mib | ≤ fixed+池名義（9.4/7.4/5.4G）+ 寬鬆 | 超出名義上限（暗示第二份副本，回頭查未決項） |
| coding profile | 正常長度、非 1 token | 提前 EOS / comp=1（D 級壓力訊號） |
| 同格 3 輪 decode | 穩態、無 run2/3 崩到 13–15 | 逐輪遞減（churn/swap 訊號） |
| bit-identical | on 臂 vs 全關逐字節一致 | 任一 token 翻臉 → 該臂無效 |

順序維持 ABBA：`8g-off → 8g-les → 6g-les → 6g-off → 4g-off → 4g-les`（小池放最後，避免最緊的池先污染）。

---

## 5. G6 正式門檻（與腳本 269-273 一致，未改）

1. **6G+LES-on decode ≥ 8G+off × 0.95**（ratio ≥ 0.95）。
2. **峰值實體佔用降 ≥ 1.5 GiB**（1536 MiB，以 `max_wired_mib` 或 resident 峰值對比）。
3. **4G+LES-on 縮池懲罰收窄 ≥ 50%**（on 相對 off 把 4G 的 decode 跌幅砍半）。
4. 全程零 churn 惡化：n_wait 不隨池縮上升、swap 不增長、on/off 兩檔 bit-exact。

**判定矩陣**：
- 6G 過門檻且峰值降 ≥1.5GB ⇒ 預設池從 8G 登記改為 6G（寬集常駐安全），同時釋放 2G 物理。
- 6G 過門檻但仍想「物理常駐 <6GB」⇒ 上 4G；4G 過門檻才登記 <6GB 配置，或轉 C 冷頁歸還。
- 6G 不過（懲罰不收斂）⇒ 封閉縮池、保留 8G 於 narrow harness，常駐寬集等 C 項根治。

---

## 6. 為什麼 G6 必須在乾淨窗口、且一次跑全六格

- v1/v2/ub512 三輪 FAIL 已證明：非乾淨窗口（swap 殘留、thermal、WorkBuddy/Chrome 佔用）產出的數字不可引用。
- 六格必須同一窗口、同一 binary、ABBA 順序，才能把「池大小」與「預取開關」兩個因子解耦；跨窗口拼圖會重蹈 8/6/4G 轉述曲線覆轍。
- 前置鏈路：commit2 落地 → 撤 affinity（8 文件，僅刪 affinity hunk、保留 context.cpp bhead/les s2c）→ 純探針編譯 → 三池 off 基線 → apply G2 → apply G3a → 編譯 → 全關 bit-identical 自證（stderr 不得有 `CGC-LES-G3`/`cgc_les_topk`）→ G3a 觀察臂 → G6 六格 → `les_g6_publish_mindmap.sh` 回寫。

---

## 7. 未決項（影響預期、G6 順手核）

1. **6G/4G 真實 LAYER_CAPS**：本文用 143 等比換算 107/72，以引擎啟動日誌實測為準。
2. **SM=SHM 8.2G 是否含第二份 expert Metal 副本**（D 判詞 §6）：若有，縮池的常駐降幅會比名義小，需先修雙副本——這是比縮池更大的槓桿。G6 每格若 resident 超名義上限即觸發此調查。
3. **wide_headroom 2048 MiB 係工程默認**：G6 六格的 swap/free 實測回來後，應校準 `budget_preflight.py` 的 `--wide-headroom-mib` 與各池健康閾，把它從經驗值變成實測值。
4. 4G 的後台 IO 是否成瓶頸（n_wait 隨池縮上升）：charter 風險 6，G6 直接量。
