# S0 實測 ／ MTP on-off 同形狀配對 A/B — 2026-09-21

承 §EN-366（「25 還有沒有機會」）的下一步。**原本要跑的「關掉 MTP 量 S0」不必跑了：
那個數字早就躺在 T2 的資料裡。** 本文記錄這個發現，以及補做的同形狀 MTP on/off 配對。

---

## 1. 發現：`prod_matrix` 的 `decode` cell 沒有 `--spec-type` ⇒ 每一個 decode 數字都是 MTP off

`scripts/check/prod_matrix.py:359` — `cell_command` 只在 `spec.get("spec")` 為真時才加
`--spec-type`。`CELLS['decode']` 沒有 `spec` 鍵 ⇒ **`decode` cell 跑的是純 decode**。
`decode-spec` 才有 `spec="draft-mtp"`，但那個 cell 被釘在 **`-b 8 -ub 8`**（M=8 隔離 cell），
所以 `decode`(512) vs `decode-spec`(8) 不是乾淨的 MTP 對比。

驗據（T2 那一輪的 `summary.json`）：`"spec_type": null`。

**MTP off ⇒ `k=0`、`mean_len=1`、`step = S0` ⇒ `S0 = 1000 / tps` [ms]。**

所以 `S0` 不是未知數，是已經量過 6 次的數（T2，llama-bench、prod25、`-d 512`、
`median(reps[1:])`）：

| 來源 | 樣本 |
|---|---|
| T2 W8 控制臂（vs W32） | 11.191 / 11.028 / 11.280 |
| T2 W8 控制臂（vs W4） | 11.149 / 11.303 / 8.360（末筆 worst=MODERATE） |

⇒ **`S0 ≈ 89–93 ms`**。

---

## 2. 補做的配對：`decode` vs 注入的 `decode-mtp`（同一個 `-b 512`）

Driver：`Backup/phase_decomp/spec_onoff_ab.py`。做法是 monkey-patch
`prod_matrix.CELLS['decode-mtp'] = dict(p=0, n=128, d=512, batch="512", spec="draft-mtp")`
⇒ 兩條命令**只差 `--spec-type draft-mtp --spec-draft-n-max 3`**（已 `--dry-run` 驗證），
其餘（`-b/-ub 512`、`-d 512`、`-n 128`、`--reps 3`、env 全部）完全相同。

env 兩臂都釘：`CGC_PREFILL_STREAM=1; CGC_GATHER_SLAB_CAP=256; CGC_SERVER_EXPERT_CACHE_BYTES=8589934592`
（與 T2 一致，好讓 OFF 臂能直接跟既有 6 筆比）。
build witness 前後一致（`llama-bench` / `libllama` / `libggml-base` sha256 前 16 位不變）。

### 結果

| pair | OFF（`decode`） | ON（`decode-mtp`） | ratio ON/OFF | 熱 |
|---|---|---|---|---|
| 1 | 10.633 | 9.454 | 0.889 | NOMINAL / NOMINAL |
| 2 | 11.211 | 8.395 | 0.749 | HEAVY / HEAVY（ON worst=HEAVY） |
| 3 | 10.719 | 12.481 | 1.164 | HEAVY / HEAVY |

- **OFF 中位 10.719 ⇒ `S0 = 93.3 ms`**（合併 T2 的 6 筆 ⇒ 9 筆落在 **10.6–11.3**）
- **ON 中位 9.454；中位比值 0.889**
- ⚠️ ON 臂變異極大（樣本 6.65→17.28→7.68），pair 2–3 熱標 HEAVY ⇒ **比值只可定性讀**

**判準（預先寫死）**：`ratio ≤ 0.95 ⇒ MTP 在此 shape 是淨成本** ⇒ 命中。**

### ⚠️ 一條必要的更正

可引用交付 **12.57 t/s** 是 **MTP on ＋ `--warm-skip 64` ＋ `--ctx-size`**。
本次在同一 shape 下把 MTP 單獨打開是**淨負**（−11%）。
⇒ **12.57 比 11.2 高的那一截，主要不是 MTP 的功勞，而是 warm-skip／ctx 那兩個形狀參數。**
別把「MTP 有 +12%」當成既有結論。

---

## 3. 假說檢定：MTP 的代價從哪裡來（池計數器實測）

`summary.json` 的 `cache` 區塊在 **llama-bench 也有**（不是 server 專屬），這是本次的實質收穫：

| | OFF（MTP 關） | ON（MTP 開） | 變化 |
|---|---|---|---|
| **池命中率** | **96.5–96.6%** | **57.5–62.1%** | **−36 pp** |
| 每次 run 讀的 bytes | 4.87 GB | 7.63–7.98 GB | **+64%** |
| io_jobs | 26,883 | 34,686–35,691 | +33% |
| **bytes / job** | **177 KiB** | **215–218 KiB** | **+23%（變大）** |
| resident | 6154 MiB | 6431 MiB | +4.5% |

**⇒ 「每 step 專家 union 變寬」是真的**：bytes +64%、命中率崩 36 pp。

**⇒ 但它不是 `-np 4` 那種 per-job 碎裂**：bytes/job 反而從 177 KiB **升到** 218 KiB。
（`-np 4` 是 133 KiB → 257 B，方向相反。）所以 §EN-364 的「per-job overhead」機制
**不能直接套到 MTP 身上**。MTP 的代價是 **pool capacity thrash**（union 變寬 ⇒ 工作集撐爆池 ⇒
命中率崩），不是 job 粒度變細。

**⇒ 而且 miss 仍然是弱驅動項**（第二次獨立證據）：miss 率從 3.4% 漲到約 39%
（**11×**），只換來 **−11%** 的 t/s。與 §EN-358「4 GiB 池 miss 1.91× 而 t/s 持平」同向。

---

## 4. 對 25 的判決：MTP 軸連「槓桿」都不是

這是本次最重要的結論，而且它把 §EN-366 的問題從「m 從哪來」降級成無關：

**把 MTP 整個關掉，單流仍然只有 10.6–11.3 t/s（`S0 = 89–93 ms`）。**

- MTP off ⇒ `t/s = 1000 / S0` ⇒ **25 需要 `S0 ≤ 40 ms`，實測 93.3 ⇒ 要砍 57%。**
- §EN-366 寫的「凍結 a、k 時 `S0 ≤ 43.1 ms`」現在有了實測對照：**93.3 ms，差 2.2×**。
- 就算 MTP 完美無代價（`m → 0`），上限也只有 §EN-366 算的 30.4 t/s；而實測 MTP 是淨負。
- **8.8× 的開銷在 `S0` 本身，不在 `m`。** 找 `m` 的來源救不了 25。

已排除的 `S0` 成分（累計）：miss、池大小、IO worker 數、dispatch 數（G4）、MTP 邊際成本。
**未歸因，也沒有可執行手段。**

---

## 5. 產物

- `Backup/phase_decomp/spec_onoff_ab.py` — 注入 `decode-mtp` cell 的配對 driver（判準寫在檔頭）。
- `Backup/phase_decomp/spec_onoff/spec_onoff.json` — 原始資料（含兩臂的 `cache` 區塊）。
- `Backup/phase_decomp/spec_onoff/run.log` — 逐臂輸出（未接管道，避免 §EN-364 的緩衝事故）。
- 單臂 summary：`Backup/prod_matrix/*_prod25_decode-mtp/summary.json`（3 筆）。

---

## 6. 下一步（如果要繼續追 `S0`）

`cache` 區塊在 llama-bench 就能讀 ⇒ **不必再走 HTTP 就能做池診斷**，這是本次新開的便宜通道。
下一步最有資訊量的一件事是把 `S0` 拆成「dense 權重的搬運」與「其餘」：
`io_bytes/token` 只有 **12.7 MB**，命中率 96.6% ⇒ 專家幾乎都從 RAM 來；
那 93.3 ms 裡到底誰在花時間，目前完全沒有儀器指到它。
