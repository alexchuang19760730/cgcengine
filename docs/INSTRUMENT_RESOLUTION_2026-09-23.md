# 儀器：同一個 config 的 1.79× 散開是盒子的熱/功率狀態 —— 以及它擋住了什麼 2026-09-23 01:00–01:41

`docs/P0_ARMS_2026-09-23.md` 量到兩個 P0 項目都是 ≤3% 的軸，而當時的單對解析度是 ±2.8 t/s（25%）。
這份文件是把那個「噪音太大」變成機制與規則的那一輪：**三段序列、共 11 個逐位元相同的臂**，
加上三個候選見證與一組跨軸配對。

---

## 0. 一句話

同一個 config（輸出**逐位元相同**）在 40 分鐘內讀到 **5.84 → 12.35 t/s（2.1×）**；三個便宜的見證
（CPU 時鐘、memcpy 頻寬、單 op GPU）**都無法排序它**，**prefill 憑證也不行**（兩對方向相反），
**60 秒冷卻也不行**；而 `thermal_pressure.py` 的**啟動當下 level** 分得開極端（NOMINAL 12.00/12.35
vs HEAVY 10.10），代價 2 ms —— 所以修法是「蓋章 + 不要在 HEAVY 量 + 複製 n≥3 並報 sd」，不是「找一個
能預測 decode 的見證」。

---

## 1. 三段序列

### 1.1 六臂 null 序列（01:00–01:14）：同 config 就是漂移曲線本身

六個 arm 全是 `ctl`（carrier nail、MTP on、k=3、prod25、8 GiB），每一臂前後各一組見證：

| arm | cpu_s(in/out) | mem_gbs(in/out) | gpu_us(in/out) | decode t/s | accept |
|---:|---|---|---|---:|---|
| 1 | 0.546 / 0.492 | 78.7 / 76.6 | 55.1 / 73.8 | **10.46** | 180/309 |
| 2 | 0.675 / 0.686 | 44.0 / 61.5 | 88.9 / 110.5 | **5.84** | 180/309 |
| 3 | 0.821 / 0.565 | 64.9 / 78.6 | 133.8 / 82.3 | **8.79** | 180/309 |
| 4 | 0.712 / 0.539 | 43.6 / 77.7 | 76.0 / 80.8 | **8.93** | 180/309 |
| 5 (60 s 冷卻) | 0.551 / 0.552 | 76.7 / 74.8 | 65.1 / 86.8 | **8.05** | 180/309 |
| 6 (60 s 冷卻) | 0.552 / 0.516 | 76.0 / 77.0 | 68.9 / 91.5 | **7.69** | 180/309 |

**accept 六臂逐位元相同**（180/309）⇒ 這六個數字描述同一個計算，差異 100% 是儀器。
`min 5.84 / max 10.46 / median 8.42 / 1.79×`，sd = **1.54 t/s（18.5%）**。

### 1.2 見證：三個都無法排序（Spearman, n=6）

| 見證 | in | out | mean |
|---|---:|---:|---:|
| CPU 時鐘（min-of-3，~0.55 s） | −0.14 | −0.60 | −0.31 |
| memcpy 頻寬（max-of-3，256 MiB） | +0.26 | +0.49 | +0.43 |
| GPU 單 op（`MUL_MAT_ID` decode 形狀，57 µs 級） | −0.37 | **−1.00** | −0.54 |

n=6 下 |ρ| 需 ≈0.83 才顯著 ⇒ **全部不顯著**。唯一完美的是 `gpu_out`（ρ=−1.00，兩側 p≈0.003），
但它是**臂結束後**才取的（post-hoc，不能當閘門），而且這一輪一共算了 9 個相關（多重比較下
family-wise ≈2.5%）⇒ **列為候選，不是判準**。更強的反證在下一句：

> **冷卻後那兩臂（見證最好：cpu 0.551 / mem 76.7 / gpu 65.1）是六臂裡第二、第三慢的（8.05 / 7.69）。**

⇒ 「用微基準認證盒子」這條路，在被實作之前就先被否證了（這正是這個 repo 的紀律：

先寫判準再寫東西）。**見證不是閘門。**

### 1.3 跨軸兩對（01:23–01:30）：prefill 憑證也不追 decode

`window_sentinel` 量的是 prefill 軸（健康帶 median 281.51、floor 0.85×=239.28 t/s）：

| 對 | prefill（含 thermal stamp） | decode |
|---|---:|---:|
| 1 | **193.63** t/s → DEGRADED，thermal **MODERATE** | **6.26** |
| 2 | **140.60** t/s → DEGRADED，thermal **MODERATE** | **9.26** |

prefill 掉了 27%，decode 卻**升** 48% —— **方向相反** ⇒ prefill 憑證不能當 decode 的窗口證明
（兩對都是 MODERATE，所以 level 在這個 n 下也分不開 6.26 與 9.26）。

### 1.4 熱閘序列（01:38–01:41）：level 分得開極端

盒子在 01:37 冷回 **level 0**，於是同一 config 連跑三臂、每臂蓋上啟動當下的 stamp：

| arm | level at launch | decode t/s |
|---:|---|---:|
| 1 | **0 NOMINAL** | **12.00** |
| 2 | **0 → 1** | **12.35** |
| 3 | **2 HEAVY** | **10.10** |

三臂 accept 仍逐位元相同（180/309）。**HEAVY 那一臂比它的兩個 NOMINAL 鄰居慢 18%**，而 12 分鐘前的
同 config 讀 6.26 ⇒ **5.84–12.35 這條 2.1× 的帶子裡，至少兩端是可歸因的**：一端是冷盒（12.0–12.35，
兩次重現），另一端是持續負載後的熱盒。

`thermal_pressure.py` 自己記的 prefill 分離（零重疊）是：level 0 → 6/6 跑 ≥250 t/s；
level 1–2 → 0/21（104.88–211.65）。**decode 沒有那麼乾淨**（MODERATE 涵蓋 6.26 與 9.26），
因為 level 有遲滯：它反映的是**取樣那一刻**，而一條臂是 70 秒的積分。

---

## 2. 統計：配對在這台機器上買不到東西

用 P0 那兩對 true-diff-zero 的配對差（+2.82 / −0.40 t/s）算出的配對 sd ≈ **2.28 t/s**，
而獨立臂預測 √2 × 1.54 = **2.18 t/s** —— 兩者相同 ⇒ **配對（AB/BA 交錯）沒有共同模態可扣**。

```
單臂 sd                      1.54 t/s  (18.5% of 8.42)
配對 sd                      2.28 t/s  ≈ √2 × 單臂 sd
可解析效果 (2σ, n 臂)         3.08/√n t/s
   n=3 -> 1.78 (21%)   n=6 -> 1.26 (15%)   n=12 -> 0.89 (11%)
要分辨 3% (0.25 t/s)          n ≈ 151 臂 ≈ 3.4 小時連續啟動
```

⇒ **規則：任何 ≤3% 的候選，不要走 server/HTTP 這條路。**

它必須變成形狀隔離的微基準（那裡的噪音是 1–7%，例如 `shape_probe`、`test-backend-ops perf`），
或者被放棄。≥15% 的候選才值得用這條路，而且要報 sd 與 n。

---

## 3. 這一輪的正面產出（不是「全都失敗」）

1. **機制**：同 config 的散開是盒子的熱/功率狀態，而它**可讀、2 ms、免 root**，並且
   **極端是分得開的**（NOMINAL 12.00/12.35 vs HEAVY 10.10）。
2. **規則**：不要在 HEAVY 量；不要相信單臂；報 sd 與 n；≤3% 走微基準。
3. **三個真實缺陷**（都不是推測，是執行時撞到的）：

   | 缺陷 | 症狀 | 修法 |
   |---|---|---|
   | `window_sentinel` 的**記憶體前置擋住物理讀數** | 這台 16 GB 盒常態 6.9 GiB usable < 8.0 ⇒ 物理探測器**永遠不跑**（它就是 09-20 抓到 5× 慢的那個），而 launcher 對同一台盒子是放行的 | 新增 `--min-usable-gib`（預設不變），讀數以 `mem_gate=overridden` 標記 |
   | `window_sentinel --json` **raise `NameError`** | `m` 在 `main()` 裡從未被賦值 ⇒ 這支「量窗口」的工具**寫不出自己的產物**（與 `harness.py show` 的 `sw.decision()` 同一類：會 raise 而不是回答） | 組出 `mem` dict 並寫入，含 `mem_gate` |
   | 手臂產物**沒有 thermal 欄** | provenance 有 model/binary/pool/env/when，獨缺解釋散開的那個變數 | `mtp_accept_ab` 在 **launch 前**讀 `thermal_pressure.stamp()`，寫入 `provenance.thermal_at_launch`（讀不到就 `UNREADABLE`，不偽造 NOMINAL） |

4. **順手更正上一份文件**：`P0_ARMS_2026-09-23.md` 的 E 原本是從 `draft_n/draft_n_accepted`
   反推的（1.3669 / 1.7167），那是**另一組計數器**；引擎自己的定義是
   `mean len = 1 + accepted / verif_steps`（`server-context.cpp:674`），權威值是 **k2 = 2.47、
   k3 = 2.76（+12%，不是 +26%）**。已在上游文件標成取代。

---

## 4. 產物與重現

```sh
python3 /tmp/detach.py /tmp/p0w/nohup.log  bash /tmp/witness_seq.sh    # 六臂 null + 見證 + 冷卻
python3 /tmp/detach.py /tmp/p0x/nohup.log  bash /tmp/cross_axis.sh     # prefill↔decode 兩對
python3 /tmp/detach.py /tmp/p0t/nohup.log  bash /tmp/thermal_seq.sh    # 熱閘三臂
```

`Backup/instrument_resolution_2026-09-23/`：三段 driver log、`sent{1,2}.json`（含修好的 window
block 與 thermal stamp）、`arm{1,2,3}.json`（熱閘序列，含 `thermal_at_launch`）、
六臂序列的 witness 原始行。

### 誠實欄

- **見證是自己寫的 C 小程式**（`/tmp/witness.c`、`/tmp/membw.c`），不是樹上的工具：CPU 取 min-of-3
  （干擾只會加時間）、memcpy 取 max-of-3。它們的**自身重複性**是量過的（memcpy 三次 76.53/77.81/77.10
  = 1.7%）。
- 六臂序列**沒有蓋 thermal stamp**（那時欄位還不存在）—— 這是為什麼 1.4 那三臂要重跑：它是唯一
  「level 與 decode 同一臂」的資料。**1.1 的六臂不能事後補章。**
- `gpu_out` 的 ρ=−1.00 是**事後**選出來的（9 個相關之一），所以只列為候選；
  要把它變成閘門需要一個**預測性**測試（用它預測下一臂），這一輪沒做。
- 跨軸只有 **2 對**，而 1.3 的結論是「方向相反」——2 對足以否證單調，不足以量化。
- decode 的絕對值仍受 regime 限制：greedy（temp 0）、短 prompt、`prod25`、8 GiB pool、
  carrier nail。與 temp 0.4 家族不可並排。

**未 commit**（`scripts/check/{window_sentinel,mtp_accept_ab}.py` 有本輪 hunk）；11 次啟動、
零殘留行程。
