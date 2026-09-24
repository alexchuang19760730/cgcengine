# 指標穩定性實測：union+gap 比 t/s 更不穩（2026-09-21）

結論先行：**`union+gap` 不是比 `t/s` 更穩的指標，它是更差的 —— 穩定性和準確性兩頭都輸。**
想要低噪音，應該用 run 級累加量 `pool_wait_us / misses`（µs/miss）。

資料來源：8 支 raw 的 `CGC-GPUTIME` 逐步行（共 286 行）＋ M-W 交付 cell 的 10 臂配對
（全部 NOMINAL、ABBA 順序翻轉、臂間冷卻 150 s、reps=3）。全程 0 重建。

---

## 1. 穩定性排名（同一交付 cell，可直接比）

| 指標 | 離散度 | 說明 |
|---|---:|---|
| **`µs/miss`（pool_wait_us/misses）** | **CV 1.95%**（w=8, n=5）／4.55%（w=2） | ✅ 最穩。run 級累加量 |
| **t/s（llama-bench avg_ts）** | **CV 7.88%**（w=8）／7.43%（w=2） | 可用，但比上面差 4× |
| `union`（含 block-mean 修正的 SEM） | **6.0–9.0% 是不加 gap 的**；union 本身 SEM 3.2–6.5% | ⚠ 見 §2 |
| **`union + gap`** | **SEM 6.0–9.0%**；逐行 CV 35–47% | ❌ 與 t/s 同級或更差 |
| **`gap` 單獨** | **逐行 CV 64–85%** | ❌ 不可用 |

⚠️ 關鍵更正：記憶裡的「單臂 t/s 噪音 ±27%」是 **K3 cell**（不同 shape、無冷卻）的值。
**交付 cell 在 reps=3 + 臂間冷卻下實測只有 7.9%** —— 差 3.4×。別混用。

## 2. 為什麼「每步很多樣本」沒幫上忙

`union+gap` 每支 run 有 37–59 行，看起來 n 很大，但樣本**強自相關**（熱漂移＋快取預熱）：

- 逐行 CV 35–47%，但用 block mean（每 5 步一塊）修正自相關後，**有效 SEM 仍有 6.0–9.0%**
  ⇒ 有效獨立樣本數只有 5–8 個，不是 37–59 個。
- **系統性漂移**：後半段 vs 前半段 −2.2% ~ **−21.6%**（8 支 run 全部為負）
  ⇒ 序列非平穩，取哪幾步平均會直接改變答案（bias，不是 noise）。

## 3. 準確性：union+gap 根本不重建步時

交付 cell `gc_s2a` 同 run 對照：

```
llama-bench avg_ts        = 12.601 t/s  →  步時 267.84 ms（ML=3.375）
GPUTIME union+gap 均值    = 127.29 ms   →  隱含  26.51 t/s
比值 步時 / (union+gap)   = 2.10 ×
```

⇒ `union+gap` 是 **GPU 時鐘跨度量**，不含主機側等待（`wait`/`cb`/`submit` 是另欄位），
**系統性低估約 2 倍**。拿它當「步時」或「天花板」會得到 26.5 t/s 這種不存在的數字。

## 4. 跨 run 更慘

8 支 run 的 `union+gap` 中位：**123.82 ~ 236.54 ms，CV 24.2%、spread 1.91×**。
即使只看**同一名義 cell**（prod25，三支 run）：**123.82 / 152.63 / 236.54 ms**
⇒ 同 cell 內 spread 仍有 1.91×，而 t/s 同 cell 只有 1.19×。

## 5. 實用換算：要測出 1.7% 效應需要幾臂（雙樣本、95%、未配對）

| 指標 | 每側臂數 |
|---|---:|
| t/s（CV 7.88%） | **≈165 臂** |
| µs/miss（CV 1.95%） | **≈10 臂** |

（配對設計可大幅減少，這裡給未配對的保守上界。
⇒ **M-W 那 1.7% 用 t/s 永遠測不出來，用 µs/miss 才測得出來** —— 這正是當初改用 µs/miss 當主指標的原因。）

## 6. 判準（往後照用）

1. **主指標用 run 級累加量**（µs/miss、`fill_batch_usec`），不要用逐步 GPUTIME 量。
2. **t/s 只當「有沒有跑反」的哨兵**，幅度一律不引用。
3. **`gap` 不作單獨證據**（CV 64–85%）。
4. **`union+gap` 不可當步時**（低估 2.10×）；要步時就用 `ML / t_s`。
5. 所有 run 的 `skipped` 均值都是 **4.9–7.3**（非 0），而程式註明「skipped 必須為 0」
   ⇒ `busy/union` 與 `gap` 都帶未知偏差，只可做同 run 內相對比較。

---

## 附：重現

- 8 支 raw：`Backup/phase_decomp/{K5/raw/nsm.stderr, K5/raw/elemw_census.stderr,
  K3/raw/on.stderr, K3/raw/off.stderr, L3/gc_s2a/*.stderr.log,
  five_20260920_224615, iopath_20260921_001914_2, five_20260920_230927}`
- 10 臂 t/s 與 µs/miss：`Backup/phase_decomp/L3/wd_w*_[cd]*/row.json`
- 相關：`docs/BUSY_UNION_RATIO_2026-09-21.md`（busy/union = 1.510）、`docs/M_W_DELIVERY_VERDICT_2026-09-21.md`
