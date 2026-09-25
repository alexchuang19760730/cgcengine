# 250 / 25 攻關 — 逐條技術白皮書總目錄（44 條）

> 每一條實驗／嘗試一對檔案（`html` ＋ `md`）：**目標 → 判準 → 結果 → 判定** ＋ 依據 ＋ 對應報告。
> 回 [mindmap 總圖](../index.html)　·　[HTML 版總目錄](index.html)

## 生產（可放生產／已交付）（14）

*① ② ③b —— 已進生產、或已是生產決策依據*

| 級 | 條目 | 子目標 | 報告 | MD | 結果 |
|---|---|---|---|---|---|
| 1 | [① 攻關成功（pp≥250 ∧ tg>12.57）](m-total.html) | 不適用 | 0 | [m-total.md](m-total.md) | ⛔ 空 —— 最接近的一次是同 cell pp 260.41 ＋ tg 12.195（差 3%） |
| 2 | [prefill ≥ 250（交付 cell）](m-prefill250.html) | 不適用 | 19 | [m-prefill250.md](m-prefill250.md) | 9 次 launch ≥250，最高 296.24；乾淨視窗 283.01；同期 decode 11.49~12.20 |
| 3b | [池大小掃描（8→4→2 GiB）](io-poolsize.html) | S 序列化消減 | 1 | [io-poolsize.md](io-poolsize.md) | 4 GiB 10.60 vs 8 GiB 10.28＝+3.1%（噪音內）；4 GiB miss 1.91×、capacity miss 3 |
| 3b | [S1 探針臂：slot table 放 GPU（數值身分）](s1-probe.html) | S 序列化消減 | 6 | [s1-probe.md](s1-probe.md) | 576/576 全同；answer_md5 相同 |
| 3b | [3a：未填充 expert 貢獻歸零（MISS_MASK / ZERO_MISS）](miss-3a.html) | S 序列化消減 | 3 | [miss-3a.md](miss-3a.md) | 兩條都過（38 層/421 元素/0 差異；Δ = −0.95 ± 2.01 ms） |
| 3b | [MTP 儀器化（接進 llama-bench）](mtp-instrument.html) | M MTP on 加速 | 6 | [mtp-instrument.md](mtp-instrument.md) | 接通（真因＝test_gen_spec 的 n_past 初始化）；此後 MTP 不再跨儀器比較 |
| 3b | [MTP 口徑與混淆定位（pool／跨啟動／順序）](mtp-caliper.html) | M MTP on 加速 | 8 | [mtp-caliper.md](mtp-caliper.md) | 定案：任何單點 MTP 數字不可引用；pool 4 vs 8 GiB 是一階混淆；同 launch 配對 ×1.06、生產 server 路 |
| 3b | [MTP 到 2× 的邊界（攤薄算術）](mtp-2x.html) | M MTP on 加速 | 2 | [mtp-2x.md](mtp-2x.md) | k 加到多大都不到 2×（server a/m=1.445、bench 0.982）；現況每產出 token 成本 45.6~56.4 vs |
| 3b | [cb 口徑定讞（42 vs 74）](cache-cb.html) | S 序列化消減 | 0 | [cache-cb.md](cache-cb.md) | 定讞 42~51 ms；74.18 撤回 |
| 3b | [K3 邊際單價（dispatch 成本）](k3-price.html) | C kernel／頻寬效率 | 5 | [k3-price.md](k3-price.md) | 45.6（平均）作廢 → 邊際 0.0195%/dispatch/步（17.9 µs）⇒ 所有舊「融合能省 X%」的算術全部作廢 |
| 3b | [shape／knob 世界模型（14–15 個旋鈕的邊界）](shape-knobs.html) | C kernel／頻寬效率 | 15 | [shape-knobs.md](shape-knobs.md) | SHAPE_KNOB_LANDED 證實落地；SHAPE_ROOFLINE 抓到「down 是 IQ3_S 不是 IQ3_XXS」等量測事實 |
| 3b | [swap 結構修復（L0–L4 + P0/P1/P2）](sys-swap.html) | 不適用 | 2 | [sys-swap.md](sys-swap.md) | launch swap 0、decode 11.49、thermal NOMINAL（commit efba7c1d5） |
| 3b | [server 窗口／box 准入（單一來源閘門）](sys-window.html) | 不適用 | 2 | [sys-window.md](sys-window.md) | BOX_ADMISSION_SINGLE_SOURCE 定為單一來源；SERVER_WINDOW_LEDGER 記錄逐次窗口 |
| 3b | [expert cache 血統設計（09-05~09-09 期）](na-design.html) | S 序列化消減 | 8 | [na-design.md](na-design.md) | HYBRID_DESIGN／INVARIANTS／INTEGRATION_DIFF／COMMIT_DIGEST：血統進了生產的 expert |

## 實驗階段（階段性）（10）

*③a —— 機制／量測成立，但產物還不能進生產*

| 級 | 條目 | 子目標 | 報告 | MD | 結果 |
|---|---|---|---|---|---|
| 3a | [CGC_EB_NOFILL 診斷臂（fill 成本）](io-nofill.html) | S 序列化消減 | 0 | [io-nofill.md](io-nofill.md) | fill 3.955 → 0.206 ms/step（−95%）⇒ 機制證；生產口徑同步 fill 僅占 step 4.7% |
| 3a | [單段提交（41 段 → 1 段）](s1-segbatch.html) | S 序列化消減 | 7 | [s1-segbatch.md](s1-segbatch.md) | A 11.30 → B 20.73 t/s；−40.3 ms/token（45.5%），其中序列化 ~36 ms、fill 僅 3.96 m |
| 3a | [異步 gather 流水線（單段＋miss 後台補＋局部重算）](s1-asyncgather.html) | S 序列化消減 | 0 | [s1-asyncgather.md](s1-asyncgather.md) | 方案設計完成、待實施（無實測）；先跑 Stage B 量 fill 時間 vs GPU 窗口 ⇒ go/no-go |
| 3a | [MTP k-sweep（verify batch T 成本曲線）](mtp-ksweep.html) | M MTP on 加速 | 5 | [mtp-ksweep.md](mtp-ksweep.md) | step ≈ 14.61 + 42.03·T（max resid 11.5）；最佳 k=2 但那格不可移植（合成 accept 0.93~0 |
| 3a | [ρ 路線（按層批次化 prefetch）](cache-rho.html) | S 序列化消減 | 5 | [cache-rho.md](cache-rho.md) | 判活：覆蓋 0.849，單獨做 14.36 t/s（+14.2%）；但視窗只有 1.30~1.59 ms，且要付 4.76 ms/步 GPU |
| 3a | [prebind／方案 A（預指派 slot）](cache-prebind.html) | S 序列化消減 | 7 | [cache-prebind.md](cache-prebind.md) | 判活：qu2/qu3 全過、qu1 邊緣 ⇒ 14.33 t/s（+14.0%）；提前一整步發起 ⇒ 視窗 ≈ 一步 |
| 3a | [k=3 的 1.43× 飄移定位 ＋ 配對認證](k3-swing.html) | 不適用 | 3 | [k3-swing.md](k3-swing.md) | 飄移已定位；配對認證 n=5 時 t=2.08 未達 df=4 的 2.776 ⇒ 16.4% 仍未認證；上一輪「bench 配對 sd 更 |
| 3a | [device span 歸因（最大一塊時間）](c-device-span.html) | C kernel／頻寬效率 | 2 | [c-device-span.md](c-device-span.md) | 成立：57% 住在 MoE 區（層內節點 0–39）、42% 住在 attention／GDN 區（40–89）；邊際 verify tok |
| 3a | [頻寬屋頂／dense GEMV 上界](c-bandwidth.html) | C kernel／頻寬效率 | 2 | [c-bandwidth.md](c-bandwidth.md) | dense GEMV 每步 13.39 ms（13.7–16.8%），實測已跑 56–108 GB/s（DRAM 峰值 108.8）；就算全 |
| 3a | [G1：可達上界 / G1-G7 sweep](g1.html) | C kernel／頻寬效率 | 10 | [g1.md](g1.md) | 串行且空轉 19.2% 是「可恢復」的形狀，但 G1 已判定不可達（下界 9.0% > 5%）⇒ 「多少」目前不可引用（44/45 份非零  |

## 已結案（廢棄／不適用）（20）

*④ 廢棄 ＋ 不適用*

| 級 | 條目 | 子目標 | 報告 | MD | 結果 |
|---|---|---|---|---|---|
| 4 | [decode ≥ 25（M-25）](m-decode25.html) | 兩者 | 15 | [m-decode25.md](m-decode25.md) | 交付 12.57＝79.6 ms ⇒ 要砍 49.6%；IO 全關也只 ~+5%（保守 +9%）⇒ 判死 |
| 4 | [IO 請求形狀（合併 pread）](io-shape.html) | S 序列化消減 | 2 | [io-shape.md](io-shape.md) | 一個 expert 三段相隔 88~115 MB ⇒ 3→1 要付 182.6× 位元組；合併邏輯早已存在（preadv）；固定開銷占 90 |
| 4 | [M-W：WORKERS 8→2](io-mw.html) | S 序列化消減 | 1 | [io-mw.md](io-mw.md) | us/job −11.1%、us_per_miss −14.7%，但 t/s −0.6% ⇒ 未分離（by-product：IO 只占 st |
| 4 | [CGC_LAYER_AHEAD_PREFETCH（提前一層）](io-layer-ahead.html) | S 序列化消減 | 0 | [io-layer-ahead.md](io-layer-ahead.md) | −7.1%（未分離、方向偏負） |
| 4 | [M-PF：背景 neighbour prefetch](io-mpf.html) | S 序列化消減 | 1 | [io-mpf.md](io-mpf.md) | 撤回（讀入位元組 1.98×，實測最好只到 35%） |
| 4 | [S1 早期診斷系列（09-16/17）](s1-refuted.html) | S 序列化消減 | 2 | [s1-refuted.md](s1-refuted.md) | 多輪被自己否證：slot owner 推論被推翻、時序推論被推翻、09-16 那七輪「第一個分歧」全部不可引用（同時踩三個盲點） |
| 4 | [3b：fill 觸發點搬出 hook ＋ batch 化](miss-3b.html) | S 序列化消減 | 0 | [miss-3b.md](miss-3b.md) | 判死：合併邏輯早已存在、幾何 182.6×、生產口徑上界 ~9% ⇒ 收益 ~2.9% |
| 4 | [3c：per-expert 重算 kernel](miss-3c.html) | S 序列化消減 | 0 | [miss-3c.md](miss-3c.md) | 不做（依賴鏈斷；上限同受 ~5% 約束；舊估 +13~18% 來自已作廢的非生產 cell） |
| 4 | [RSL-MTP（draft top-8 ⊆ 已付費並集）＋ 練 draft head](mtp-rsl.html) | M MTP on 加速 | 4 | [mtp-rsl.md](mtp-rsl.md) | 被自己的數據否決：0.70–0.97×，無一格 ≥1.0；m=0.474 下即使 a→1 也不可能 2× |
| 4 | [verify residency thrash ／ 池配額 ／ prefetch 開關](mtp-verify-opt.html) | M MTP on 加速 | 3 | [mtp-verify-opt.md](mtp-verify-opt.md) | 降到次要：彈性只有 0.10–0.17（k=1/3 甚至 −0.38）；k_eff 單變數解釋 84%；修 residency 上界只剩 ~ |
| 4 | [accept rule／dynamic-k／n_max 調整](mtp-accept.html) | M MTP on 加速 | 2 | [mtp-accept.md](mtp-accept.md) | 不做：dynamic-k oracle 只 1.035×；n_max 3→5 作廢；greedy 下 accept 不可由 accept r |
| 4 | [K0–K5 系列（融合／小 op 群）](k-series.html) | C kernel／頻寬效率 | 3 | [k-series.md](k-series.md) | K1 不做（G2 禁區）、K4 主機側否證（gpu_union 占步時 92%）、K5 不做、L2 歸零、M-K5 不做；加 thread  |
| 4 | [G4：元素級融合 kernel](g4.html) | C kernel／頻寬效率 | 11 | [g4.md](g4.md) | 判 0 ⇒ 不寫融合 kernel（回歸斜率 0.0132 µs/numel ⇒ 融合的 gain 是 0） |
| 4 | [IOCACHE 約束承認 ＋ 段邊界 S2](io-constraint.html) | S 序列化消減 | 5 | [io-constraint.md](io-constraint.md) | IOCACHE：#6 沒做之前 #3 的 async 版本不可達（且不可並列相加）；S2：建議不做完（段邊界／drain） |
| 4 | [M3／M4／M5 離開條件（09-17 期）](m3.html) | 不適用 | 1 | [m3.md](m3.md) | M3 未達（9.82 vs 門檻 15）；M4 靠一個本身壞掉的量測被否決、修好後才關閉 —— 而那個量測現已作廢 |
| 4 | [OMLX verify kernel ／ 其他 verify 路線](misc-omlx.html) | C kernel／頻寬效率 | 1 | [misc-omlx.md](misc-omlx.md) | 判死（見 OMLX_VERIFY_KERNEL_VERDICT） |
| 4 | [softpool／L4 v1v2／doublebuffer spike（09-05~09-06 期）](na-softpool.html) | 不適用 | 11 | [na-softpool.md](na-softpool.md) | 推定被 hybrid 路線取代（37/41 份零外部引用 ⇒ 已孤立） |
| 4 | [舊口徑數據報告（Gemma4／MTP_BENCHMARK_WIN8GB／TPOT 路線圖）](na-olddata.html) | 不適用 | 11 | [na-olddata.md](na-olddata.md) | 作廢：09-17 起 decode 一律 llama-bench、warm-skip 口徑（`MEMORY_PERF.md` 裁定），且 M |
| na | [入口／索引／決策頁（非實驗）](na-entry.html) | 不適用 | 5 | [na-entry.md](na-entry.md) | 不進四級：它們是入口與索引 |
| na | [跨線／其他產品（Wan2.2、HarmonyOS、Windows client、Colibri、Unified IR…）](na-crossline.html) | 不適用 | 0 | [na-crossline.md](na-crossline.md) | 本線無實測權或非本線主題 ⇒ 不塞進四級 |

---

由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成。
