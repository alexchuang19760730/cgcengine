# 窗口哨兵：一個「thermal NOMINAL ＋ 記憶體夠」的窗口，也可能是慢 5 倍的

日期：2026-09-20（20:0x）｜線：線A (ace)｜產物：`scripts/check/window_sentinel.py`、
`scripts/check/window_sentinel_ref.json`、`scripts/check/prod_profile.py` 的 `window_crosscheck()`

---

## 1. 為什麼需要它：兩個閘門都過，機器卻慢 5 倍

2026-09-20 18:1x，同一個 decode cell 的三臂回 **2.17 / 1.36 / 1.35 t/s**，而同一個 cell 在
16:10 是 **10.64**、17:5x 是 9.49/10.83。同一時間：

| 既有閘門 | 讀數 | 判定 |
|---|---|---|
| thermal key（`thermalpressurelevel`） | **NOMINAL，全程** | 過 |
| usable memory | **54%** | 過 |
| 有沒有別的 session | **沒有** | 過 |

三個條件全過，盒子卻慢 ~5 倍。而這種窗口**在被人發現之前已經污染了兩輪量測**（其中一輪是我自己的）。
第二輪的簽名更明顯：`on 11.44 / off 10.16 / off 3.24 / on 2.24` —— 後兩臂在 NOMINAL 下崩掉。

⇒ 結論不是「閘門設錯了」，而是**「盒子健康嗎」是一個物理問題，不是帳面問題**。
帳面（thermal key、可用記憶體）只能證明盒子**自稱**沒事；要證明它真的沒事，只能**量吞吐**。

---

## 2. 它量什麼，以及為什麼選 prefill

哨兵形狀＝**凍結生產 profile 的 prefill cell**：`-p 2048 -n 0 -b 5632 -ub 5632 --load-mode none`。

選 prefill 的三個理由，每一個都對著 decode 的弱點：

1. **prefill 由持續 matmul 吞吐主導** ⇒ 緊跟時脈/功耗狀態，降級會立刻反映；
2. **一次前向就結束** ⇒ 便宜（實測 **59 s**，含載入）；
3. **單臂散佈小** ⇒ decode cell 的單臂散佈是 **±27%**（§EN-335），用它當哨兵，
   「盒子變慢了」和「treatment 有效」會混在一起分不開。prefill 不會。

判準是**頻帶**，不是門檻：讀數必須 ≥ 記錄中位數 × 0.85。

| | 值 |
|---|---|
| 記錄中位數 | **281.51 t/s** |
| 健康樣本跨度 | 275.65 – 300.43（1.09×） |
| 下限（85%） | **239.28** |
| 18:1x 那個降級窗口 | **~0.2×** 中位數 |

⇒ **這條線不是刀口**：最差的健康樣本離下限還有 13%，而降級窗口落在 0.2×。中間是寬的。

---

## 3. 參考頻帶的來源，與它不誠實的地方

`window_sentinel_ref.json` 的五個樣本**全部來自既有記錄**，不是在一台「當場認證健康」的機器上重錄的：

- 276.25 / 300.43 —— `prod_profile.py` docstring 引用的 prefill-house 讀數
- 292.34 —— 2026-09-18 可引用的 prefill
- 281.51 —— `Backup/prod_profile/prod_profile_20260920_1528.json`（PASS，NOMINAL/NOMINAL）
- 275.65 —— 同一支工具 15:50（只因跑完離開 NOMINAL 而被判不合格）

⚠️ **這是本工具最大的弱點，寫在這裡而不是藏起來**：如果那幾次記錄本身就有降級，
頻帶會把降級認証成健康。修正方式是一台你**獨立認證過**的機器上重錄：

```sh
python3 scripts/check/window_sentinel.py --record --this-box-is-healthy
```

`--record` **一定要**配 `--this-box-is-healthy`，否則工具拒絕寫入 —— 在降級的盒子上錄參考，
會讓哨兵從此把降級認証成健康。

---

## 4. 第一次實跑（20:11:50–20:12:50，59 s）

```
thermal NOMINAL | usable 38.7% | free 0.59 GiB
measured 274.14 t/s (rc=0)
reference median 281.51  floor 239.28 (85%)  ->  HEALTHY
```

⚠️ **這一次讀數的乾淨度要打折**：事後發現 20:11–20:16 期間另一條線（`線 I` 的
`mtp_accept_ab` L3 dose arm1）的 `llama-server` 正在這台機器上跑。也就是說
**哨兵在「有人共用」的盒子上仍讀出 274（0.97× 中位數）**。

這有兩個含意，方向相反，都別省略：

- **好消息**：哨兵對「別人也載了一份模型」這種共用**不敏感** ⇒ 在共用盒子上它不會一直誤報；
- **壞消息**：它因此**不能證明盒子是獨佔的**。獨佔性還是要靠 `pgrep` 那條閘門，
  哨兵只回答「物理上快不快」，不回答「有沒有人跟你搶」。

---

## 4.1 同一晚的自證：214.30（壞）vs 281.95（好）

哨兵做完的當晚，盒子自己當了對照組 —— 同一支工具、同一形狀、相隔 15 分鐘：

| 時刻 | 哨兵讀數 | 同一刻既有閘門 |
|---|---|---|
| 20:19:35 | **214.30 t/s**（0.76× 中位數）⇒ DEGRADED | `PASS -- NOMINAL, no other session, usable 49.6%` |
| 20:34 | **281.95 t/s**（1.00× 中位數）⇒ HEALTHY | NOMINAL / NOMINAL |

差 **1.32×**。這確認了兩件事：頻帶的位置是對的（好窗口落在帶內、壞窗口落在帶外），
以及**壞窗口確實會在閘門全綠的時候出現**。

而且要再強調一次 §2 那點：**214.30 只慢 24%，而 decode 單臂噪音帶是 ±27%**。
換句話說，如果沒有哨兵，這一輪會產出 4 個「只是偏低一點」的 decode 數字，
被當成噪音吸收掉 —— 甚至被讀成 treatment 效應。哨兵當場把它擋下來，省下 3 臂。

## 5. 接到了哪裡

### (a) A/B 工具：每一臂發射前自證（`Backup/metal_fusion_dispatch_cost_ab.py`）

每臂之前跑一次哨兵；**DEGRADED 就停跑**，而不是把剩下的臂繼續燒在壞窗口上
（18:1x 那兩輪就是這麼被浪費掉的）。每臂的 `sentinel_ts` / `sentinel` 會寫進 JSON，
所以事後看得出「這一臂是在什麼樣的盒子上量的」。

```sh
python3 Backup/metal_fusion_dispatch_cost_ab.py --order on,off,off,on   # 預設開哨兵
python3 Backup/metal_fusion_dispatch_cost_ab.py --no-sentinel           # 除非你另有理由相信窗口
```

### (b) `prod_profile.py`：零 GPU 成本的交叉檢查

`prod_profile.py` 本來就同時跑 prefill 與 decode 兩軸，而**它的 prefill 軸就是哨兵形狀**
（pp2048，profile 自己的 batch）⇒ 不必多發一次，直接拿它自己量到的 prefill 讀數對頻帶比：

```python
window_crosscheck(recs) -> {"verdict": "HEALTHY"|"DEGRADED"|"UNKNOWN", "ratio_to_median": …}
```

DEGRADED 時印一段警告，並在 JSON 頂層寫 `window`。**單軸 verdict 刻意不改**：
降級窗口通常自己就過不了 250/12，而這個檢查真正要抓的是**別人拿這張表去做相對比較**。

自測（合成輸入）：281.5 → HEALTHY（0.999×）、60.0 → DEGRADED（0.213×，正是 18:1x 的簽名）、
沒有 prefill 軸 → UNKNOWN。

---

## 6. 它**不**做什麼

- **不修降級**，也不解釋它。最可能是 SoC 功耗/熱管理（當天連續六小時、swap 一度 6.6 GB），
  而它不在 OS 的 thermal key 裡 —— 哨兵只負責**在它發生的時候抓到**。
- **不證明獨佔**（見 §4）。
- **不取代配對設計**。單臂 ±27% 的噪音還在；哨兵只是把「盒子慢 5 倍」這種**大幅**污染擋掉，
  小幅慢漂仍然要靠 `paired_ab.py` 的 AB/BA 交替。

## 7. 下一步

1. 在一台**獨佔且哨兵 HEALTHY** 的盒子上用 `--record --this-box-is-healthy` 重錄頻帶，
   把 §3 那個弱點關掉。
2. 把 18:1x 被污染的 fusion 歸因（`on/off` 配對）在認證窗口裡重跑 —— 判準在跑之前就寫死了：
   OFF 比 ON 慢不到 5% ⇒ 一個 dispatch 不值錢 ⇒ 放棄集群 1。

## 8. 附：盒子進 DEGRADED 時該怎麼辦（已知無效 vs 已知有效）

- **無效**：等 thermal key 回到 NOMINAL。降級窗口的定義就是「key 說 NOMINAL 但很慢」，
  `wait_nominal()` 一查就通過、立刻返回，什麼也不會發生。
- **有效但目前說不出機制**：**給它時間**。20:19 那次降到 0.76× 之後，盒子在 20:22 直接
  進 HEAVY，再過約 10 分鐘自己回到 NOMINAL，20:34 讀到 1.00×。
  A/B 工具因此改寫成「睡 `--sentinel-wait`（預設 240 s）再重測」，最多 `--sentinel-retries` 次。
- **未證實**：swap 一度 4.5–5.1 GiB，是否為原因之一沒有測過。
