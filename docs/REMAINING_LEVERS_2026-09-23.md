# 本机 decode 还剩哪些杠杆（2026-09-23 23:0x）

> 起因：使用者問「如果要達到本機最佳的 decode 速度，以目前最新的 commit 還有啥可以做的」。
> 本篇是**純結帳**（0 重建、0 benchmark），把 `MILESTONE_MAP_2026-09-21.md` 的判決與
> `DEVICE_BUSY_ATTRIBUTED_2026-09-23.md` 的歸因合起來，分出「已判死」與「還能碰」兩欄。
> 口徑一律是**交付 cell**（`prod_profile.py`, reps=3, NOMINAL, `--warm-skip 64`）：**12.57 t/s / step 247.98 ms**。

---

## 0. 結論先行

| | 值 |
|---|---|
| 今天交付 | **12.57 t/s** |
| **最值得做的第一件事** | **提高池覆蓋率**（§2）—— 唯一不依賴任何已否證假設的真實槓桿 |
| 它可拿到 | **13.4 ~ 14.3 t/s**（+6.5% ~ +14%） |
| **結構上限**（cb 全藏） | **15.8 t/s**（`DEVICE_BUSY` §4 已給；本篇 §2 獨立算出 15.14，差 4%） |
| 25 t/s | **不可達**（見 `DECODE_25TPS_VERDICT_2026-09-23.md`） |

---

## 1. 已判死 — 別再花時間（每條都附判據）

| 路線 | 判據 |
|---|---|
| async fill / A‑B 段拆分 / 背景預取（**ρ、prebind、M‑PF**） | `L3_WINDOW_ZERO`：MoE 永遠在 segment 第一個 buffer（**5811/5811**）⇒ 可遮窗口 = 0。ρ 實跑：hit +26pp 但 **t/s −21%**（`fill_wait_us` 44 ms → 15.4 s） |
| **M‑K5 / L2**（壓 GPU busy） | 判 **(C) 成立** ⇒ 30–38% 上界作廢 ⇒ **L2 = 0**（§EN‑406/414） |
| **M‑L1** 融合（9‑ADD / cluster‑1） | 單項 0.76% / 0.57%，全低於 3% 門檻 ⇒ 結案不做 |
| **M‑PF** neighbour prefetch R=1 | 淨虧：時間 ×1.87–3.05，步級 IO 309–429 ms ≫ 陰影 159.77 ms |
| **25 t/s** | `GPU busy` 142.6~181.6 ms > 預算 124.7 ms ⇒ P<3%（`DECODE_25TPS_VERDICT`） |

**這五條的共同教訓**：它們全都在試圖**隱藏**同一段 IO。依賴鏈決定了它藏不掉。

---

## 2. ★ 還能做的第一名：提高池覆蓋率 ⇒ 讓 miss 少發生

### 為什麼它與上面五條不同

上面五條的方向是「**把 IO 藏進 GPU 的空檔**」——需要空檔存在，而 `wait` 視窗內 GPU 是滿的（device/wait ≈ 1.0）。
本條的方向是「**讓 IO 根本不發生**」：提高 `hit%` ⇒ miss 次數下降 ⇒ `cb` 線性下降。**不依賴任何未被驗證的重疊假設。**

### 現狀實測（`base_r1` 起動 log，不是外推）

```
llama_expert_cache: L4 metal pool: 143 slots/layer
llama_expert_cache: slab fills: pool=6153.7 MiB disk=4940.3 MiB (non-resident share 44.5%)
llama_expert_cache: hits=13462/21908 (61.4%)
```

⇒ resident **6153.7 MiB**，**143 slots/layer** ⇒ **覆蓋率 55.9%**，實測 **hit 61.4%**（= 覆蓋率 **+5.5pp**）。
這與 §EN‑471 的結論一致：**hit% 是被「記憶體能放幾個專家」決定的，不是被預測器決定的。**

### 收益表（per-expert slot = 1.0703 MiB，三源閉合 0.3%）

| pool 增量 | slots/layer | 覆蓋率 | miss 降幅 | cb → | t/s |
|---:|---:|---:|---:|---:|---:|
| 今天 | 143 | 55.9% | — | 42.0 ms | **12.57** |
| +1.5 GiB | 180 | 70.2% | −37% | 26.8 ms | **13.39** |
| +3.0 GiB | 215 | 84.1% | −73% | 11.5 ms | **14.33** |
| +4.6 GiB | 260 | 100% | −100% | 0 | **15.14** |

> **交叉驗證**：全常駐那一格本篇算 **15.14**，`DEVICE_BUSY` §4 由完全不同路徑（`step 197.6`）給 **15.8**
> ⇒ 兩源差 **4%** ⇒ 這個表可信。
> ⚠ 保守假設 `hit ≈ 覆蓋率 + 5.5pp`；LRU 的 temporal locality 在高覆蓋區通常讓 hit 更高 ⇒ **這是下界**。

### ★ 前提：多出來的內存從哪來（**這是第一個要查的數**）

機器是 Apple M4、**16 GiB 統一內存**（不是獨立顯卡）。今天的分配大致是：

```
專家 pool resident  6153.7 MiB
dense 部分           ~2.02 GiB
macOS + GPU working set + KV cache(ctx 4096) + 激活  →  剩餘
```

⇒ 唯一可能讓位的大戶是 **KV cache**。
⚠ **尚未驗證**：實跑 stderr 沒有印 ctx/KV 的分配行（`grep "KV cache|n_ctx|llama_new_context"` 零命中）
⇒ **第 0 步就是把它印出來**（或直接比 `ctx 4096 → 2048` 的 rss 差）。

⚠ **不可拿內存的地方**：swap。實測 swap 5301 MiB 時 hit% 68.7→57.2、base **12.57 → 4.62**。

### 同效果的便宜變體：不改記憶體總量，改 **per-expert bytes↓**

resident 固定時，每專家佔的位元組越小 ⇒ 能裝的 slot 越多 ⇒ 覆蓋率上升。**這條不用跟 KV 搶記憶體。**
（具體量化組合未定，但方向與上表同一個公式。）

---

## 3. 還能做的第二名：替 attention/GDN 區做一次 kernel 地板

`DEVICE_BUSY_ATTRIBUTED` §2／§4 的歸屬（細切，T=4，n=48）：

| 區域 | ms/step | 佔比 | 有 kernel 地板可對照嗎 |
|---|---:|---:|---|
| MoE 區（0–39） | 95.4 | 57% | ✅ expert bank 家族 T=4 = 31.8 ms/步 |
| **attention/GDN 區（40–89）** | **69.3** | **42%** | ❌ **無 —— 從沒做過** |

而且在**邊際 verify token** 上，`+26.3 ms` 裝置時間裡 **attn/GDN 佔 +13.41 ms**（MoE 只 +14.22）。
文中自己也寫：「KV 讀取的地板約 0.4 ms/token，**離 13.4 很遠**」—— 雖然那句是用來說明「未知」，
但也正是**這塊 42% 完全沒被定過價**的意思。

⇒ **建議動作**（不需重建、不需 server）：把 `mmid_shapes.py` 那一套 shape probe 套到 attention/GDN 家族，
先拿到地板，才知道有沒有東西可拿。**在拿到之前，任何「這區能省 X%」的說法都不可引用。**

---

## 4. 還能做的第三名：M‑S2（里程碑表裡唯一沒被判死的一格）

`MILESTONE_MAP_2026-09-21.md` §3 把它與那些被否證的 async fill **明確分開**：

> S2 是另一件事：把 slot_table 的發布搬到裝置，讓 `CGC_SUBMIT_AHEAD=1` 從「故意錯的探針」變成合法。
> 它拿掉的是 **wait ＋ host 讀 ids ＋ 記帳** 那段，**不是** fill 的 IO。

⇒ 因為它不碰 fill，**不撞 `L3_WINDOW_ZERO` 的依賴鏈**。
- **22.9 那端不可達**（假設 cb 能被 submit‑ahead 吸收，已實測不能）⇒ **19.0 那端才是真上界**。
- 文檔自評：「上界 ≤ ~5%，且不可引用」，且 `gap ⊆ cb + submit` 的交叉檢查**以 10.09 ms 失敗** ⇒ 現階段不給數字。
- 狀態：**未動** ⇒ 是表上唯一還活著的走向。

---

## 5. 最便宜的一格：M‑W 在交付 cell 重驗

| | |
|---|---|
| 成本 | **~20 min，0 重建** |
| 已知 |現測 **+1.7%**（低於 3% 門檻）|
| 弱點 | 那是另一個 cell（`t/s` 7.6–12.1），**與交付的 12.57 不同口徑** ⇒ 換 cell 後可能不同（也可能歸零）|

`MILESTONE_MAP` §4 把它列為「便宜到貴」的第 1 位，理由就是**零風險**。

---

## 6. MoE 區本身的空間（有地板，但要小心讀）

MoE 區（0–39）**95.4 ms/step** vs expert bank 家族 T=4 地板 **31.8 ms/步** ⇒ 表面差 **3.0×**。
⚠ 但該節點區含 `shared_expert_gate`、`norm`、`conv`，比那個家族多 ⇒
**3.0× 是「差距的上界」，不是量到的差距。** 要收窄只能靠 kind 鍵化的歸屬（emitter 改動）。

---

## 7. 建議順序（便宜到貴）

| # | 動作 | 成本 | 產出 |
|---|---|---|---|
| 0 | **印出 KV cache 實際占用**（或 ctx 4096 vs 2048 的 rss 差） | ~10 min | 決定 §2 能走多遠 |
| 1 | **M‑W 在交付 cell 重驗**（AB/BA 配對，主指標 `µs/miss`） | ~20 min，0 重建 | 那一格到底值幾個點 |
| 2 | **attention/GDN shape probe**（类比 `mmid_shapes.py`） | 不需重建 | 替 42% 的裝置時間定價 |
| 3 | 依 0 的結果，決定池 +1.5 或 +3.0 GiB，並用交付 cell A/B | 要重建 | 目標 13.4~14.3 |
| 4 | M‑S2 slot_table 裝置化 | 工程 | 未評估值，先看 1/2/3 |

**不要同時做** —— 這個 repo 的每一個比較都建立在「同一個 build、同一個 cell、 swap=0」這個前提上。

---

*無新量測。引用產物：`MILESTONE_MAP_2026-09-21.md`、`DEVICE_BUSY_ATTRIBUTED_2026-09-23.md`、
`CB_DELIVERY_SETTLED_2026-09-23.md`、`EXPERT_IO_SHAPE_FIRST_PRINCIPLES_2026-09-23.md`、
`DECODE_25TPS_VERDICT_2026-09-23.md`。*
