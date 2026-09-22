# 半步 (B)：把測量工具的 footprint 降下來 —— 結果出來了，**不是** footprint 的鍋

日期：2026-09-23 01:30–02:30
對象：`docs/JOINT_SHAPE_KERNEL_AXIS_2026-09-22.md` §8.5 留下來的 (A)/(B) 待決問題
結論（一句話）：**footprint 這根槓桿是真的、也量得出來，但它比真正的原因小 20 倍以上 ⇒ 共用常數 `NEED_MB=8000` 不動，要改的是「儀器要在乾淨窗口才准跑」。**

---

## 1. 原本待決的問題是什麼

joint（shape × kernel）代理 pilot 連跑三次，**儀器三次都拒答**（null cell 遠大於 ≤5% 的門檻，
最後一次還出現負的 %峰值）。當時排除到兩個可能：

- (A) 維持共用門檻 `NEED_MB=8000` ⇒ 端到端 sweep 今晚、明晚都發不出去
- (B) 是「測量工具太吃記憶體」，把 footprint 降一點看看

使用者選 (B)，理由是「這是低成本的實驗，值得做」，而且明確要求：**動共用常數之前先問過，不要偷偷放寬。**
本文是把 (B) 做完的結果。**做完的答案是 (B) 不成立**，而且附帶抓到了真兇。

---

## 2. 先要有活能量：新工具 `scripts/check/shape_probe/footprint_ab.py`

原本沒有任何東西能回答「一次 probe invocation 到底吃掉多少」。寫壞的前提是先要知道三件事，
而它們彼此分離、砍錯一件就白砍：

| 成分 | 用什麼調 | 是不是我們能砍的 |
|---|---|---|
| **分配量** pool / bank / host fill buffer | `--pool-mib` | 是（但 bank 不行） |
| **持續時間** 熱頁被摸多久 | `--target-ms` | 是 |
| **幾何** bank 本身多大（model 的 256 experts） | `--experts` | **不是**——砍它就是換一個模型在量 |

所以儀器一次問兩個可分離的問題：
arm **A** = 今天的預設（pool 320 / target 1500）、arm **B** = 瘦池、arm **C** = 短算。
A−B 隔離「分配量」，A−C 隔離「持續時間」。

量什麼（每個 invocation 一次）：

- `peak_rss_mb`：行程自己的峰值 RSS（macOS `/usr/bin/time -l`）
- `dip_mb`：跑的當下，共用視窗自己的 `vm_free_mb` 從出發點掉多深
- `wall_s`、swap 變化

selftest **26/26**（含 `.footprint_ab.py` 自己的拒答邏輯）。

---

## 3. 實驗一：三臂，單次區間（reps=3，兩個形狀）

| arm | pool | target | 峰值 RSS | 盒子可用下陷 | wall |
|---|---|---|---|---|---|
| A（今天） | 320 MiB | 1500 ms | 232 MB | **487 MB** | 1.81 s |
| B（瘦池） | 146 / 174 MiB | 1500 ms | 232 MB | **358 MB（−26.4%）** | 1.79 s |
| C（短算） | 320 MiB | 100 ms | 232 MB | 464 MB（−4.8%） | 0.39 s |

兩個立刻有用的結果：

1. **RSS 完全沒動**（三臂都 232 MB）。RSS − bank ≈ 120–150 MB，正好是一個行程的固定底。
   ⇒ **pool 裡超過 bank 的那些頁從來沒被觸及**，砍 pool 不會減少行程自己佔的記憶體。
   這件事用推的會推錯方向，只能量。
2. **但盒子的可用記憶體下陷真的降了 26.4%**，而且超出自身離散 ⇒ 可分辨。
   ⇒ 「降低 footprint」這個方向**有一部分是真的**，不是空想。
3. 砍 `target-ms` 換到的下陷只有 4.8%，落在自身離散內 ⇒ 儀器誠實回 `NO READING`（而不是把它兜成一個小贏），
   但它把每次呼叫從 1.81 s 砍到 0.39 s（4.6×）。這是後面有用的另一件事。

---

## 4. 實驗二：帶著瘦池真的去跑 joint sweep

只動一個變數，同一 build（`/tmp/cgc-nsgid-build`，dylib `650dd9d1d9fb89a4`），只加 `--pool-mib 0`：

| 次數 | 起始可用 | null cell（gate / down） | 判準 | 結果 |
|---|---|---|---|---|
| 9-22 20:00 | 5.18 GiB | 44% / — | ≤5% | `INVALID` |
| 9-23 01:05 | 1.47 GiB | 57.7% / 21.0% | ≤5% | `NO READING` |
| 9-23 01:21 | 0.83 GiB | 241.5% / 13.6% | ≤5% | `NO READING`（還出現負 %峰值） |
| **9-23 02:0x（瘦池）** | 1.25 GiB | **26.2% / 15.7%** | ≤5% | `NO READING` |

有進步（241.5% → 26.2%），**但仍然讀不到**——離 ≤5% 還差 3–5 倍。

---

## 5. 實驗三：真正的線索在一次躲貓貓的加減法

joint sweep 的標頭寫 `usable 1.25 GiB`，可是單次 invocation 只會掉 ~0.4 GiB。
**中間少了 6 GB** —— 少了的那一塊叫「累積」。所以加了 `--sequence` 模式：
每臂連打 12 次、**三臂交錯**（讓共有的飄移公平分攤），追蹤每次的**出發點**。

```
arm A  pool 320 MiB  target 1500 ms  baseline 9368 -> 8235 MB  DRIFT -1133 MB
arm B  pool 146 MiB  target 1500 ms  baseline 9470 -> 8216 MB  DRIFT -1254 MB
arm C  pool 320 MiB  target  100 ms  baseline 9392 -> 8179 MB  DRIFT -1213 MB
```

三種 footprint 掉的量**一模一樣**。也就是：這 ~1.2 GB 不是我們能調的那兩個旋鈕造成的。

再加 `--settle-ms 2000`（每次之間留 2 秒還原）也不改善。

---

## 6. 真兇：這台盒子的可用記憶體自己會甩 8.5 GB，而且甩的時候常常有別人

最後一次 joint sweep 加了 begin/end 視窗比對（本節因此才有辦法寫）：

```
window  begin: admits=False reclaimable=1278 MB foreign=[(80017, 'llama-server')]
window  end  : admits=False reclaimable=7727 MB foreign=[]
```

**跑 sweep 的整整兩分鐘裡，另一條線的 llama-server 正在載模型。**
而另一次序列測試裡我紀錄到：同一台機器在 18 秒之內從 1463 MB 甩回 9869 MB。

把它拼起來就是完整的答案：

| 項目 | 大小 | 誰造成的 |
|---|---|---|
| 我們能調的整套 footprint（pool + 持續時間） | **0.13 GB**（±0.05） | 我們，量得出 |
| 每一次 joint sweep 期間流失的可用記憶體 | **~1.2 GB** | 跟我們怎麼設 config 無關 |
| 盒子的可用記憶體自己的擺盪幅度 | **~8.5 GB**（1463 ↔ 9869 MB），數十秒內 | 另一條線的 server 載入／退出 |
| `NEED_MB` 共用門檻 | 8.0 GB | 共用，不動 |

**結論很直白：footprint 這一槓比真正的原因小 20 倍以上。**
五次 joint sweep，五次都沒有一天是在乾淨盒子上的。那五個 null cell 數字（44 / 57.7 / 241.5 / 26.2 / 36.5%）測的是**別人的記憶體環境**，不是 kernel tile 的效果。

---

## 7. 所以 (A)/(B) 的答案是：都不是

- **不要動 `NEED_MB=8000`**。而且現在不動它有更好的理由：不是「預設如此」，是「量過了，降 footprint 換不到可讀性」。
- **也不是「儀器太重」**。（B）做完了，它對，但不是主因。
- **正解是第三條**：這台盒子的joint pilot — 要在共用視窗真的 admit 的時候**才准跑**，而且跑完要再問一次。

`shape_knob_search.py` 上一輪已經接了這個閘門；**`mmid_shapes.py` 沒有，它只會印不會擋**——
這正是它在五分鐘前還會把一次兩分鐘的量測浪費在別人的載入過程裡的原因。已經補上（見下節）。

---

## 8. 這一輪動了哪些code（全部零 GPU，`mmid_shapes.py` selftest **34 → 41/41**）

`scripts/check/shape_probe/mmid_shapes.py`

- `POOL_MIB_DEFAULT / BLOCK_BYTES / lean_pool_mib() / pool_for()`：`marginal()` 裡原本硬寫的 `320`
  一律變成可以調、而且會被 JSON 記錄。`--pool-mib 0` = 瘦池（iq2_s 146 / iq3_s 174 MiB）。
- `window_now()`：模組層的 `WINDOW` 是 import 那一刻的 snapshot，回答不了「跑的時候盒子是誰的」。
- `lost_the_box(w0, w1)`：**有方向的**——只有「起點乾淨、終點被佔」算遺失；反方向不算。
  probe 讀不到時回 `None` 不回 `False`（這條是 selftest 抓出來的，不是 review）。
- **`--nsg-sweep` 現在會擋**：begin 不 admit 就 `return 2` 並印原因；`--allow-busy` 才放行，且放行也要留下 `overridden: true`。

`scripts/check/shape_probe/footprint_ab.py`（新增）

- 三臂 A/B/C、`--sequence N`（交錯）、`--settle-ms`、`--pool-mib/--target-ms` override。
- 共用視窗不會自己(player?)—before 不 admit 就 `return 2`；`--allow-busy` 是會留下紀錄的逃生門。
- selftest **26/26**。含：讀數不足以下結論時（缺 null cell）就回 `unjudged`，不把小差異兜成小贏。

---

## 9. 順便更正我上一輪自己寫錯的兩句話（必須寫下來）

1. 「我把 swapfile 從 5.1 GiB 推到 6.1 GiB，**不會自動縮回去**」—— **錯，它會。**
   實測 total 回到 4096 MiB（used 3348），期間我自己也沒做什麼特別的事。
   我上一輪把「當時觀測到它還沒縮」寫成「它不會縮」，這是把一個暫態當成本質。
2. 「這台機器今晚沒有 8 GB 可回收的空檔」—— **錯，它有，只是不穩定。**
   02:1x 實測六次連採都在 9364–9428 MB，`admits=True`。
   我上一輪把「我採到它的時間點剛好都很低」寫成「它沒有」。這是把一次樣本當成它的分佈。

兩句都是同一個毛病：**用「我看到的」代替「它的性質」**。

---

## 10. 下一步（jin to

一次 clean 窗口有 joint 軸可以重跑時（要求：`server_window.decision()` admit 且 begin/end 都 admit）：

```
scripts/check/shape_probe/mmid_shapes.py --nsg-sweep unset,1,4,8,16,32 --reps 3 \
    --pool-mib 0 --libdir /tmp/cgc-nsgid-build/bin \
    --json Backup/shape_probe/mmid_nsg_lean_<date>.json
```

在那之前，**不要**用 `--allow-busy` 去賭：五次的實驗告訴我們那 2 分鐘一定白花。
而本案留下的證據是：我們能調的那部分，只有這台盒子自己擺盪幅度的 1/20 —— 這正是那些讀數不能引用的理由。
