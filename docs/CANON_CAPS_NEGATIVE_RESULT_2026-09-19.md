# ② 槽位幾何的引擎化：**canonical 歸約順序修不了它**（2026-09-19）

> 問題：§EN-241 量到 `LAYER_CAPS` 會改變模型輸出（A vs E2：M1 1/884、M2 25/884），機制被記成
> 假說「caps 改變每層 expert tensor 的 `ne[2]` ⇒ 改 gather 的歸約樹 ⇒ 每層換一組等價但不同的浮點
> 順序」。本輪要把「等價」拿回來，手段是**讓歸約順序與槽位無關**（按 expert id 排序、fp32 累加），
> 然後用 884 步長 probe 驗證 `LAYER_CAPS` 不再改變輸出，最後把 cb share 16.4% → 13.2% 那一份拿回。
>
> 結論：**那個手段不成立**，而假說本身也被自己的數據否證。下面每一格都是同一顆 binary
> （`libllama aab787412e572550`，D5 閘門同一顆）、同一支 884 步長 probe（prod25、400 tok）量到的。

---

## 1. 手段本身早就存在，而且確實生效——只是對這件事無效

`CGC_CANON_ORDER`（M1 工作項 4，2026-09-17 落地）已經接線：`llama-cgc-canon.h` 的
`cgc_canon_build_perm/apply`、`llama-graph.cpp` 的置換節點、`llama-context.cpp` 同步寫入 remap leaf、
`run_server.sh` 的 env allowlist。離線測試 A 重跑通過（`raw=1185/2000` 對順序敏感、
`canonical=0/2000`）⇒ **不是「機制在、開關沒接」**，這條排除掉了。

| 比對 | n_common | cross_tab | 判定 |
|---|---:|---|---|
| `canonA` vs `canonA2`（**同配置重跑，null control**） | 1003 | `eq_eq=1003` | 儀器在 canon 下完全確定性 |
| **`canonA` vs `canonE2`（canon=1 × LAYER_CAPS）** | 968 | `eq_eq=1, ne_dec_eq=12, ne_ne=955` | **LAYER_CAPS 仍然改變輸出** |
| `capsA` vs `canonA`（同 caps，canon off vs on） | 884 | `eq_eq=0, ne_ne=829` | canon 確實生效（改了數值，如設計） |
| `capsE2` vs `canonE2` | 940 | `eq_eq=0, ne_ne=887` | 同上，在 E2 caps 下也生效 |

⇒ **固定 k=8 那條鏈的歸約順序，既沒縮小分歧、也沒改變它的形狀。**

## 2. 被否證的載體（兩個，都是量出來的）

**(a) 張量形狀／`ne[2]` 本身不是載體。**
decode 路徑把 expert tensor 的專家軸設成該層 slot 數（`llama-context.cpp:7045`
`wt->ne[2] = llama_expert_cache_slots_per_layer_l(...)`），所以 caps 確實改形狀。但**pool 大小改得更兇**
而完全不動數值：

| 臂 | 每層 slot（launch log 原文） | 總數 | vs capsA |
|---|---:|---:|---|
| capsA（8 GiB） | `avg 145.8/layer, min 143/layer` | 5976 | — |
| **pool4（4 GiB）** | `avg 75.5/layer, min 71/layer` | 3096 | **884/884 逐位元相同** |
| `pool4b`（同配置重跑） | 同上 | 3096 | 相同（null 通過） |

每層 `ne[2]` 砍半、輸出逐位元相同 ⇒「形狀變 ⇒ 歸約樹變 ⇒ 數值變」不是完整解釋。

**(b) layer 40（MTP head）不是載體，兩個方向都排除了。** E2 字串裡唯一不是 trunk 重分配的一項是
`40-40:256`：

| 臂 | caps 內容 | vs capsA |
|---|---|---|
| `e2only40` | **只** `40-40:256` | **884/884 相同**（且總數仍是 5976 ⇒ 這根本沒改到值，L40 本來就是 256） |
| `e2noh40` | E2 全部重分配**扣掉** layer 40 | **1/884 分歧**，`ne_dec_eq=24, ne_ne=859` |

而且 `capsA_vs_e2noh40` 與 `capsA_vs_capsE2` 的 **onset 逐值相同**（index 1、
`sum` 50513.5641 vs 60725.3806、`argmax` 109705、`n_cand` 1333/1333）⇒
**E2 的分歧全部來自 trunk 重分配，L40 貢獻 0。** `capsE2 vs e2noh40` 是 940/940 相同，佐證同一件事。

**(c) 「同樣總量、換成均勻分配」也不是解。** `0-40:163`（總數 6683，與 e2noh40 的 6703 同量級、
均勻）對 capsA 仍**分歧**；只改 trunk 前半 `0-7`（總數 6454）也**分歧**。

## 3. §EN-241(c) 的「20% logits 差」要更正

原文用 `sum` 差 ~20%（`A=50513.56 B=60725.38`）推論「不只是最後一位的捨入」。那個 20% 是
**248,320 個有正負項的總和**的相對差：除以 `n_vocab` 後，首個分歧行的每個 logit 差是 **0.0411**，
而且該行的 `argmax` 兩臂相同（109705）、`n_cand` 相同（1333）。所以那句話的方向對
（不是最後一位），但**量級被簽號相消放大成 20%**，引用時必須換算成 per-logit 值。

首次分歧的形狀（capsA vs e2noh40，884 條共同鍵）：**index 0 逐位元相同 → index 1 出現數值差 →
index 5 出現 argmax 差**。這是「一次注入、決策後續被混沌放大」的形狀，不是尾端漂移。

## 4. 這對 §EN-242 的「① 槽位幾何」意味著什麼

- §EN-241 否決的是「把它當**等價**旋鈕」；本輪進一步說明：**它也不是用歸約順序能修的**。
- cb share 16.4% → 13.2% 那一份**仍然拿不回來**（它需要 caps 這個配置本身），而且現在知道
  「拿回來」不會來自 canonical order。
- 「固定歸約順序」仍然有價值，但它的定位回到 09-17 狀態文件 §5 講的那個：**為 ids 由 slot 順序
  衍生的未來架構準備**，不是修當前這顆跨配置漂移。

## 5. 下一個可否證的實驗（唯一還沒排除的機制族）

剩下的候選只有一類：**caps 改變了「每層走哪條計算路徑」**（pool 路徑 vs overflow/wide 路徑、
常駐 vs 冷讀），而不是改變形狀或順序。判別方式（每個臂 ~90 秒）：

1. `CGC_POOL_SPLIT_DBG=1` 在 capsA 與 `unif163` 各跑一次，比較逐層的 `mode=pool|wide` 與 `ne2=`
   （注意該 DBG 目前只印 `il <= 2`；要涵蓋全部 41 層需要先把那個條件放寬，否則只看得到前 3 層）。
2. 若兩臂的逐層 mode 不同 ⇒ 載體是**路徑選擇**；修法是把路徑做成 caps 無關（同一層永遠走同一條），
   而不是修順序。
3. 若逐層 mode 相同 ⇒ 載體是**同一條路徑內、由常駐集合決定的資料流**（例如冷讀填入 vs 直讀），
   那就必須在該路徑內部找，`LAYER_CAPS` 就只能是「另一種配置」。

**驗收線**：任何修法都要能讓 884 步長 probe 上 `capsA` vs 該修法下的 E2 caps 得到 M1 **884/884**，
並且 null control（同配置重跑）維持 1003/1003 或 884/884。

## 6. 產物與重跑

| 檔案 | 內容 |
|---|---|
| `Backup/canon_caps_longprobe_20260919.sh` | canon × caps 三臂（canonA/canonA2/canonE2）＋四組比對 |
| `Backup/pool_size_longprobe_20260919.sh` | pool 8 vs 4 GiB 兩臂（pool4/pool4b）＋三組比對 |
| `Backup/layer40_caps_split_20260919.sh` | layer 40 切分（e2noh40/e2only40）＋三組比對 |
| `Backup/caps_pattern_split_20260919.sh` | 樣式 vs 壓力（unif163/trunk07）＋三組比對 |
| `Backup/phase_decomp/oracle_long_canon*.jsonl` | canon 三臂 dump（1003/1003/968 條） |
| `Backup/phase_decomp/oracle_long_pool4*.jsonl` | pool 4 GiB 兩臂 dump（884 條） |
| `Backup/phase_decomp/oracle_long_e2{noh40,only40}_20260919.jsonl`、`unif163`、`trunk07` | 切分臂 dump |
| `Backup/phase_decomp/oracle_long_*_vs_*.json` | 十二組比對報告（**每一份都帶 engine 身分**，`engine_identity` 蓋章） |

```sh
# 全部都是 detached 執行，因為每臂要載入模型（暖頁面下 ~90 s，冷則數分鐘）
python3 /tmp/detach.py /tmp/canonprobe.log bash Backup/canon_caps_longprobe_20260919.sh
python3 /tmp/detach.py /tmp/poolprobe.log  bash Backup/pool_size_longprobe_20260919.sh
python3 /tmp/detach.py /tmp/l40probe.log   bash Backup/layer40_caps_split_20260919.sh
python3 /tmp/detach.py /tmp/patprobe.log   bash Backup/caps_pattern_split_20260919.sh
```

⚠ 誠實邊界：所有 caps 臂都是**單次**，沒有各自的 null control；可歸因性建立在
「同配置重跑逐位元相同」已在 capsA（A vs A2 = 884/884）與 pool4（pool4 vs pool4b）上各自成立，
以及本引擎在固定配置下完全確定性（§EN-241(e)）。若要發表，每個 caps 臂應各補一次重跑。
