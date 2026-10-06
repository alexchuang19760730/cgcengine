# 本線 MTP-on 現況 ＋ 與其他線的整合點（2026-10-06）

**觸發**：operator「目前本線 mtp on 狀況如何，怎麼跟其他線整合」。
**一句話**：本線（S／口徑線的 spec-tree 寬-2 分支）的 MTP-on **機制可用、指標未過**；
它的**打平判準已經機器化**（`mtpon_tie_gate.py`），缺的只是**一個乾淨窗**。
整合的關鍵是：這道閘的判詞**就是 M 軸（L25-4）reopen 條件的 counter 版**，
且它與交付線（lineA 的 `exp-spectree-mtpon-default-2026-10-06`）應該**共用同一 session**。

---

## 1. 本線 MTP-on 現況

### 1.1 唯一的判準（寫死進 gate）

```
on 打平 off  ⟺  emit_on · T_off  ≥  R_on
gain = emit_on · T_off / R_on  ≥ 1
```

- `T_off` = off 的 ms/token（純解碼 1 token/步）
- `R_on`  = on 的 ms/verify 輪（**含 replay 輪**）
- `emit_on` = `emit_per_round`（counter，`CGC-BENCH-ACCEPT`，**不引用 t/s**）
- 兩側**同 session**（AB ＋ BA 兩跑序）⇒ `gain_bal` ＝ 兩跑序的幾何平均；
  `margin = max(--margin-min 2%, 本 session 的跑序散度)` ⇒ 跑序效應與時代差都被式子吸收。
- 機制閘（不過 ⇒ VOID 不給判詞）：`replay/rounds<10%`、`trim_fail=0`、`emit>0`、
  每臂 `rc==0`、無 STALL/TIMEOUT、盒閘起跑前綠（或有 `--waive-reason` 留痕）。
- **形狀閘**：`n_prompt>0` 一律拒收（llama-bench 把 prompt run 放在計時區內，
  `llama-bench.cpp:4072` 在 `:4065` 之後 ⇒ avg_ns 含 prefill）。解碼支柱必須是 `-p 0 -n N -d D`。

### 1.2 counter 事實（真文字輪，19 臂，`Backup/spec_tree/T1W2_REALPROMPT_2026-10-06.md`）

| config | n | live emit/round | p1A（head top-1） | ceilA | 備註 |
|---|---|---|---|---|---|
| W1 real | 5 | **2.039** [1.985,2.098] | 0.696 | **2.047** [2.016,2.082] | live≈ceil ⇒ head-limited |
| W1 rand | 5 | 2.211 [2.048,2.633] | 0.752 | 2.302 [2.126,2.710] | 舊 2.707 ＝此分布上緣 |
| W2 real | 4 | 1.777 [1.438,2.246] | **0.247** | 1.294 [1.079,1.549] | p1B 0.415 ⇒ **A 塌陷、亞軍鏈 B 救援** |
| W2 rand | 5 | 2.216 [1.803,2.667] | 0.704 | 2.117 [1.437,2.686] | 2/5 也落塌陷 regime |

- 真文字把 W1 天花板從儀器讀到的 **2.707** 拉低到 **~2.05**（完全分離，排列單尾 p≈0.40%）
  ⇒ 「headroom 在 head」定性不變，**但量級一直被 prompt-blind 儀器高估 ~0.65**。
- **W2 現行是「亞軍救援」，不是品質槓桿**（真文字 4/4 A 塌陷；一趟 emit 是可讀重複迴圈）⇒ **先修 A 塌陷再談寬-2**。
- frontend 誤差：判準單位＝delivery cell 的 `token/decode`（**分母含 replay 輪**）；樹臂 replay=0 ⇒
  **不可互換**，只能用上面的原始式。

### 1.3 瓶頸定位：是 `R_on`（每輪成本），不是 head

同 session ABBA 三臂拆解（`t1w2_cost_split.py`，診斷形狀）：

| 項 | 每輪成本 | 對照 |
|---|---|---|
| context overrides（`kv_unified`/`n_rs_seq=2`/`n_seq_max=2`） | **+8.2 ms（≈+4%）** | B−A |
| tree code（batch/draft 建構、walk、ctx_dft 鏡像裁切） | **+17.1 ms（≈+3.4%）** | C−B |
| 樹 vs 鏈（同 overrides） | 差 **0.4%** | 樹 ≈ 鏈 |

⇒ 先前「tree 多 ~100 ms/輪」是**時代差**，今天不是。而寬-2 的每輪成本只比鏈多 **~6%**
（`T1C_BAR_205_DERIVATION` §2，實測 1.06，非外推的 1.416）⇒ **原判準 2.05 高估 25–30%，修正後 ≈1.5–1.6**。

### 1.4 閘的狀態

- `scripts/check/mtpon_tie_gate.py`：`--check / --go / --watch / --replay / --selftest`；
  **selftest 29 項 PASS**（純函式＋子程序；不吃 GPU）；exit **0=TIE、1=SHORTFALL、2=拒跑/VOID**。
- **尚未取得乾淨窗正式判詞**：起跑前後盒閘皆 **SHORT**（`l255_close.py`：可回收 4780 MB
  vs 存活線 7703 MB；swap 存量 0 MB、thermal 0、壓縮機 quiet、埠空、無 foreign llama）。
- 盒子綠即跑：`--watch`（預設最長等 180 分，超時不產量測）；不綠硬跑：`--go --waive-reason`。

---

## 2. 與其他線的關係

| 線 | 這是什麼 | 與本線的關係 |
|---|---|---|
| **M 軸（L25-4）** | MTP 軸本體；k 軸經濟學已判 **DEAD**（結案·排除） | 本線是它的**重開工程 #2**（spec-tree 寬 2）；#1（分類修正 caps≥17）已由 `e-mtp-cb-classify-2026-10-03` 處理 |
| **L20（20+）／交付格** | 交付格 (default) ＝ `-p512 -n128 -b2048 -ub512 -d0 -r5`；靠 overlap fence 抬到 **C14 13.722**（OFF 10.840，×1.266） | fence 是 **S 軸**槓桿、spec-tree 是 **M 軸**槓桿；**兩者都在同一格上量 ⇒ 必須明講 ON 臂是哪個 ON** |
| **lineA（加速交付）** | `exp-spectree-mtpon-default-2026-10-06.yaml`：同一件事的**交付口徑**版（temp=0.4、釘 seed、取真 tg 配對） | **主要整合點**：本線的 counter 判詞 ＋ 它的 t/s 判詞，應來自**同一 session** |
| **shared（口徑）** | `exp-mtp-onoff-standard.yaml`；`quote_gate`／`caliber_certify`／`decode_board` | 閘用 counter 端點 ⇒ **不受窗口漂移**；t/s 一律要過 `quote_gate` |
| **棘輪／已認證表** | `certify_anchor.decode_ts=15.210976`（**已退役形狀**）；乾淨窗最高 **C12 13.8674** | 本線判詞若要做成 t/s，只能進 (default) 格，且**不推全局棘輪** |

---

## 3. 四個具體接點（建議）

1. **Gate 判詞 ＝ L25-4 reopen 條件的 counter 版。**
   現在的 reopen 條件是「乾淨窗（`none` ∧ ≤1.10 ∧ reps3）成對出現 `B_tps ≥ A_tps`」——
   那是 **t/s**，要過 quote_gate 且被窗口漂移綁死。本閘給的是同一件事的 **counter** 版
   （`gain ≥ 1`，同 session ABBA）。⇒ 建議在 L25-4 的 reopen 條件**補一條等價的 counter 條件**，
   讓「有沒有打平」不必等一個 t/s 乾淨窗。

2. **同一 session 餵兩張卡（counter ＋ delivery）。**
   `exp-spectree-mtpon-default` 要的是 (default) 格、temp=0.4、釘 seed 的 tg 配對。
   應讓它與本閘**共用 cell／seed／跑序**：閘輸出 `emit_on/T_off/R_on`（判詞），
   卡輸出兩臂 tg（判詞）。兩者方向若不一致，就是 **bench vs server 結構差**又現形。

3. **交付格上的 2×2（fence × MTP-on）尚未做。**
   交付格現在的 ON 是 fence-ON（S 軸）。spec-tree MTP-on 從未在同一格上與它共存量過
   ⇒ 「spec-tree 是否推進交付」目前只能答「機制 work、指標未過」。
   一次 2×2（off／fence／mtp／fence+mtp）就能把兩軸的**可加性**一次問清楚（注意：S 與 M 收益不可假設可相加）。

4. **結構差列為待裁決，不當結論。**
   bench 2.039 vs server 2.50（k=3、2025-token 真 prompt）差在：bench 第一輪 draft 的 `pending_h=0`、
   且 prompt 從不進 draft 側（`common_speculative_process` 唯一呼叫點在 verify batch，`llama-bench.cpp:3441`）。
   ⇒ 要由 **server 同 seed 配對**（`exp-mtp-onoff-standard` 的 shared 口徑）回答，不從 bench 推。

---

## 4. 誠實邊界

- 閘的判詞尚未在乾淨窗取得；smoke 判詞（形狀極小、非 house cell）**不可引用**。
- 全部本線證據都是 **counter**；t/s 一律不可引用（窗口/時代不穩）。
- W2 真文字只有 4 條軌跡 ⇒ 塌陷是**傾向**（4/4 vs 2/5），不是分離；W1 才是分離。
- on/off 比值**時代矛盾未裁決**（10-01 鏈 −5%／1.05 vs 10-06 鏈/樹 ≈0.744）——
  正是本閘要用「同 session ABBA」解掉的那個問題。
- `--spec-type draft-mtp` 是 harness 全域旗標 ⇒ on/off 分兩次 invocation（repo 既有慣例）。

## 5. 產物

- `scripts/check/mtpon_tie_gate.py`（本閘；**未 commit**）
- `Backup/spec_tree/T1W2_REALPROMPT_2026-10-06.md`（真文字 19 臂）
- `Backup/spec_tree/T1C_BAR_205_DERIVATION_2026-10-06.md`（判準重推導＋成本拆解）
- `docs/L254_K_AXIS_ECON_2026-10-01.md`（k 軸 DEAD ＋ reopen 條件）
- `scripts/check/charters/exp-spectree-mtpon-default-2026-10-06.yaml`（交付口徑卡）
- `scripts/check/decode_board_2026-09-29.yaml`（L25-4／C14／棘輪的單一來源）
