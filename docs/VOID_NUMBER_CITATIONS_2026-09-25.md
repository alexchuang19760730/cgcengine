# 作廢數字的「引用」檔案（2026-09-25）

> 本檔是 `docs/MEASUREMENT_CONTRACT_2026-09-25.md` **§7 作廢數字登記表**的配套，回答兩件事：
> **① 憑什麼廢**（§1，逐字引用原始依據，可自行核對）**② 誰還在引**（§3，全庫盤點）。
> 機檢：`scripts/check/void_number_check.py`（selftest **12/12**）。

## §0 一句話

`9.82`／`12.62`／`+28.5%` 是 **HTTP／`llama-speculative-simple`、無 warm-skip** 的舊口徑產物；
而**同 cell 的生產口徑 MTP off 實測已是 12.05 ± 0.57（09-24 `MEMORY_PERF.md:147-148`）**、
**今日再測 11.03~12.20** ⇒ **分母本身失效**，比值無意義。

⚠ 而且**這不是今天才發現**：`MEMORY_PERF.md:146` 在 **2026-09-24 20:56** 就寫過
「在本線交付口徑下不可再引用」——**但它仍在索引與多份判決裡流通**。
⇒ 這正是「作廢數字必須可機檢」的理由：靠人記必失效。

## §1 作廢依據（逐字引用）

### 1.1 口徑歸屬：`12.62` 不屬於落地儀器
> 出自 `docs/G7_DELIVERY_ARITHMETIC_2026-09-20.md:19-22`
>
> - `12.62` 屬 **HTTP／`llama-speculative-simple`**（`docs/MTP_ABBA_RECHECK_2026-09-18.md:12`
>   自己寫的），而 G7 的 `instrument` 欄寫的是 `prod_matrix.py (llama-bench, BOTH axes)`。
>   **⇒ 這個閘門的成功判準引用了它自己聲明不用的那台儀器的基準。**
> - 落地儀器上的記錄值是 **10.78–10.98**。⇒ 需要的倍率不是 25/12.62 = **1.98×**，而是 **2.28–2.32×**。

### 1.2 它「既沒有複現、也沒有被證偽」
> 出自 `docs/MTP_ABBA_RECHECK_2026-09-18.md:73-76`
>
> ⇒ **12.62 既沒有複現、也沒有被證偽**：它與 10.x 的 21% 差異，**落在本輪量到的環境噪聲量級之內**
> ⇒ **用當前的測量方法判不出來。** 這不是「不知道就算了」——它是一個**可操作的結論**：
> 在噪聲降到 ~5% 之前，任何 MTP on/off 的 t/s 結論都是不可信的。
>
> 同文件 `:21-22` 另給了同一現象的規模：**負載 128,920 請求相同、build 相同**，t/s 卻是 **7.89 vs 9.62**。

### 1.3 ★ 09-24 就已裁定「不可再引用」，並給了同 cell 的實測替代
> 出自 `.workbuddy/memory/MEMORY_PERF.md:146-151`
>
> - ⚠ **[2026-09-24 20:56] 「MTP ×1.28（9.82 → 12.62）」在本線交付口徑下不可再引用。**
>   實測（同 cell prod-new／b=ub 5632／ctx 0／**warm-skip 64**／r3／NOMINAL，唯一變數＝儀器）：
>   **MTP off 乾淨 = 12.05 ± 0.57 t/s**，已貼近交付 anchor 12.57 ⇒ 本口徑下 MTP 只剩 **+4.3%**。
>   9.82 vs 12.62 是**無 warm-skip 的舊口徑**（HTTP／`llama-speculative-simple` 路徑），
>   與 cold-start 混雜；舊口徑的單臂 sd 是 **±3.54（35%）**，warm-skip 後掉到 **±0.57（4.8%）**
>   ⇒ 那 1.28× 裡有冷啟動成份。**MTP 增益必須在 warm-skip 口徑下重測**。

★ 這條比本輪更早：**同一數字在 09-24 已被降級**，卻仍在索引與判決流通 ⇒ 本輪把它升為**登記表**（唯一入口）。

### 1.4 生產 server 路徑方向相反（`×0.695`）
> 出自 `.workbuddy/memory/MEMORY_PERF.md:402-405`
>
> **★★★★ 09-18 12:1x 裁決（同檔 ABBA、生產路徑）：MTP 是 ×0.695，不是 ×1.0。**
> 新臂 `p25-nail-mtpoff`（釘 `CGC_SERVER_MODEL=Nail-…denseIQ4X.gguf` ＋ `MTP=0` ⇒ **同檔**）
> vs `p25-mtp-on`，`--profile prod25 --n-predict 96 --rounds 3`：
> 前向（ON 先）9.40/11.28 = **0.833**、反向（OFF 先）6.32/10.91 = **0.579**
> ⇒ **ABBA 校正 M = √(0.833×0.579) = 0.695**（兩個順序都 < 1）

⇒ 「MTP on 一定更快」在生產路徑上**至少不成立**；方向本身未定 ⇒ 不能拿舊比值當決策前提。

### 1.5 今日生產口徑（operator 的判準：「我們現在 MTP off 就已經 11+」）

| 讀數 | 值 | 產物 | 口徑 |
|---|---|---|---|
| MTP off（今日） | **11.034**（sd 0.177） | `Backup/nofill_prod/nf2_fill.json` | llama-bench、prod-new cell、warm-skip 64 |
| MTP off（今日） | **11.793**（sd 0.217） | `Backup/nofill_prod/nf_fill.json` | 同上 |
| MTP off（今日） | **12.195**（sd 0.272） | `Backup/mw_ab/mw_ctrl.json` | 同上 |
| MTP off（09-24 同 cell） | **12.05 ± 0.57** | `MEMORY_PERF.md:147-148` | 同上（**warm-skip 口徑**） |
| MTP off（交付錨點） | **12.57** | `docs/G7_DELIVERY_ARITHMETIC_2026-09-20.md` | 同上；★ **不作廢**，作為「目前最好」 |

⇒ 分母已是 11~12.6，而作廢的分子/分母是 9.82/12.62 ⇒ **比值不可用**。

## §2 引用格式（要引用時怎麼寫）

| 情境 | 寫法 |
|---|---|
| ⛔ **禁止** | 把「MTP off 9.82 → on 12.62（+28.5%）」當**增益**寫進判決／報告／算術 |
| ✅ 歷史敘述 | 「MTP off ~~9.82~~ → on ~~12.62~~（**+28.5% 已作廢**，見 `MEASUREMENT_CONTRACT` §7）」——**同段必須帶標記** |
| ✅ 現況寫法 | 「生產口徑 MTP off = **11.03~12.20**（`nf2_fill`／`nf_fill`／`mw_ctrl`）；**交付 cell 的 MTP on/off 配對＝ UNRESOLVED（待以 warm-skip 口徑掃 k）**」 |
| ✅ 需要數字 | 用替代：增益約 **+4.3%**（12.05 對 anchor 12.57，**非同 launch 配對 ⇒ 必須標為估算**） |

**三條硬規則**
1. **提及即標記**（同一段內 `作廢`／`⛔`）——機檢才不會紅。
2. **不得把作廢數字放進算術**（乘數／比值／上界）——這是它最常被誤用的方式
   （例：`9.82 × 1.28 × 2.0 = 25.1` 這種「剛好擦線」的算術）。
3. **要下結論 ⇒ 用替代值，或明寫 UNRESOLVED**；不可用「大概」「趨勢上」帶過。

## §3 引用盤點（**292 處／51 檔**，由 `--citations` 生成）

- **決策面 55 處**（已標記 55／未標記 0）——這是**強制**面，必須全綠。
- **歸檔面 237 處**——dated 產物與逐日誌，依專案慣例**不回改**，但**一律不得再被引用**。

### 3.1 分佈（前 15 名）

| 檔案 | 處 |
|---|---|
| `.workbuddy/memory/2026-09-17.md` | 23 |
| `agent_harness/memory/2026-09-17.md` | 23 |
| `.workbuddy/memory/MEMORY_PERF.md` | 22 |
| `.workbuddy/memory/2026-09-25.md` | 17 |
| `agent_harness/memory/MEMORY_PERF.md` | 14 |
| `.workbuddy/memory/2026-09-24.md` | 10 |
| `docs/ROUTING_TRACE_2026-09-17.md` | 10 |
| `.workbuddy/memory/MEMORY.md` | 9 |
| `.workbuddy/memory/2026-09-18.md` | 9 |
| `agent_harness/memory/2026-09-18.md` | 9 |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 9 |
| `docs/BIGGER_FISH_2026-09-22.md` | 8 |
| `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` | 8 |
| `docs/MEASUREMENT_CONTRACT_2026-09-25.md` | 7 |
| `.workbuddy/memory/2026-09-22.md` | 7 |

### 3.2 決策面（55 處）

| 檔案 | 行 | 數字 | 已標記 | 上下文 |
|---|---|---|---|---|
| `.workbuddy/memory/MEMORY.md` | 50 | `12.62` | ✅ | …MTP／speculative」。   ⛔ **「MTP off 9.82 vs on 12.62 ＝ +28.5%」已作廢**（2026-09-2… |
| `.workbuddy/memory/MEMORY.md` | 147 | `12.62` | ✅ | …5 / ③b 3 / ④ 8 ⇒ 本線沒有 ①。**    （MTP 那格因 `9.82/12.62/+28.5%` 作廢，由 ③b 降為 ③a。）… |
| `.workbuddy/memory/MEMORY.md` | 156 | `12.62` | ✅ | …/8 紅＝執行線未接線）。 ⑧ **作廢數字登記表（契約 §7）**：`9.82` / `12.62` / `+28.5%` **不得作決策依據**… |
| `.workbuddy/memory/MEMORY.md` | 50 | `28.5` | ✅ | …ulative」。   ⛔ **「MTP off 9.82 vs on 12.62 ＝ +28.5%」已作廢**（2026-09-25 operato… |
| `.workbuddy/memory/MEMORY.md` | 147 | `28.5` | ✅ | …3 / ④ 8 ⇒ 本線沒有 ①。**    （MTP 那格因 `9.82/12.62/+28.5%` 作廢，由 ③b 降為 ③a。）    無實測權… |
| `.workbuddy/memory/MEMORY.md` | 156 | `28.5` | ✅ | …）。 ⑧ **作廢數字登記表（契約 §7）**：`9.82` / `12.62` / `+28.5%` **不得作決策依據**    （HTTP 舊口… |
| `.workbuddy/memory/MEMORY.md` | 50 | `9.82` | ✅ | …PERF.md`「## MTP／speculative」。   ⛔ **「MTP off 9.82 vs on 12.62 ＝ +28.5%」已作廢*… |
| `.workbuddy/memory/MEMORY.md` | 147 | `9.82` | ✅ | …/ ③a 5 / ③b 3 / ④ 8 ⇒ 本線沒有 ①。**    （MTP 那格因 `9.82/12.62/+28.5%` 作廢，由 ③b 降為… |
| `.workbuddy/memory/MEMORY.md` | 156 | `9.82` | ✅ | …（7/7，目前 4/8 紅＝執行線未接線）。 ⑧ **作廢數字登記表（契約 §7）**：`9.82` / `12.62` / `+28.5%` **不… |
| `.workbuddy/memory/MEMORY_HYGIENE.md` | 199 | `12.62` | ✅ | …ll.json`、`Backup/mw_ab/mw_ctrl.json`） / / **`12.62`**（MTP on） / 同口徑；`MTP_AB… |
| `.workbuddy/memory/MEMORY_HYGIENE.md` | 203 | `12.62` | ✅ | …點）**不作廢** ⇒ **「目前最好」一律以 12.57 為準**，不含 MTP-on 12.62。 機檢 `scripts/check/void_… |
| `.workbuddy/memory/MEMORY_HYGIENE.md` | 200 | `28.5` | ✅ | …反而是 **×0.695** / **無** ⇒ 交付 cell 缺量測 / / **`+28.5%`** / 兩個作廢數字之比 / **UNRESO… |
| `.workbuddy/memory/MEMORY_HYGIENE.md` | 198 | `9.82` | ✅ | …批三筆：  / 作廢數字 / 為什麼 / 替代 / /---/---/---/ / **`9.82`**（MTP off） / 09-17 **HTT… |
| `.workbuddy/memory/MEMORY_PERF.md` | 146 | `12.62` | ✅ | …。 - ⚠ **[2026-09-24 20:56] 「MTP ×1.28（9.82 → 12.62）」在本線交付口徑下不可再引用。**   實測（同… |
| `.workbuddy/memory/MEMORY_PERF.md` | 149 | `12.62` | ✅ | …hor 12.57 ⇒ 本口徑下 MTP 只剩 **+4.3%**。   9.82 vs 12.62 是**無 warm-skip 的舊口徑**（HT… |
| `.workbuddy/memory/MEMORY_PERF.md` | 159 | `12.62` | ✅ | ….62 t/s（accept 73.81%）｜   MTP on nb-aware 修正 12.62 t/s（accept 58.25%）** ⇒ *… |
| `.workbuddy/memory/MEMORY_PERF.md` | 166 | `12.62` | ✅ | …的是 `MTP-on ≥ MTP-off`   （6.74 < 8.87）**」⇒ 今天 12.62 ≥ 9.82 ⇒ **M4 的功能缺口關閉**。… |
| `.workbuddy/memory/MEMORY_PERF.md` | 287 | `12.62` | ✅ | …三次 **1.1%**）。   **⚠ 未決：MTP 不在這個口徑裡** —— 歷史的 `12.62 vs 9.82` 是 HTTP／`llama-s… |
| `.workbuddy/memory/MEMORY_PERF.md` | 382 | `12.62` | ✅ | …token / **1.0**（MTP off） / ⛔ MTP-on `9.82 → 12.62`（×1.28）**已作廢**（契約 §7）⇒ 交… |
| `.workbuddy/memory/MEMORY_PERF.md` | 389 | `12.62` | ✅ | …5.1` ⇒ **剛好擦線**。而： - ⚠️ **兩個乘數能不能相乘還沒量過** —— 12.62 與 16.82 是**不同儀器**量的。本檔早已… |
| `.workbuddy/memory/MEMORY_PERF.md` | 400 | `12.62` | ✅ | …路互相印證 0.4%） / / MTP（每步 token） / ⛔ ×1.28（9.82→12.62）**已作廢**（契約 §7） / **×1.06… |
| `.workbuddy/memory/MEMORY_PERF.md` | 586 | `12.62` | ✅ | …⛔ **2026-09-25 operator 下令：本節（及全檔）的 `9.82`／`12.62`／`+28.5%` 一律不得作為決策依據** >… |
| `.workbuddy/memory/MEMORY_PERF.md` | 159 | `28.5` | ✅ | …/s（accept 58.25%）** ⇒ **MTP-on ≥ MTP-off 達成（+28.5%）**，   修前是淨負。★ **accept 下… |
| `.workbuddy/memory/MEMORY_PERF.md` | 586 | `28.5` | ✅ | …-09-25 operator 下令：本節（及全檔）的 `9.82`／`12.62`／`+28.5%` 一律不得作為決策依據** > —— HTTP… |
| `.workbuddy/memory/MEMORY_PERF.md` | 1238 | `28.5` | ✅ | …0 只建 1 個 CPY，否則 K=4）⇒ 30 層×4=120。但 MTP on 是 +28.5%   ⇒ 別動。折疊也❌：K 個 CPY 寫進 K… |
| `.workbuddy/memory/MEMORY_PERF.md` | 95 | `9.82` | ✅ | …14 的 build，而今天同類量測（Nail、8 GiB、MTP off）   是 **9.82**（`ROUTING_TRACE` §14.2），… |
| `.workbuddy/memory/MEMORY_PERF.md` | 146 | `9.82` | ✅ | …E` 的學費）。 - ⚠ **[2026-09-24 20:56] 「MTP ×1.28（9.82 → 12.62）」在本線交付口徑下不可再引用。**… |
| `.workbuddy/memory/MEMORY_PERF.md` | 149 | `9.82` | ✅ | …貼近交付 anchor 12.57 ⇒ 本口徑下 MTP 只剩 **+4.3%**。   9.82 vs 12.62 是**無 warm-skip 的… |
| `.workbuddy/memory/MEMORY_PERF.md` | 158 | `9.82` | ✅ | …l 8 GiB，`n_predict=96`，每臂 3 請求）：   **MTP off 9.82 t/s｜MTP on 修前（`CGC_IDS_LI… |
| `.workbuddy/memory/MEMORY_PERF.md` | 166 | `9.82` | ✅ | …on ≥ MTP-off`   （6.74 < 8.87）**」⇒ 今天 12.62 ≥ 9.82 ⇒ **M4 的功能缺口關閉**。仍未處理：`pl… |
| `.workbuddy/memory/MEMORY_PERF.md` | 287 | `9.82` | ✅ | …**）。   **⚠ 未決：MTP 不在這個口徑裡** —— 歷史的 `12.62 vs 9.82` 是 HTTP／`llama-speculativ… |
| `.workbuddy/memory/MEMORY_PERF.md` | 382 | `9.82` | ✅ | …/ / 每步 token / **1.0**（MTP off） / ⛔ MTP-on `9.82 → 12.62`（×1.28）**已作廢**（契約… |
| `.workbuddy/memory/MEMORY_PERF.md` | 388 | `9.82` | ✅ | …** ⇒ **那個數字幾乎全部住在「每步 token 數」那一項**。  **算術**：`9.82 × 1.28（MTP，已量）× 2.0（去序列化，… |
| `.workbuddy/memory/MEMORY_PERF.md` | 400 | `9.82` | ✅ | …um` 兩路互相印證 0.4%） / / MTP（每步 token） / ⛔ ×1.28（9.82→12.62）**已作廢**（契約 §7） / **… |
| `.workbuddy/memory/MEMORY_PERF.md` | 586 | `9.82` | ✅ | …為權威）  > ⛔ **2026-09-25 operator 下令：本節（及全檔）的 `9.82`／`12.62`／`+28.5%` 一律不得作為決… |
| `.workbuddy/memory/MEMORY_S1.md` | 479 | `12.62` | ✅ | …mit 的實測）：**M1/M2/M3 對 v6 全 9/9；decode 8.62 → 12.62 t/s（⛔ `12.62` 已作廢，契約 §7）… |
| `.workbuddy/memory/MEMORY_S1.md` | 479 | `12.62` | ✅ | …/M2/M3 對 v6 全 9/9；decode 8.62 → 12.62 t/s（⛔ `12.62` 已作廢，契約 §7）； prefill 命中率… |
| `docs/MEASUREMENT_CONTRACT_2026-09-25.md` | 137 | `12.62` | ✅ | …ode 比目前最好還要好**（目前最好＝交付錨點 **12.57**；⛔ MTP-on `12.62` **已作廢、不算「目前最好」**，見 §7）… |
| `docs/MEASUREMENT_CONTRACT_2026-09-25.md` | 180 | `12.62` | ✅ | …ch、prod-new cell、MTP off、warm-skip 64 / / **`12.62 t/s`**（MTP on） / 同上（09-1… |
| `docs/MEASUREMENT_CONTRACT_2026-09-25.md` | 181 | `12.62` | ✅ | …P-on 讀數**缺量測（UNRESOLVED）** / / **`+28.5%`**（＝12.62÷9.82） / 兩個已作廢數字之比 / 分子分母… |
| `docs/MEASUREMENT_CONTRACT_2026-09-25.md` | 187 | `12.62` | ✅ | …交付口徑。   ⇒ **「目前最好」一律以 `12.57` 為準，不含 MTP-on `12.62`。** - 新增作廢數字 ⇒ 只能加到本表（含理… |
| `docs/MEASUREMENT_CONTRACT_2026-09-25.md` | 181 | `28.5` | ✅ | …cell 的 MTP-on 讀數**缺量測（UNRESOLVED）** / / **`+28.5%`**（＝12.62÷9.82） / 兩個已作廢數… |
| `docs/MEASUREMENT_CONTRACT_2026-09-25.md` | 179 | `9.82` | ✅ | …） / 作廢理由 / 替代（現行權威） / /---/---/---/---/ / **`9.82 t/s`**（MTP off） / 09-17，*… |
| `docs/MEASUREMENT_CONTRACT_2026-09-25.md` | 181 | `9.82` | ✅ | …數**缺量測（UNRESOLVED）** / / **`+28.5%`**（＝12.62÷9.82） / 兩個已作廢數字之比 / 分子分母皆作廢 ⇒… |
| `docs/MILESTONE_MAP_RECHECK_2026-09-25.md` | 131 | `12.62` | ✅ | …只看這一格**：decode 有沒有超越目前最好 **12.57**（⛔ MTP-on `12.62` **已作廢、不算最好**，見契約 §7） /… |
| `docs/MILESTONE_MAP_RECHECK_2026-09-25.md` | 133 | `12.62` | ✅ | …**13~17 t/s** / ⛔ 原寫「交付口徑 MTP off 9.82 → on 12.62（**+28.5%**），已在生產級設置」——**… |
| `docs/MILESTONE_MAP_RECHECK_2026-09-25.md` | 133 | `28.5` | ✅ | …t/s** / ⛔ 原寫「交付口徑 MTP off 9.82 → on 12.62（**+28.5%**），已在生產級設置」——**三個數字全部作廢*… |
| `docs/MILESTONE_MAP_RECHECK_2026-09-25.md` | 133 | `9.82` | ✅ | …代入交付 acc ⇒ **13~17 t/s** / ⛔ 原寫「交付口徑 MTP off 9.82 → on 12.62（**+28.5%**），已在… |
| `docs/MTP_AMORTIZATION_2026-09-25.html` | 127 | `12.62` | ✅ | …數 ≈ 0</b>（本頁 §1）；⛔ <s>交付口徑 MTP off 9.82 → on 12.62 = +28.5%</s> —— <b>此比值已作… |
| `docs/MTP_AMORTIZATION_2026-09-25.html` | 127 | `28.5` | ✅ | …（本頁 §1）；⛔ <s>交付口徑 MTP off 9.82 → on 12.62 = +28.5%</s> —— <b>此比值已作廢</b>（HTT… |
| `docs/MTP_AMORTIZATION_2026-09-25.html` | 131 | `28.5` | ✅ | …MTP-off 提升 ≥ 門檻</b>（門檻待 operator 定；⛔ 原寫「目前 +28.5%」**已作廢** ⇒ <b>交付 cell 尚無配… |
| `docs/MTP_AMORTIZATION_2026-09-25.html` | 144 | `28.5` | ✅ | …缺量測）⇒ 「交付口徑的 m 是多少」目前<b>不知道</b>， ⛔ 原寫「只知道端點 +28.5%」—— <b>該端點已作廢</b>（契約 §7）⇒… |
| `docs/MTP_AMORTIZATION_2026-09-25.html` | 127 | `9.82` | ✅ | …/b>：<b>攤薄係數 ≈ 0</b>（本頁 §1）；⛔ <s>交付口徑 MTP off 9.82 → on 12.62 = +28.5%</s> —… |
| `docs/S1_LINE_VERDICT_2026-09-25.md` | 20 | `28.5` | ✅ | …那個 2× 是 S1 的，不是 MTP 的。**（⚠ 原寫「交付口徑 MTP 只值 **+28.5%**」—— **該比值已作廢**，見 `MEASU… |
| `docs/S1_LINE_VERDICT_2026-09-25.md` | 115 | `28.5` | ✅ | …＝ miss 處理） / / MTP / spec 的增益 / ⛔ 原寫「交付口徑 **+28.5%**」**已作廢**（HTTP 舊口徑、無 war… |

### 3.3 歸檔面（237 處）

| 檔案 | 行 | 數字 | 已標記 | 上下文 |
|---|---|---|---|---|
| `.workbuddy/memory/2026-09-17.md` | 2056 | `12.62` | ⛔ 未標記 | …正（預設） / **58.25%** / 180/309 / **21.74** / **12.62** /  - **M4 的離開條件「MTP-on… |
| `.workbuddy/memory/2026-09-17.md` | 2058 | `12.62` | ⛔ 未標記 | ….62** /  - **M4 的離開條件「MTP-on ≥ MTP-off」達成了**：12.62 ≥ 9.82（**+28.5%**）；修前是淨負… |
| `.workbuddy/memory/2026-09-17.md` | 2097 | `12.62` | ⛔ 未標記 | …-/---/---/ / headline **25 t/s**（MTP on） / **12.62** / ×1.98 / **不更樂觀** / /… |
| `.workbuddy/memory/2026-09-17.md` | 2109 | `12.62` | ⛔ 未標記 | …M4**    —— M4 今天**花掉了**，給的是 **×1.285**（9.82→12.62）。 2. **roadmap 自己判過**：`0… |
| `.workbuddy/memory/2026-09-17.md` | 2178 | `12.62` | ⛔ 未標記 | …= 0.76190`（16/21、mean len 3.29）。**與 §14.2 的 12.62 不可直接比**（24 vs 96 token、… |
| `.workbuddy/memory/2026-09-17.md` | 2328 | `12.62` | ✅ | …被抬高**的家族；修好後 accept 降到 58.25%    而 decode 升到 12.62 ⇒ 改成「MTP-on ≥ MTP-off ＋… |
| `.workbuddy/memory/2026-09-17.md` | 2439 | `12.62` | ⛔ 未標記 | …非 MTP **12.95 可持續**（n=124）； **MTP-on（r33 修正後）12.62 vs MTP-off 9.82**（同 bina… |
| `.workbuddy/memory/2026-09-17.md` | 2474 | `12.62` | ⛔ 未標記 | …2／9.32／9.38 ＝ **1.1%**）。 - **⚠ MTP 掉到口徑外。** `12.62 vs 9.82`（今天那條「MTP 由淨負轉淨正… |
| `.workbuddy/memory/2026-09-17.md` | 3032 | `12.62` | ⛔ 未標記 | …EAVY**、swap 2.5–4.6 GiB 的箱子上量的 ⇒ **不可與 10.78／12.62 那個家族並排**。**比值（+38%，MTP-o… |
| `.workbuddy/memory/2026-09-17.md` | 3118 | `12.62` | ⛔ 未標記 | …** ⇒ **「被服務的配置」在現行口徑下沒有數字**。 最接近的是 HTTP 口徑的 `12.62`（14:2x，同 binary A/B），而他們… |
| `.workbuddy/memory/2026-09-17.md` | 3119 | `12.62` | ⛔ 未標記 | …ompt ＋ HEAVY ＋ swap 2.5–4.6 GiB ⇒ **與 `10.78／12.62` 家族不可並排**。  **他們已把我開的處方實… |
| `.workbuddy/memory/2026-09-17.md` | 3247 | `12.62` | ⛔ 未標記 | …10.98 是 **MTP off** 的 decode。最接近的是 HTTP 口徑的 12.62（14:2x）。  ### M1/M2/M3：**… |
| `.workbuddy/memory/2026-09-17.md` | 4050 | `12.62` | ✅ | …高**了這個數字（73.81% → 修後 58.25%，同時 decode 8.62 → 12.62 t/s）。   改成三條：① MTP-on de… |
| `.workbuddy/memory/2026-09-17.md` | 2058 | `28.5` | ⛔ 未標記 | …的離開條件「MTP-on ≥ MTP-off」達成了**：12.62 ≥ 9.82（**+28.5%**）；修前是淨負   （8.62 < 9.82）… |
| `.workbuddy/memory/2026-09-17.md` | 2116 | `28.5` | ⛔ 未標記 | …結構上不可能更快**）。  **15 稍微樂觀的理由**：MTP **由淨負轉淨正**（+28.5%，同一 binary）是今天唯一「真的變快」的量測… |
| `.workbuddy/memory/2026-09-17.md` | 2054 | `9.82` | ⛔ 未標記 | …-/---/ / MTP off（分母） / n/a / 0/0 / 19.24 / **9.82** / / MTP on，`CGC_IDS_LIN… |
| `.workbuddy/memory/2026-09-17.md` | 2058 | `9.82` | ⛔ 未標記 | …- **M4 的離開條件「MTP-on ≥ MTP-off」達成了**：12.62 ≥ 9.82（**+28.5%**）；修前是淨負   （8.62… |
| `.workbuddy/memory/2026-09-17.md` | 2059 | `9.82` | ⛔ 未標記 | …了**：12.62 ≥ 9.82（**+28.5%**）；修前是淨負   （8.62 < 9.82）⇒「那個功能是被一個本身壞掉的量測正確地否決掉的」… |
| `.workbuddy/memory/2026-09-17.md` | 2098 | `9.82` | ⛔ 未標記 | …*不更樂觀** / / M3 的離開條件 **15 t/s**（MTP off） / **9.82** / ×1.53 / **稍微更樂觀** / /… |
| `.workbuddy/memory/2026-09-17.md` | 2109 | `9.82` | ⛔ 未標記 | ….5 記在 M4**    —— M4 今天**花掉了**，給的是 **×1.285**（9.82→12.62）。 2. **roadmap 自己判過… |
| `.workbuddy/memory/2026-09-17.md` | 2330 | `9.82` | ✅ | …14 的 8.87 比的；今天同類（Nail、8 GiB、    MTP off）是 **9.82**，且 r33 改動 routing（hit 57… |
| `.workbuddy/memory/2026-09-17.md` | 2439 | `9.82` | ⛔ 未標記 | …**（n=124）； **MTP-on（r33 修正後）12.62 vs MTP-off 9.82**（同 binary、n=96）。⇒ **prod… |
| `.workbuddy/memory/2026-09-17.md` | 2474 | `9.82` | ⛔ 未標記 | …38 ＝ **1.1%**）。 - **⚠ MTP 掉到口徑外。** `12.62 vs 9.82`（今天那條「MTP 由淨負轉淨正」）是   `mt… |
| `.workbuddy/memory/2026-09-18.md` | 1373 | `12.62` | ⛔ 未標記 | ….5 t/s**；步預算另一條（`gap → 0`）⇒ **12.1 t/s**。 ⇒ `12.62 × 1.5 ≈ 19` ⇒ **到不了 25**… |
| `.workbuddy/memory/2026-09-18.md` | 1754 | `12.62` | ⛔ 未標記 | …實測**：M1/M2/M3 對 **v6** 全 9/9；**decode 8.62 → 12.62 t/s**；**prefill 命中率 57.1… |
| `.workbuddy/memory/2026-09-18.md` | 2135 | `12.62` | ✅ | …/ **×1.06** / / `8e9e830e5` 後（跨啟動） / 9.82 → 12.62 / +28.5%，**不可引用**（純順序效應就… |
| `.workbuddy/memory/2026-09-18.md` | 4385 | `12.62` | ⛔ 未標記 | ….09 = 12.8` vs 實測 **12.4–12.7** ✓（亦與記錄的 HTTP 12.62 互證）。  **需求表**（`mean_len… |
| `.workbuddy/memory/2026-09-18.md` | 4761 | `12.62` | ⛔ 未標記 | ….68, 13.48〕 / 110.44〔109.95, 110.44〕 / 記錄的 **12.62** /  - **12.x 回來了、而且超過**… |
| `.workbuddy/memory/2026-09-18.md` | 4763 | `12.62` | ⛔ 未標記 | …** /  - **12.x 回來了、而且超過**：MTP on 的 **13.68 > 12.62**（+8.4%）。驅動自己印 `bar 250/… |
| `.workbuddy/memory/2026-09-18.md` | 4770 | `12.62` | ⛔ 未標記 | …2.40`（與 19:05 那筆逐位相同）。 - **不宣稱因果**：MTP-on 從 12.62 到 13.68 可能來自環境、也可能來自今天的改… |
| `.workbuddy/memory/2026-09-18.md` | 2135 | `28.5` | ✅ | …6** / / `8e9e830e5` 後（跨啟動） / 9.82 → 12.62 / +28.5%，**不可引用**（純順序效應就有 12.5%）… |
| `.workbuddy/memory/2026-09-18.md` | 2135 | `9.82` | ✅ | …nch 配對」 / **×1.06** / / `8e9e830e5` 後（跨啟動） / 9.82 → 12.62 / +28.5%，**不可引用**… |
| `.workbuddy/memory/2026-09-19.md` | 725 | `12.62` | ⛔ 未標記 | …⇒ **13.68 是「丢掉第一个请求之后的稳态」，不是冷启动值。** portal 的 12.62（配对中位）同类。  ### (b) 冷启动对账：… |
| `.workbuddy/memory/2026-09-19.md` | 752 | `12.62` | ⛔ 未標記 | …bench_matrix.py"` ⇒ 它是 llama-bench**； 而它记的 **12.62 来自 HTTP**（`ROUTING_TRACE… |
| `.workbuddy/memory/2026-09-19.md` | 3577 | `12.62` | ✅ | …nion ≤ 38×mean_len`）。 同時 `current_text` 加註：**12.62 只有 http_duo 的讀數、而它三個臂都報… |
| `.workbuddy/memory/2026-09-19.md` | 3579 | `12.62` | ✅ | …一次生產讀數）。 驗過：`sorted(keys())` 完整、`current` 仍是 12.62、無重複鍵、`build_portal.py --… |
| `.workbuddy/memory/2026-09-20.md` | 643 | `12.62` | ⛔ 未標記 | …/ 249.06，全部 <250**）；decode ≥25 仍 `not-met`（**12.62，需 2.07×**； 按 steady 重算是… |
| `.workbuddy/memory/2026-09-20.md` | 1553 | `12.62` | ⛔ 未標記 | …ix.py (llama-bench, BOTH axes)`，而 `from` 是 **12.62** —— 那是 **HTTP／`llama-sp… |
| `.workbuddy/memory/2026-09-20.md` | 1555 | `12.62` | ⛔ 未標記 | …K_2026-09-18.md:12` 用自己的話寫的）。 而同一份 §5 的結論是 **12.62 既沒複現也沒被證偽**：它與 10.x 的 21… |
| `.workbuddy/memory/2026-09-20.md` | 1557 | `12.62` | ⛔ 未標記 | …2 t/s**）。 ⇒ 改成 **10.78–10.98**，並寫明需要的倍率不是 25/12.62 = **1.98×** 而是 **2.28–2.… |
| `.workbuddy/memory/2026-09-20.md` | 1856 | `12.62` | ⛔ 未標記 | …rom` ＝ 10.78–10.98 ⇒ 25 需要 **2.28–2.32×**（不是 12.62 給的 1.98×）。 **prefill 不是「… |
| `.workbuddy/memory/2026-09-22.md` | 506 | `12.62` | ⛔ 未標記 | …⚠ 但別說成「bug 讓它變快」：本線 09-19 實測修正是**變快**（8.62→12.62），方向相反。  ## §EN-425 「① 12… |
| `.workbuddy/memory/2026-09-22.md` | 521 | `12.62` | ✅ | …-17 記憶說的「被抬高的 regime」（修前 73.81% 同族）；修後同格實測 **12.62 ≈ 12.57**；   ③ 那份文檔自述「**… |
| `.workbuddy/memory/2026-09-22.md` | 589 | `12.62` | ✅ | …省 90 個 CPY ＝ **淨虧**（**MTP off 9.82 vs MTP on 12.62 ＝ +28.5%**，   而 CPY 只值 *… |
| `.workbuddy/memory/2026-09-22.md` | 589 | `28.5` | ✅ | …Y ＝ **淨虧**（**MTP off 9.82 vs MTP on 12.62 ＝ +28.5%**，   而 CPY 只值 **2–4.6%**… |
| `.workbuddy/memory/2026-09-22.md` | 592 | `28.5` | ✅ | …ot＝K 份不同回滾快照）；   ③ `n_max` 3→1 ❌ 大概率淨虧（MTP 值 28.5% ≫ CPY 2–4.6%；且 `n_max 3→… |
| `.workbuddy/memory/2026-09-22.md` | 597 | `28.5` | ✅ | …** 不是 ms，需釐清）。 - **建議**：① 不做 CPY／MTP 方向（先驗被 +28.5% 否決）；② 不做 `node` 命名（09-18… |
| `.workbuddy/memory/2026-09-22.md` | 589 | `9.82` | ✅ | …都走不通**：① 關 MTP 省 90 個 CPY ＝ **淨虧**（**MTP off 9.82 vs MTP on 12.62 ＝ +28.5%*… |
| `.workbuddy/memory/2026-09-23.md` | 408 | `12.62` | ⛔ 未標記 | …退休。 2. 當時**唯一合法的例外＝ MTP/spec**：bench 表達不了 ⇒ `12.62 vs 9.82` 用 HTTP／    `lla… |
| `.workbuddy/memory/2026-09-23.md` | 1726 | `12.62` | ⛔ 未標記 | …MTP**（每 step 3.117 token；MTP off 9.82 vs on 12.62 = **+28.5%**）⇒ 直接歸零。 - 本… |
| `.workbuddy/memory/2026-09-23.md` | 1726 | `28.5` | ⛔ 未標記 | …ep 3.117 token；MTP off 9.82 vs on 12.62 = **+28.5%**）⇒ 直接歸零。 - 本機 `import m… |
| `.workbuddy/memory/2026-09-23.md` | 408 | `9.82` | ⛔ 未標記 | …**唯一合法的例外＝ MTP/spec**：bench 表達不了 ⇒ `12.62 vs 9.82` 用 HTTP／    `llama-specul… |
| `.workbuddy/memory/2026-09-23.md` | 1726 | `9.82` | ⛔ 未標記 | …- 換過去會**失去 MTP**（每 step 3.117 token；MTP off 9.82 vs on 12.62 = **+28.5%**）… |
| `.workbuddy/memory/2026-09-24.md` | 595 | `12.62` | ✅ | …a-speculative-simple` 口徑**（`MEMORY_PERF`：   「12.62 vs 9.82 是 HTTP 口徑」），**與交… |
| `.workbuddy/memory/2026-09-24.md` | 1940 | `12.62` | ⛔ 未標記 | …MTP 的 cell」應有的值**：既有對照 **MTP off 9.82 vs on 12.62（×1.28）** （`MEMORY_PERF.m… |
| `.workbuddy/memory/2026-09-24.md` | 2006 | `12.62` | ⛔ 未標記 | …anchor **12.57**    ⇒ 既有「MTP off 9.82 vs on 12.62 ＝ ×1.28」（`MEMORY_PERF.md… |
| `.workbuddy/memory/2026-09-24.md` | 70 | `28.5` | ⛔ 未標記 | …「誰全面更好」都不成立。**  ### ⑥ 功能完整度：我們明顯領先  MTP（我們有，+28.5%；對方無）／i-quant IQ2_S/IQ3_S… |
| `.workbuddy/memory/2026-09-24.md` | 71 | `28.5` | ⛔ 未標記 | …生產 server（對方只有 bench 腳本）。 ⇒ 換過去**直接丟掉 MTP 的 +28.5%**。 對方唯一領先：**oracle ceili… |
| `.workbuddy/memory/2026-09-24.md` | 83 | `28.5` | ⛔ 未標記 | …LX**（miss 魯棒性 6.6×、 命中率 +6.7pp、位元組 −34%、MTP +28.5%、16 GB 可行）；引擎純算力平手；**絕對 t… |
| `.workbuddy/memory/2026-09-24.md` | 554 | `28.5` | ⛔ 未標記 | …~25%）**。它的 5.3×/2.0× 是相對**自回歸**，   我們已有 MTP（+28.5%）⇒ 基準不同，不能拿來比。 - **★ 阻斷**… |
| `.workbuddy/memory/2026-09-24.md` | 595 | `9.82` | ✅ | …tive-simple` 口徑**（`MEMORY_PERF`：   「12.62 vs 9.82 是 HTTP 口徑」），**與交付 anchor… |
| `.workbuddy/memory/2026-09-24.md` | 1940 | `9.82` | ⛔ 未標記 | …4 不是退化，是「關掉 MTP 的 cell」應有的值**：既有對照 **MTP off 9.82 vs on 12.62（×1.28）** （`ME… |
| `.workbuddy/memory/2026-09-24.md` | 2006 | `9.82` | ⛔ 未標記 | ….57**，已貼近交付 anchor **12.57**    ⇒ 既有「MTP off 9.82 vs on 12.62 ＝ ×1.28」（`MEM… |
| `.workbuddy/memory/2026-09-25.md` | 1556 | `12.62` | ✅ | …code 比目前最好還要好**（目前最好：交付錨點 **12.57**、MTP-on **12.62**） / / **② 攻關過線** / **pr… |
| `.workbuddy/memory/2026-09-25.md` | 1571 | `12.62` | ✅ | …a（零成本＋逐位正確）／S1 數值身分（576/576）／MTP-spec（**已在生產：12.62**）／swap 結構修復（執行線主張未複核） /… |
| `.workbuddy/memory/2026-09-25.md` | 1628 | `12.62` | ✅ | …-1353 ★★★ 作廢數字登記表（operator 12:4x 下令）：`9.82`/`12.62`/`+28.5%` 全部作廢  ### (a)… |
| `.workbuddy/memory/2026-09-25.md` | 1631 | `12.62` | ✅ | …# (a) operator 的指正 > 「交付口徑 MTP off 9.82 → on 12.62，**我們現在 MTP off 就已經 11+ 了… |
| `.workbuddy/memory/2026-09-25.md` | 1635 | `12.62` | ✅ | ….md:146`「[2026-09-24 20:56]『MTP ×1.28（9.82 → 12.62）』在本線交付口徑下不可再引用」）， **但索引與… |
| `.workbuddy/memory/2026-09-25.md` | 1642 | `12.62` | ✅ | …fill` **11.793**／`mw_ctrl` **12.195** / / **`12.62`** / 同口徑 / ①同上 ②`MTP_ABB… |
| `.workbuddy/memory/2026-09-25.md` | 1643 | `12.62` | ✅ | …** / **無** ⇒ 交付 cell 缺量測 / / **`+28.5%`** / ＝12.62÷9.82 / 分子分母皆廢 ⇒ 比值無效；卻被用… |
| `.workbuddy/memory/2026-09-25.md` | 1645 | `12.62` | ✅ | …（交付錨點）不作廢** ⇒ **「目前最好」一律以 12.57 為準，不含 MTP-on 12.62**。  ### (c) 已改的決策面（6 檔）… |
| `.workbuddy/memory/2026-09-25.md` | 1293 | `28.5` | ⛔ 未標記 | …2.45 = +8.3%** ⇒ **使用者的直覺正確：MTP 只值個位數%（交付口徑 +28.5%），它撐不出 2×。** ⇒ 那個 2× 是 **… |
| `.workbuddy/memory/2026-09-25.md` | 1617 | `28.5` | ⛔ 未標記 | …TP on 加速** / 攤薄係數 m 與 acc / **攤薄 ≈ 0**；交付端點 +28.5% / / **兩者** / 需 S 與 M 同時成… |
| `.workbuddy/memory/2026-09-25.md` | 1643 | `28.5` | ✅ | …×0.695（−30%）** / **無** ⇒ 交付 cell 缺量測 / / **`+28.5%`** / ＝12.62÷9.82 / 分子分母皆… |
| `.workbuddy/memory/2026-09-25.md` | 1628 | `9.82` | ✅ | …## §EN-1353 ★★★ 作廢數字登記表（operator 12:4x 下令）：`9.82`/`12.62`/`+28.5%` 全部作廢  #… |
| `.workbuddy/memory/2026-09-25.md` | 1631 | `9.82` | ✅ | …` 全部作廢  ### (a) operator 的指正 > 「交付口徑 MTP off 9.82 → on 12.62，**我們現在 MTP off… |
| `.workbuddy/memory/2026-09-25.md` | 1635 | `9.82` | ✅ | …RY_PERF.md:146`「[2026-09-24 20:56]『MTP ×1.28（9.82 → 12.62）』在本線交付口徑下不可再引用」），… |
| `.workbuddy/memory/2026-09-25.md` | 1641 | `9.82` | ✅ | …/ 出處／口徑 / 作廢理由 / 替代 / /---/---/---/---/ / **`9.82`** / 09-17 **HTTP／`llama-… |
| `.workbuddy/memory/2026-09-25.md` | 1643 | `9.82` | ✅ | …*無** ⇒ 交付 cell 缺量測 / / **`+28.5%`** / ＝12.62÷9.82 / 分子分母皆廢 ⇒ 比值無效；卻被用來支撐「MT… |
| `.workbuddy/memory/2026-09-25.md` | 1655 | `9.82` | ✅ | …**8/8**） - 來源＝契約 §7 表格（自動解析 token，含 `19.82 ≠ 9.82` 的邊界保護）。 - **分層**：**決策面**… |
| `agent_harness/memory/2026-09-17.md` | 2062 | `12.62` | ⛔ 未標記 | …正（預設） / **58.25%** / 180/309 / **21.74** / **12.62** /  - **M4 的離開條件「MTP-on… |
| `agent_harness/memory/2026-09-17.md` | 2064 | `12.62` | ⛔ 未標記 | ….62** /  - **M4 的離開條件「MTP-on ≥ MTP-off」達成了**：12.62 ≥ 9.82（**+28.5%**）；修前是淨負… |
| `agent_harness/memory/2026-09-17.md` | 2103 | `12.62` | ⛔ 未標記 | …-/---/---/ / headline **25 t/s**（MTP on） / **12.62** / ×1.98 / **不更樂觀** / /… |
| `agent_harness/memory/2026-09-17.md` | 2115 | `12.62` | ⛔ 未標記 | …M4**    —— M4 今天**花掉了**，給的是 **×1.285**（9.82→12.62）。 2. **roadmap 自己判過**：`0… |
| `agent_harness/memory/2026-09-17.md` | 2184 | `12.62` | ⛔ 未標記 | …= 0.76190`（16/21、mean len 3.29）。**與 §14.2 的 12.62 不可直接比**（24 vs 96 token、… |
| `agent_harness/memory/2026-09-17.md` | 2334 | `12.62` | ✅ | …被抬高**的家族；修好後 accept 降到 58.25%    而 decode 升到 12.62 ⇒ 改成「MTP-on ≥ MTP-off ＋… |
| `agent_harness/memory/2026-09-17.md` | 2445 | `12.62` | ⛔ 未標記 | …非 MTP **12.95 可持續**（n=124）； **MTP-on（r33 修正後）12.62 vs MTP-off 9.82**（同 bina… |
| `agent_harness/memory/2026-09-17.md` | 2480 | `12.62` | ⛔ 未標記 | …2／9.32／9.38 ＝ **1.1%**）。 - **⚠ MTP 掉到口徑外。** `12.62 vs 9.82`（今天那條「MTP 由淨負轉淨正… |
| `agent_harness/memory/2026-09-17.md` | 3038 | `12.62` | ⛔ 未標記 | …EAVY**、swap 2.5–4.6 GiB 的箱子上量的 ⇒ **不可與 10.78／12.62 那個家族並排**。**比值（+38%，MTP-o… |
| `agent_harness/memory/2026-09-17.md` | 3124 | `12.62` | ⛔ 未標記 | …** ⇒ **「被服務的配置」在現行口徑下沒有數字**。 最接近的是 HTTP 口徑的 `12.62`（14:2x，同 binary A/B），而他們… |
| `agent_harness/memory/2026-09-17.md` | 3125 | `12.62` | ⛔ 未標記 | …ompt ＋ HEAVY ＋ swap 2.5–4.6 GiB ⇒ **與 `10.78／12.62` 家族不可並排**。  **他們已把我開的處方實… |
| `agent_harness/memory/2026-09-17.md` | 3253 | `12.62` | ⛔ 未標記 | …10.98 是 **MTP off** 的 decode。最接近的是 HTTP 口徑的 12.62（14:2x）。  ### M1/M2/M3：**… |
| `agent_harness/memory/2026-09-17.md` | 4056 | `12.62` | ✅ | …高**了這個數字（73.81% → 修後 58.25%，同時 decode 8.62 → 12.62 t/s）。   改成三條：① MTP-on de… |
| `agent_harness/memory/2026-09-17.md` | 2064 | `28.5` | ⛔ 未標記 | …的離開條件「MTP-on ≥ MTP-off」達成了**：12.62 ≥ 9.82（**+28.5%**）；修前是淨負   （8.62 < 9.82）… |
| `agent_harness/memory/2026-09-17.md` | 2122 | `28.5` | ⛔ 未標記 | …結構上不可能更快**）。  **15 稍微樂觀的理由**：MTP **由淨負轉淨正**（+28.5%，同一 binary）是今天唯一「真的變快」的量測… |
| `agent_harness/memory/2026-09-17.md` | 2060 | `9.82` | ⛔ 未標記 | …-/---/ / MTP off（分母） / n/a / 0/0 / 19.24 / **9.82** / / MTP on，`CGC_IDS_LIN… |
| `agent_harness/memory/2026-09-17.md` | 2064 | `9.82` | ⛔ 未標記 | …- **M4 的離開條件「MTP-on ≥ MTP-off」達成了**：12.62 ≥ 9.82（**+28.5%**）；修前是淨負   （8.62… |
| `agent_harness/memory/2026-09-17.md` | 2065 | `9.82` | ⛔ 未標記 | …了**：12.62 ≥ 9.82（**+28.5%**）；修前是淨負   （8.62 < 9.82）⇒「那個功能是被一個本身壞掉的量測正確地否決掉的」… |
| `agent_harness/memory/2026-09-17.md` | 2104 | `9.82` | ⛔ 未標記 | …*不更樂觀** / / M3 的離開條件 **15 t/s**（MTP off） / **9.82** / ×1.53 / **稍微更樂觀** / /… |
| `agent_harness/memory/2026-09-17.md` | 2115 | `9.82` | ⛔ 未標記 | ….5 記在 M4**    —— M4 今天**花掉了**，給的是 **×1.285**（9.82→12.62）。 2. **roadmap 自己判過… |
| `agent_harness/memory/2026-09-17.md` | 2336 | `9.82` | ✅ | …14 的 8.87 比的；今天同類（Nail、8 GiB、    MTP off）是 **9.82**，且 r33 改動 routing（hit 57… |
| `agent_harness/memory/2026-09-17.md` | 2445 | `9.82` | ⛔ 未標記 | …**（n=124）； **MTP-on（r33 修正後）12.62 vs MTP-off 9.82**（同 binary、n=96）。⇒ **prod… |
| `agent_harness/memory/2026-09-17.md` | 2480 | `9.82` | ⛔ 未標記 | …38 ＝ **1.1%**）。 - **⚠ MTP 掉到口徑外。** `12.62 vs 9.82`（今天那條「MTP 由淨負轉淨正」）是   `mt… |
| `agent_harness/memory/2026-09-18.md` | 1379 | `12.62` | ⛔ 未標記 | ….5 t/s**；步預算另一條（`gap → 0`）⇒ **12.1 t/s**。 ⇒ `12.62 × 1.5 ≈ 19` ⇒ **到不了 25**… |
| `agent_harness/memory/2026-09-18.md` | 1760 | `12.62` | ⛔ 未標記 | …實測**：M1/M2/M3 對 **v6** 全 9/9；**decode 8.62 → 12.62 t/s**；**prefill 命中率 57.1… |
| `agent_harness/memory/2026-09-18.md` | 2141 | `12.62` | ✅ | …/ **×1.06** / / `8e9e830e5` 後（跨啟動） / 9.82 → 12.62 / +28.5%，**不可引用**（純順序效應就… |
| `agent_harness/memory/2026-09-18.md` | 4391 | `12.62` | ⛔ 未標記 | ….09 = 12.8` vs 實測 **12.4–12.7** ✓（亦與記錄的 HTTP 12.62 互證）。  **需求表**（`mean_len… |
| `agent_harness/memory/2026-09-18.md` | 4767 | `12.62` | ⛔ 未標記 | ….68, 13.48〕 / 110.44〔109.95, 110.44〕 / 記錄的 **12.62** /  - **12.x 回來了、而且超過**… |
| `agent_harness/memory/2026-09-18.md` | 4769 | `12.62` | ⛔ 未標記 | …** /  - **12.x 回來了、而且超過**：MTP on 的 **13.68 > 12.62**（+8.4%）。驅動自己印 `bar 250/… |
| `agent_harness/memory/2026-09-18.md` | 4776 | `12.62` | ⛔ 未標記 | …2.40`（與 19:05 那筆逐位相同）。 - **不宣稱因果**：MTP-on 從 12.62 到 13.68 可能來自環境、也可能來自今天的改… |
| `agent_harness/memory/2026-09-18.md` | 2141 | `28.5` | ✅ | …6** / / `8e9e830e5` 後（跨啟動） / 9.82 → 12.62 / +28.5%，**不可引用**（純順序效應就有 12.5%）… |
| `agent_harness/memory/2026-09-18.md` | 2141 | `9.82` | ✅ | …nch 配對」 / **×1.06** / / `8e9e830e5` 後（跨啟動） / 9.82 → 12.62 / +28.5%，**不可引用**… |
| `agent_harness/memory/2026-09-19.md` | 731 | `12.62` | ⛔ 未標記 | …⇒ **13.68 是「丢掉第一个请求之后的稳态」，不是冷启动值。** portal 的 12.62（配对中位）同类。  ### (b) 冷启动对账：… |
| `agent_harness/memory/2026-09-19.md` | 758 | `12.62` | ⛔ 未標記 | …bench_matrix.py"` ⇒ 它是 llama-bench**； 而它记的 **12.62 来自 HTTP**（`ROUTING_TRACE… |
| `agent_harness/memory/2026-09-19.md` | 3583 | `12.62` | ✅ | …nion ≤ 38×mean_len`）。 同時 `current_text` 加註：**12.62 只有 http_duo 的讀數、而它三個臂都報… |
| `agent_harness/memory/2026-09-19.md` | 3585 | `12.62` | ✅ | …一次生產讀數）。 驗過：`sorted(keys())` 完整、`current` 仍是 12.62、無重複鍵、`build_portal.py --… |
| `agent_harness/memory/2026-09-20.md` | 649 | `12.62` | ⛔ 未標記 | …/ 249.06，全部 <250**）；decode ≥25 仍 `not-met`（**12.62，需 2.07×**； 按 steady 重算是… |
| `agent_harness/memory/2026-09-20.md` | 1559 | `12.62` | ⛔ 未標記 | …ix.py (llama-bench, BOTH axes)`，而 `from` 是 **12.62** —— 那是 **HTTP／`llama-sp… |
| `agent_harness/memory/2026-09-20.md` | 1561 | `12.62` | ⛔ 未標記 | …K_2026-09-18.md:12` 用自己的話寫的）。 而同一份 §5 的結論是 **12.62 既沒複現也沒被證偽**：它與 10.x 的 21… |
| `agent_harness/memory/2026-09-20.md` | 1563 | `12.62` | ⛔ 未標記 | …2 t/s**）。 ⇒ 改成 **10.78–10.98**，並寫明需要的倍率不是 25/12.62 = **1.98×** 而是 **2.28–2.… |
| `agent_harness/memory/2026-09-20.md` | 1862 | `12.62` | ⛔ 未標記 | …rom` ＝ 10.78–10.98 ⇒ 25 需要 **2.28–2.32×**（不是 12.62 給的 1.98×）。 **prefill 不是「… |
| `agent_harness/memory/MEMORY_PERF.md` | 147 | `12.62` | ✅ | ….62 t/s（accept 73.81%）｜   MTP on nb-aware 修正 12.62 t/s（accept 58.25%）** ⇒ *… |
| `agent_harness/memory/MEMORY_PERF.md` | 154 | `12.62` | ✅ | …的是 `MTP-on ≥ MTP-off`   （6.74 < 8.87）**」⇒ 今天 12.62 ≥ 9.82 ⇒ **M4 的功能缺口關閉**。… |
| `agent_harness/memory/MEMORY_PERF.md` | 275 | `12.62` | ⛔ 未標記 | …三次 **1.1%**）。   **⚠ 未決：MTP 不在這個口徑裡** —— 歷史的 `12.62 vs 9.82` 是 HTTP／`llama-s… |
| `agent_harness/memory/MEMORY_PERF.md` | 369 | `12.62` | ⛔ 未標記 | …步 token / **1.0**（MTP off） / MTP-on **9.82 → 12.62**（×1.28） / ×1.6–2.0 / ac… |
| `agent_harness/memory/MEMORY_PERF.md` | 376 | `12.62` | ✅ | …5.1` ⇒ **剛好擦線**。而： - ⚠️ **兩個乘數能不能相乘還沒量過** —— 12.62 與 16.82 是**不同儀器**量的。本檔早已… |
| `agent_harness/memory/MEMORY_PERF.md` | 387 | `12.62` | ⛔ 未標記 | …兩路互相印證 0.4%） / / MTP（每步 token） / ×1.28（9.82→12.62） / **×1.06（同一 launch 配對）… |
| `agent_harness/memory/MEMORY_PERF.md` | 147 | `28.5` | ✅ | …/s（accept 58.25%）** ⇒ **MTP-on ≥ MTP-off 達成（+28.5%）**，   修前是淨負。★ **accept 下… |
| `agent_harness/memory/MEMORY_PERF.md` | 93 | `9.82` | ✅ | …14 的 build，而今天同類量測（Nail、8 GiB、MTP off）   是 **9.82**（`ROUTING_TRACE` §14.2），… |
| `agent_harness/memory/MEMORY_PERF.md` | 146 | `9.82` | ✅ | …l 8 GiB，`n_predict=96`，每臂 3 請求）：   **MTP off 9.82 t/s｜MTP on 修前（`CGC_IDS_LI… |
| `agent_harness/memory/MEMORY_PERF.md` | 154 | `9.82` | ✅ | …on ≥ MTP-off`   （6.74 < 8.87）**」⇒ 今天 12.62 ≥ 9.82 ⇒ **M4 的功能缺口關閉**。仍未處理：`pl… |
| `agent_harness/memory/MEMORY_PERF.md` | 275 | `9.82` | ⛔ 未標記 | …**）。   **⚠ 未決：MTP 不在這個口徑裡** —— 歷史的 `12.62 vs 9.82` 是 HTTP／`llama-speculativ… |
| `agent_harness/memory/MEMORY_PERF.md` | 369 | `9.82` | ⛔ 未標記 | …倒 / / 每步 token / **1.0**（MTP off） / MTP-on **9.82 → 12.62**（×1.28） / ×1.6–2… |
| `agent_harness/memory/MEMORY_PERF.md` | 375 | `9.82` | ✅ | …** ⇒ **那個數字幾乎全部住在「每步 token 數」那一項**。  **算術**：`9.82 × 1.28（MTP，已量）× 2.0（去序列化，… |
| `agent_harness/memory/MEMORY_PERF.md` | 387 | `9.82` | ⛔ 未標記 | …_sum` 兩路互相印證 0.4%） / / MTP（每步 token） / ×1.28（9.82→12.62） / **×1.06（同一 launc… |
| `agent_harness/memory/MEMORY_S1.md` | 485 | `12.62` | ⛔ 未標記 | …mit 的實測）：**M1/M2/M3 對 v6 全 9/9；decode 8.62 → 12.62 t/s； prefill 命中率 57.1% →… |
| `docs/AGENT_HARNESS_PORTAL.html` | 132 | `12.62` | ✅ | …≈ 12.0–12.9（8 GiB pool、prod25 profile）；配對中位 12.62。需要 2.07×。⚠ 2026-09-19：這個… |
| `docs/AGENT_HARNESS_PORTAL.html` | 132 | `12.62` | ✅ | …yle="margin-top:5px"><b>差距</b>：2.07×（要 25、現在 12.62）。★ 2026-09-19 更正：「已判掉的三條… |
| `docs/BEST_SHAPE_IQ3XXS_M4_2026-09-22.md` | 137 | `12.62` | ⛔ 未標記 | …/ / 关 MTP 省 CPY / k 步长 / MTP off 9.82 vs on 12.62 ⇒ **−28.5%，别动** / `MEMOR… |
| `docs/BEST_SHAPE_IQ3XXS_M4_2026-09-22.md` | 137 | `28.5` | ⛔ 未標記 | …省 CPY / k 步长 / MTP off 9.82 vs on 12.62 ⇒ **−28.5%，别动** / `MEMORY_PERF.md`… |
| `docs/BEST_SHAPE_IQ3XXS_M4_2026-09-22.md` | 137 | `9.82` | ⛔ 未標記 | …etal:11901` / / 关 MTP 省 CPY / k 步长 / MTP off 9.82 vs on 12.62 ⇒ **−28.5%，别动… |
| `docs/BIGGER_FISH_2026-09-22.md` | 69 | `12.62` | ✅ | …PY / ❌ **淨虧** / **MTP off 9.82 t/s vs MTP on 12.62 t/s ＝ +28.5%**（§EN 09-17… |
| `docs/BIGGER_FISH_2026-09-22.md` | 10 | `28.5` | ✅ | …**真魚，但先驗條件否決** / 120 個 CPY 來自 MTP；而 MTP 值 **+28.5%**，CPY 只值 2–4.6% ⇒ 淨虧 / /… |
| `docs/BIGGER_FISH_2026-09-22.md` | 48 | `28.5` | ⛔ 未標記 | …*  ---  ## 2. `cache` 的 CPY：來源是 MTP，而 MTP 值 +28.5%  ### 2.1 因果鏈（本次查證，全部有源碼）… |
| `docs/BIGGER_FISH_2026-09-22.md` | 69 | `28.5` | ✅ | …/ **MTP off 9.82 t/s vs MTP on 12.62 t/s ＝ +28.5%**（§EN 09-17）。CPY 只值 2–4.… |
| `docs/BIGGER_FISH_2026-09-22.md` | 71 | `28.5` | ✅ | …mic-k oracle 1.035×）；反方向減小會損失 accept，而 MTP 值 28.5% ≫ CPY 2–4.6% /  CPY 的時間價… |
| `docs/BIGGER_FISH_2026-09-22.md` | 77 | `28.5` | ⛔ 未標記 | …**4.6%**  ⇒ **2–4.6%，在 3% 門檻附近或略超，但遠小於 MTP 的 28.5%。**  ---  ## 3. 比兩條魚都大的：非… |
| `docs/BIGGER_FISH_2026-09-22.md` | 102 | `28.5` | ⛔ 未標記 | …步（按性價比）  1. **不做** CPY／MTP 方向 —— 先驗已被 MTP 的 +28.5% 否決 2. **不做** `node` 命名 —… |
| `docs/BIGGER_FISH_2026-09-22.md` | 69 | `9.82` | ✅ | …-/ / 關 MTP 省 90 個 CPY / ❌ **淨虧** / **MTP off 9.82 t/s vs MTP on 12.62 t/s ＝… |
| `docs/BITIDENTITY_AND_0907_KNOWHOW_2026-09-22.md` | 146 | `12.62` | ⛔ 未標記 | …re 修正（預設） / 58.25% / 180/309 / **21.74** / **12.62** /  同一份记忆明确写着：  > 「**修前… |
| `docs/BITIDENTITY_AND_0907_KNOWHOW_2026-09-22.md` | 153 | `12.62` | ⛔ 未標記 | …regime**；25.17 是在**错误路由**下量出来的。 修好之后的同一格 = **12.62**，而我们的封版 = **12.57**。**两… |
| `docs/BITIDENTITY_AND_0907_KNOWHOW_2026-09-22.md` | 144 | `9.82` | ⛔ 未標記 | …-/---/ / MTP off（分母） / n/a / 0/0 / 19.24 / **9.82** / / MTP on，`CGC_IDS_LIN… |
| `docs/CAPABILITY_VS_FLASHMLX_2026-09-24.md` | 118 | `12.62` | ⛔ 未標記 | …tive / ✅ 每 step 3.117 token，**off 9.82 vs on 12.62 = +28.5%** / ❌ 不支援 / / i… |
| `docs/CAPABILITY_VS_FLASHMLX_2026-09-24.md` | 118 | `28.5` | ⛔ 未標記 | …每 step 3.117 token，**off 9.82 vs on 12.62 = +28.5%** / ❌ 不支援 / / i-quant（IQ… |
| `docs/CAPABILITY_VS_FLASHMLX_2026-09-24.md` | 123 | `28.5` | ⛔ 未標記 | …_bank_oracle_hits.py` /  ⇒ 換過去會**直接丟掉 MTP 的 +28.5%**（§EN-474）。  ---  ## 7.… |
| `docs/CAPABILITY_VS_FLASHMLX_2026-09-24.md` | 146 | `28.5` | ⛔ 未標記 | …34%（vs 其主力 4-bit） / / 功能完整度（MTP） / **我們** / +28.5% / / 約束適配（16 GB 可行） / **我… |
| `docs/CAPABILITY_VS_FLASHMLX_2026-09-24.md` | 118 | `9.82` | ⛔ 未標記 | …P / speculative / ✅ 每 step 3.117 token，**off 9.82 vs on 12.62 = +28.5%** /… |
| `docs/CROSS_LINE_STATUS_AND_TARGET_MATH_2026-09-20.md` | 115 | `12.62` | ⛔ 未標記 | …rgets.json` 現在的 `decode-25` 混用了兩個口徑：`from` 寫 12.62（HTTP 口徑）、`gates` 的步時 139… |
| `docs/FLASHMLX_EVALUATION_2026-09-23.md` | 119 | `12.62` | ✅ | …3.117 token（接受率 77.9%），MTP off 實測 9.82 vs on 12.62 = **+28.5%**。換過去這 28.5%… |
| `docs/FLASHMLX_EVALUATION_2026-09-23.md` | 119 | `28.5` | ✅ | …（接受率 77.9%），MTP off 實測 9.82 vs on 12.62 = **+28.5%**。換過去這 28.5% 直接歸零。 / / 生… |
| `docs/FLASHMLX_EVALUATION_2026-09-23.md` | 119 | `28.5` | ✅ | …TP off 實測 9.82 vs on 12.62 = **+28.5%**。換過去這 28.5% 直接歸零。 / / 生產配置 / prod25… |
| `docs/FLASHMLX_EVALUATION_2026-09-23.md` | 157 | `28.5` | ⛔ 未標記 | …44%/pp** vs 對方 **2.90%/pp** / / 會失去 MTP / **+28.5%** 直接歸零 / / 實測數字不可比 / 對方… |
| `docs/FLASHMLX_EVALUATION_2026-09-23.md` | 119 | `9.82` | ✅ | …我們每 step 產 3.117 token（接受率 77.9%），MTP off 實測 9.82 vs on 12.62 = **+28.5%**。… |
| `docs/FLEET_PORTAL_DELIVERY_20260918_1330.html` | 715 | `12.62` | ⛔ 未標記 | …← 人工提供     - decode 25 t/s 未達成（現值 12.62 t/s，2.07×）          ← 人工… |
| `docs/G7_DELIVERY_ARITHMETIC_2026-09-20.md` | 19 | `12.62` | ⛔ 未標記 | …也在下面被更正。  ## 1. G7 的 `from` 是**退役儀器**上的數  - `12.62` 屬 **HTTP／`llama-specula… |
| `docs/G7_DELIVERY_ARITHMETIC_2026-09-20.md` | 22 | `12.62` | ⛔ 未標記 | …** - 落地儀器上的記錄值是 **10.78–10.98**。⇒ 需要的倍率不是 25/12.62 = **1.98×**，而是   **2.28–… |
| `docs/G7_DELIVERY_ARITHMETIC_2026-09-20.md` | 24 | `12.62` | ⛔ 未標記 | …/12.62 = **1.98×**，而是   **2.28–2.32×**。 - 而 `12.62` 本身**既沒有複現也沒有被證偽**：它與 10… |
| `docs/G7_DELIVERY_ARITHMETIC_2026-09-20.md` | 62 | `12.62` | ⛔ 未標記 | …配對**算出來的。它是一個**結構正確的估計**，    不是那台儀器的讀數。 2. ❌「12.62 → 13.6 是進步」—— 兩者不同儀器、不同配… |
| `docs/HTTP_VS_BENCH_CALIBER_2026-09-18.md` | 97 | `12.62` | ✅ | …*，兩臂一致，比穩態低約 32%。  **不可引用** - 任何把今天的 11.75 當「12.62 重現」的說法——**12.62 是 MTP on… |
| `docs/HTTP_VS_BENCH_CALIBER_2026-09-18.md` | 97 | `12.62` | ✅ | …。  **不可引用** - 任何把今天的 11.75 當「12.62 重現」的說法——**12.62 是 MTP on**，本輪兩側**都 MTP o… |
| `docs/HTTP_VS_BENCH_CALIBER_2026-09-18.md` | 159 | `12.62` | ⛔ 未標記 | …證偽——E5 改用前景就沒再發生。  ## 6. 我的判斷（是推論，不是記錄裡的話）  `12.62 vs 10.9` 那個 +15.8% 的差，**… |
| `docs/KNOB_INVENTORY_VERIFIED_2026-09-22.md` | 117 | `12.62` | ⛔ 未標記 | …-mtp` / 不是 autotuner 的軸；且 MTP off 9.82 vs on 12.62 ⇒ 別動 / / `CGC_DRAFT_DECO… |
| `docs/KNOB_INVENTORY_VERIFIED_2026-09-22.md` | 117 | `9.82` | ⛔ 未標記 | …-type draft-mtp` / 不是 autotuner 的軸；且 MTP off 9.82 vs on 12.62 ⇒ 別動 / / `CGC… |
| `docs/M1_M4_M5_DEV_BRIEF_20260917_1525.html` | 80 | `12.62` | ⛔ 未標記 | …">功能缺口已關</div>     <div class="small">MTP-on 12.62 ≥ MTP-off 9.82；剩 <code>p… |
| `docs/M1_M4_M5_DEV_BRIEF_20260917_1525.html` | 141 | `12.62` | ⛔ 未標記 | …strong> 58.25%，而 decode <strong>上升到</strong> 12.62。   ⇒ <strong>任何只看 accept… |
| `docs/M1_M4_M5_DEV_BRIEF_20260917_1525.html` | 188 | `12.62` | ⛔ 未標記 | …lass="num">21.74</td><td class="num"><strong>12.62</strong></td></tr> </tab… |
| `docs/M1_M4_M5_DEV_BRIEF_20260917_1525.html` | 80 | `9.82` | ⛔ 未標記 | …<div class="small">MTP-on 12.62 ≥ MTP-off 9.82；剩 <code>plain_match</code… |
| `docs/M1_M4_M5_DEV_BRIEF_20260917_1525.html` | 128 | `9.82` | ⛔ 未標記 | …/strong>；今天同類量測（Nail、8 GiB、MTP off）是 <strong>9.82</strong>，而且 r33 修正改動了 rou… |
| `docs/M1_M4_M5_DEV_BRIEF_20260917_1525.html` | 186 | `9.82` | ⛔ 未標記 | …td><td class="num">19.24</td><td class="num">9.82</td></tr> <tr><td>MTP on，… |
| `docs/M1_WORKITEM2_PHASE_SPLIT_STATUS_2026-09-17.md` | 106 | `12.62` | ⛔ 未標記 | …iB. They are NOT comparable with the 10.78 / 12.62 t/s family measured earl… |
| `docs/M1_WORKITEM4_CANON_ORDER_STATUS_2026-09-17.md` | 131 | `12.62` | ✅ | …高**了這個數字：73.81% → 修後 58.25%，同時 decode 8.62 → 12.62 t/s）。改成三條：  1. **MTP-on… |
| `docs/M1_WORKITEM4_CANON_ORDER_STATUS_2026-09-17.md` | 141 | `12.62` | ⛔ 未標記 | …31%）✓**。與 §14.2 在 14:12 binary 上量到的 58.25% / 12.62 相同到 accept 的第 4 位有效數字，… |
| `docs/M1_WORKITEM4_CANON_ORDER_STATUS_2026-09-17.md` | 121 | `9.82` | ⛔ 未標記 | …iB `6.36` vs 8 GiB `8.87` 是當時的 build；今天同類量測是 9.82）。改成可量測形式：  1. **同 build、同… |
| `docs/M4_M5_DELIVERY_M3_M5_CLOSURE_20260917_2100.html` | 129 | `12.62` | ✅ | …了這個數字（73.81% → 修後 58.25%，   同時 decode 8.62 → 12.62 t/s）。<b>任何修前記錄的 accept（含… |
| `docs/M4_M5_DELIVERY_M3_M5_CLOSURE_20260917_2100.html` | 406 | `12.62` | ⛔ 未標記 | …s（12–15 token prompt、熱狀態未控制）上，       與 10.78／12.62 那個家族不可比。<b>可引用的是比值（+31%）… |
| `docs/MTP3_25TPS_ARITHMETIC_2026-09-18.md` | 56 | `12.62` | ⛔ 未標記 | …72.4 ms/step ⇒ 12.41–12.71 t/s**，與記錄的 HTTP **12.62 t/s** 相符。 2. MTP-off 的一步… |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 1 | `12.62` | ⛔ 未標記 | …# 12.62 的重驗證：ABBA 四臂，llama-bench… |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 8 | `12.62` | ⛔ 未標記 | …used` 12178／13312 MB  ## §0 為什麼要重驗  使用者的問題：「12.62（MTP on、nb-aware 修正後）重驗證一… |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 12 | `12.62` | ⛔ 未標記 | …PERF.md:223-240`）：decode 速度有**四個互不相容的定義**， 而 12.62 屬 **HTTP／`llama-speculat… |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 64 | `12.62` | ⛔ 未標記 | …**「連跑造成的自我加熱」＋「背景負載」的合計**，不再是「純環境」。  ## §5 對「12.62 vs 10.x」的分解  / 成分 / 量級 /… |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 69 | `12.62` | ✅ | …1 的 2×2） / / **MTP 本身** / HTTP 口徑 **+28.5%**（12.62 vs 9.82）；llama-bench 口徑下… |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 73 | `12.62` | ⛔ 未標記 | …負載）～ **65%**（四臂極差 5.81–9.62） / ★ 本輪實測 /  ⇒ **12.62 既沒有複現、也沒有被證偽**：它與 10.x 的… |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 69 | `28.5` | ✅ | …FOUND` §1 的 2×2） / / **MTP 本身** / HTTP 口徑 **+28.5%**（12.62 vs 9.82）；llama-b… |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 68 | `9.82` | ✅ | …lama-bench `--spec-type draft-mtp` / MTP off 9.82 vs 10.62 ⇒ **約 8%** / 有記錄… |
| `docs/MTP_ABBA_RECHECK_2026-09-18.md` | 69 | `9.82` | ✅ | …/ / **MTP 本身** / HTTP 口徑 **+28.5%**（12.62 vs 9.82）；llama-bench 口徑下**只量到負載增加… |
| `docs/MTP_BENCH_PARITY_20260918_1945.html` | 257 | `12.62` | ⛔ 未標記 | …um">110.44〔109.95, 110.44〕</td>   <td>記錄的 <b>12.62</b></td> </tr> </table>… |
| `docs/MTP_BENCH_PARITY_20260918_1945.html` | 268 | `12.62` | ⛔ 未標記 | …><b>12.x 回來了，而且超過</b>：MTP on 的 <b>13.68 &gt; 12.62</b>（+8.4%）。</li>   <li>驅… |
| `docs/MTP_VERIFY_HEADROOM_2026-09-24.md` | 138 | `12.62` | ✅ | …speculative-simple` 口徑（`MEMORY_PERF.md`：「歷史的 12.62 vs 9.82 是 HTTP 口徑」），與交付… |
| `docs/MTP_VERIFY_HEADROOM_2026-09-24.md` | 122 | `9.82` | ⛔ 未標記 | …測估計值遠高於門檻  用 `step(T=1) ≈ 101.83 ms`（MTP-off 9.82 t/s 反推）與 `step(T=4) = 247… |
| `docs/MTP_VERIFY_HEADROOM_2026-09-24.md` | 138 | `9.82` | ✅ | …ve-simple` 口徑（`MEMORY_PERF.md`：「歷史的 12.62 vs 9.82 是 HTTP 口徑」），與交付 anchor（ll… |
| `docs/ORNITH_ALIGNED_AB_AND_25_ARITHMETIC_20260918_0445.html` | 100 | `12.62` | ⛔ 未標記 | …<ul>   <li><b>取代 §6.2／§6.3 的「去序列化 ×1.8–2.3」與「12.62 × 2.0 ＝ 25.24 剛好擦線」</b>：… |
| `docs/ORNITH_ALIGNED_AB_AND_25_ARITHMETIC_20260918_0445.html` | 134 | `12.62` | ⛔ 未標記 | …，但探針輸出損壞）</b>與 <b>MTP（×1.28，已量）</b>。算術 <code>12.62 × 2.0 ＝ 25.24</code> 剛好擦… |
| `docs/ORNITH_ALIGNED_AB_AND_25_ARITHMETIC_20260918_0445.html` | 326 | `12.62` | ✅ | …f）</td>     <td class="num">MTP-on：9.82 → <b>12.62</b>（×1.28）</td>     <td… |
| `docs/ORNITH_ALIGNED_AB_AND_25_ARITHMETIC_20260918_0445.html` | 367 | `12.62` | ✅ | …xt> </svg> <h3>§6.3 算術，與三個警告</h3> <pre><code>12.62（MTP-on 實測）× 2.0（去序列化，未證）… |
| `docs/ORNITH_ALIGNED_AB_AND_25_ARITHMETIC_20260918_0445.html` | 369 | `12.62` | ✅ | …code></pre> <ol>   <li><b>兩個乘數能不能相乘，還沒量過。</b>12.62 與 31–44 ms/步是<b>不同儀器</b>… |
| `docs/ORNITH_ALIGNED_AB_AND_25_ARITHMETIC_20260918_0445.html` | 438 | `12.62` | ✅ | …</li>   <li><b>不主張 25 是一個計畫。</b>它是一條算術：<code>12.62 × 2.0</code>。缺的那一步（儀器）在… |
| `docs/ORNITH_ALIGNED_AB_AND_25_ARITHMETIC_20260918_0445.html` | 326 | `9.82` | ✅ | …1.0（MTP off）</td>     <td class="num">MTP-on：9.82 → <b>12.62</b>（×1.28）</td… |
| `docs/PREFILL250_DECODE25_STRATEGY_20260919.html` | 178 | `9.82` | ⛔ 未標記 | …<b>1.00</b>（依定義）</td><td><b>~105 ms</b>（9.18–9.82 t/s ÷ 1.0）</td><td>9.18–9… |
| `docs/PREFILL250_DECODE25_STRATEGY_20260919.html` | 178 | `9.82` | ⛔ 未標記 | …105 ms</b>（9.18–9.82 t/s ÷ 1.0）</td><td>9.18–9.82</td><td>—</td></tr> </tbo… |
| `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` | 353 | `12.62` | ⛔ 未標記 | …lass="num">21.74</td><td class="num"><strong>12.62</strong></td></tr> </tab… |
| `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` | 356 | `12.62` | ⛔ 未標記 | …ble>  <p>這張表把 M4 那條離開條件「MTP-on ≥ MTP-off」補上了（12.62 ≥ 9.82，+28.5%）。修前是淨負 （8.… |
| `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` | 372 | `12.62` | ⛔ 未標記 | …iv class="box b-amber">   <p><strong>與 4.1 的 12.62 不可直接相比，原因有兩個，而且都不是「哪個比較快… |
| `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` | 382 | `12.62` | ⛔ 未標記 | …accept 從 73.81% 降到 58.25%，而 decode 從 8.62 升到 12.62</strong>。 兩個都是真的，而且不衝突 —… |
| `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` | 356 | `28.5` | ⛔ 未標記 | …M4 那條離開條件「MTP-on ≥ MTP-off」補上了（12.62 ≥ 9.82，+28.5%）。修前是淨負 （8.62 &lt; 9.82）—… |
| `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` | 351 | `9.82` | ⛔ 未標記 | …td><td class="num">19.24</td><td class="num">9.82</td></tr> <tr><td>MTP on，… |
| `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` | 356 | `9.82` | ⛔ 未標記 | …>這張表把 M4 那條離開條件「MTP-on ≥ MTP-off」補上了（12.62 ≥ 9.82，+28.5%）。修前是淨負 （8.62 &lt;… |
| `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` | 357 | `9.82` | ⛔ 未標記 | …ff」補上了（12.62 ≥ 9.82，+28.5%）。修前是淨負 （8.62 &lt; 9.82）—— 也就是說，這個功能在修好之前是被一個<str… |
| `docs/PROD_MATRIX_RECHECK_2026-09-18.md` | 3 | `9.82` | ⛔ 未標記 | …denseIQ4X.gguf / NOMINAL / NOMINAL / 10.61 / 9.82 / / prefill250 / decode-u… |
| `docs/PROD_MATRIX_STANDARD_20260917_1630.html` | 147 | `12.62` | ⛔ 未標記 | …putime</code>）</td></tr> <tr><td class="num">12.62</td><td>MTP A/B（HTTP <co… |
| `docs/PROD_MATRIX_STANDARD_20260917_1630.html` | 359 | `12.62` | ⛔ 未標記 | …draft/MTP</code> 在它的原始碼零命中）。所以歷史上的     <code>12.62 vs 9.82</code>（MTP 由淨負轉淨… |
| `docs/PROD_MATRIX_STANDARD_20260917_1630.html` | 359 | `9.82` | ⛔ 未標記 | …</code> 在它的原始碼零命中）。所以歷史上的     <code>12.62 vs 9.82</code>（MTP 由淨負轉淨正）在「只認 <c… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 848 | `12.62` | ⛔ 未標記 | …ault) / **58.25%** / 180/309 / **21.74** / **12.62** /  * **M4's exit condi… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 850 | `12.62` | ⛔ 未標記 | …/  * **M4's exit condition is met**: MTP-on 12.62 >= MTP-off 9.82 (+28.5%)… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 924 | `12.62` | ⛔ 未標記 | …*fixed (nb-aware)** / 58.25% / **21.74** / **12.62** / / `prefill250` / pre… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 929 | `12.62` | ⛔ 未標記 | …. So **MTP-on >= MTP-off in both profiles** (12.62 vs 9.82; 11.79 vs 9.53),… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 850 | `28.5` | ⛔ 未標記 | …ion is met**: MTP-on 12.62 >= MTP-off 9.82 (+28.5%). In the pre-fix arm MTP… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 846 | `9.82` | ⛔ 未標記 | …TP off (denominator) / n/a / 0/0 / 19.24 / **9.82** / / MTP on, `CGC_IDS_LI… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 850 | `9.82` | ⛔ 未標記 | …condition is met**: MTP-on 12.62 >= MTP-off 9.82 (+28.5%). In the pre-fix… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 851 | `9.82` | ⛔ 未標記 | …e-fix arm MTP was   net **negative** (8.62 < 9.82) -- i.e. the feature was… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 928 | `9.82` | ⛔ 未標記 | …e binary, same pool): default profile decode 9.82 t/s; `prefill250` decode… |
| `docs/ROUTING_TRACE_2026-09-17.md` | 929 | `9.82` | ⛔ 未標記 | …P-on >= MTP-off in both profiles** (12.62 vs 9.82; 11.79 vs 9.53), which is… |
| `docs/S1_FRONTIER_2026-09-18.md` | 21 | `12.62` | ✅ | …寫。 > 修後：**M1/M2/M3 對 v6 全 9/9、decode 8.62 → 12.62 t/s、prefill 命中率 57.1% →… |
| `docs/S2MOE_VERIFY_COST_ASSESSMENT_2026-09-24.md` | 138 | `28.5` | ⛔ 未標記 | ….3×/2.0×」就整套移植 —— 那兩個數字是相對**自回歸**，我們已經有 MTP（+28.5%），基準不同。  ---  ## 6. 來源  /… |
| `docs/S2MOE_VERIFY_COST_ASSESSMENT_2026-09-24.md` | 153 | `28.5` | ⛔ 未標記 | …、`2026-09-24.md` §EN-476/479 / / MTP on/off +28.5%、M-W ~+1.7% / `MEMORY_PER… |
| `docs/SHAPE_FOR_25_2026-09-22.md` | 123 | `28.5` | ⛔ 未標記 | …- 今天 ml **沒有乾淨值**：記憶裡並存 1.28（MTP on/off = +28.5%）／1.7／3.375   ⇒ step 可能是… |
| `docs/TODAY_TECH_DECISIONS.html` | 62 | `12.62` | ✅ | …≈ 12.0–12.9（8 GiB pool、prod25 profile）；配對中位 12.62。需要 2.07×。</td><td>2.07×（… |
| `docs/TODAY_TECH_DECISIONS.html` | 62 | `12.62` | ✅ | …）；配對中位 12.62。需要 2.07×。</td><td>2.07×（要 25、現在 12.62）。已判掉的三條路合計上界 ≈17–20 ⇒ <b… |

## §4 機檢與維護

```sh
python3 scripts/check/void_number_check.py              # 決策面（違規 ⇒ exit 1）
python3 scripts/check/void_number_check.py --all        # 另列歸檔面稽核
python3 scripts/check/void_number_check.py --citations  # 重生本檔 §3 的表
python3 scripts/check/void_number_check.py --selftest   # 12/12
```

**設計要點**
- **分層**：決策面強制、歸檔面稽核 —— 因為「dated 產物不回改」與「不得再被引用」必須同時成立。
- **判定粒度＝段落**（空行分段）：同段有 `作廢／廢棄／不可引用／⛔` 即放行。
- **誤報防護**：百分比型（登記表寫成 `+28.5%`）要求 **真的接 `%`** 且**同段出現 MTP／spec／加速／攤薄**，
  否則 `28.5` 這種常見數值會誤報（實測已排除 `ZERO_REGION_TRIAGE` 的 28.5% 佔比）。
- **數字邊界**：`19.82` 不命中 `9.82`。

**如何新增作廢數字**：只能加進 `MEASUREMENT_CONTRACT` **§7 的表格第一欄**（反引號內寫全，
例：`` `12.34 t/s` ``；百分比型寫 `+xx.x%`），並補齊理由與替代 ⇒ 檢查器**自動**納入。
