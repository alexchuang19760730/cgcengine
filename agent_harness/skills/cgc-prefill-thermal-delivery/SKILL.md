---
name: cgc-prefill-thermal-delivery
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）上讓一個 prefill t/s 數字取得可交付的地位。核心是一條 11 ms、不需要 root 的讀數 notifyutil -g com.apple.system.thermalpressurelevel（0=Nominal）：發射前讀到 0 才授權 250 級數字。當使用者問「prefill 250 交付了嗎」「這個 t/s 能不能引用」「prefill 為什麼忽快忽慢」「能不能 conditional 交付」「怎麼量散熱條件」、要跑 prefill 驗收、或要比較兩個 prefill 數字時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-prefill-thermal-delivery/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-17 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# prefill 吞吐的條件式交付（flashkv-devserver）

專案：`/Users/alexchuang/Documents/flashkv-devserver` ·
目標：`CGC_SERVER_PROFILE=prefill250`，Qwen3.6-35B-A3B（`Nail-…-denseIQ4X.gguf`）
· 機器：MacBook Air M4 16GB，**無風扇**。

**這份 skill 解決的問題**：同一份 binary、同一個 A11 指紋、同一組 `[perf]`，
prefill 可以在 118 與 290 t/s 之間移動。於是一句「250 交付了嗎」可以有兩個都找得到支持的答案。
這裡寫的是**判準**，不是感覺。

---

## 1. 條件（一句話）

```bash
notifyutil -g com.apple.system.thermalpressurelevel     # 必須是 0
```

**0 = Nominal**（1 Moderate / 2 Heavy / 3 Trapping / 4 Sleeping）。
這是 **notify(3) 的 key，也正是 `powermetrics` 的 thermal sampler 用來印 `Current pressure level` 的那一條**。
不需要 root、約 **11 ms/次**（20 次 0.225 s）、可 2 Hz 取樣不擾動被測量。

**判準的實測分離度（request 級，零重疊）**

| 發射時等級 | ≥250 | 範圍 |
|---|---|---|
| `0` Nominal | **6/6** | 253.42 – 271.64 |
| `1` Moderate / `2` Heavy | **0/21** | 104.88 – 211.65 |

間隙 41.77 t/s。

★ **但「發射時讀到 0」不足——2026-09-16 更正。** `run_req2_retest.sh:327-350` 自己有分類，
而它訂的門檻高得多：

| 標籤 | 條件 | 授權 |
|---|---|---|
| `COLD-STATE` | `quiet since previous arm ended >= 1800 s` | **交付級**：這個 t/s 可以引用（whitepaper §11.14） |
| `HOT-STATE` | 否則 | 「**do NOT quote it as the delivery number**」 |

它附的分離度就是證據：**同一個 binary／配置／指紋**，COLD 得 **254.29 / 282.38 / 265.27**，
HOT 得 **167.41–200.58**。⇒ **「發射時讀到 0」與「箱子是冷的」是兩件事**，
而一個 250 級數字要能被引用，需要的是後者。`0` 是閘門開，不是冷。

**跑 A/B 時更要注意兩件事**：
- **靜置時間不是你可以指定的值。** 它取決於前一臂留下的熱（實測同一晚兩個臂分別只等到 120 s 與 195 s）
  ⇒ 「每臂等讀數回 0」**不會**讓兩臂配對在相同的靜止條件上。可引用的 A/B 要固定靜置長度，或直接等 COLD。
  反面教材：2026-09-16 的 prefill A/B，四個臂裡**最慢的兩個正是只靜置 40 s 的兩個**，
  而被測變數（SPAC）的兩輪**符號相反** ⇒ 量到的是「安靜多久」。這種資料要**作廢**，不要解釋它。
- **失敗要 fail closed，不要「跑了然後貼標籤」。** 等不到就跳過該臂，並在總結裡留一個洞——
  一個熱態數字加上「熱態」標籤，比沒有數字更容易被下一個人誤引。

**條件讀的是「發射前那一刻」，不是「全程」。** 15:18:46 那臂發射時 0，中途升 1→2，
三個請求仍全部 ≥250 ⇒ 中途上升不撤回該臂。機制（governor 反應落後）是**假設**，不是結論。

---

## 2. 鐵律

1. **prefill t/s 是機器狀態的讀數，不是產物性質。** 引用前先讀 §1 的閘門。
2. **條件要「讀」，不要「推」。** 「安靜夠久 ⇒ 等級一定回 0」是錯的（§1 那臂就是反例）。
   安靜秒數與「≥150 s」都不是條件（安靜 18 s 的臂 req1 = 289.86，安靜 34 s 的臂只有 166.95）。
3. **`swap` 不是條件，是果。** 反例順序相反：289.86 t/s @ `used 4975.06M`、188.35 @ `4549.69M`、
   13:51 慢臂 @ `5066.06M`；而那 32 分鐘安靜期間它自己從 5066.06M 回到 2743.00M。
4. **要量「兩臂之間空了幾秒」，用事件檔案的 mtime，不要用驅動腳本的報告戳記。**
   戳記是腳本啟動時間，每臂自己約 90 s ⇒ 相減會把「上一臂在跑」算成「上一臂在休息」。
   實測差一個量級：戳記法 111/94/144 s vs 真實 21/8/55 s。
5. **排除一個候選要找「順序相反」的配對，不是找相關。** 相關會把同一個潛在變數的兩個果配在一起。

---

## 3. 判準數字（2026-09-16 量到，直接照用）

- **閘門開（發射 0/NOMINAL）**：`266.81 / 269.35 / 271.64`（15:31:04）與
  `257.41 / 253.42 / 262.64`（15:18:46）；兩臂都 `survived=yes` 到 req3、0 crash report。
- **閘門關（發射 2/HEAVY）**：`196.42 / 211.65 / 189.49`、`143.89 / 142.14 / 120.74`、
  `104.88 / 150.25 / 127.43`；ABBA 四臂 117.53–169.09。
- **時脈 → 吞吐**：`t(ms/token) = a + b/f_eff`，`a≈0.47、b≈4216`（合併 6 點）。
  250 t/s 對應有效時脈 **1154–1227 MHz**。1470 MHz → 约 291、928 → 227、618 → 135。
  看到 ~185 就是「兩階之間」。
- **可交付的述句**：「**讀到 0 的那一臂**，其 req1–req3 全部 ≥250」。
  **不可**說「250 隨時可重現」。
- **powermetrics 佐證（block 級）**：Nominal 2/2 → ≥250（276.25、300.43）；
  非 Nominal 4/4 → <250（227.27、151.28、182.39、145.79）。

---

## 4. 怎麼跑

```bash
cd /Users/alexchuang/Documents/flashkv-devserver

# 閘門：條件不成立就不跑臂（exit 3，fail closed）。要刻意產熱態樣本才用 GATE_FORCE=1
ARMS=2 OUTDIR=Backup/cgc_logs/thermal_gate bash Backup/run_thermal_gate.sh

# 驗收臂自帶證據（in-band 讀數寫進報告）
IDLE_BEFORE=0 OUTDIR=Backup/cgc_logs/x bash Backup/run_req2_retest.sh
#   報告會印：
#     # [thermal pressure] before launch = 0/NOMINAL   (...)
#     [thermal pressure] before req1 = 0/NOMINAL
#     ...
#     high-water : 0/NOMINAL -- condition SATISFIED across this arm
#   讀不到時印 ?/UNREADABLE，不會靜默當成 0。

# 外掛 2 Hz 序列（看臂中途的變化）
bash Backup/thermal_pressure_probe.sh Backup/cgc_logs/x/pressure.tsv 0.5 900
bash Backup/thermal_pressure_probe.sh --summarize Backup/cgc_logs/x/pressure.tsv

# 有 root 時的權威版本（DVFS 駐留分佈；**不是閘門**，需要人啟動）
sudo scripts/check/powermetrics_gpu_freq.sh
python3 scripts/check/powermetrics_gpu_freq_parse.py Backup/cgc_logs/powermetrics_prefill_*.log
```

量兩版產物的差異用 ABBA 交錯（不要 A-then-B）：

```bash
bash Backup/run_lib_ab.sh        # 檔案互換 + 一臂暖機丟棄；退出時自動還原
```

---

## 5. 陷阱（都踩過）

1. **不要用 `NSProcessInfo.thermalState`。** 它看起來就是那個缺失的儀器（非 root、Foundation、
   四級刻度），但 367 個樣本跨越滿載與 4 分鐘閒置**全部讀到 `1/fair`**，零區辨力。
   **找到一個介面不等於找到一個儀器；認證靠它產生的分離度。**
2. **不要寫「沒有儀器可讀」。** 那是關於某個工具的推論。要寫「試過哪些介面、各自的結果」。
   （`pmset -g therm` 無資訊、`sysctl` 沒有鍵、`ioreg -c IOAccelerator` 有利用率而**沒有時脈**、
   `GPU Performance States` 只暴露 channel id 不是讀數。）
3. **不要在同一則訊息裡對同一個檔案送出兩個編輯** —— 會 race，其中一組靜默消失。
   若「呼叫點在、定義不在」，未定義函式在 `$( )` 裡只讓 stdout 變空字串 ⇒
   報告出現**空白讀數**，看起來像儀器讀不到。收尾用 `grep -n` 確認定義與呼叫都在。
4. **「改了原始碼沒重建」會讓整段量測屬於舊產物。** 判準是 `cmake --build` 有沒有印編譯行
   （exit code 在「已最新」與「剛編好」都是 0）。
5. **`sysctl -n vm.loadavg` 在 idle 也不是 0**（實測 1.71–6.47）：常駐兩個 `Xcasca`（Electron）
   renderer，而無風扇 M4 是**共享熱包絡** ⇒ 背景負載會吃掉 GPU 的散熱餘裕。未歸零的混淆項。
   `ps` 在本 sandbox 被擋，列行程用 `pgrep -fl`。
6. **`Backup/` 與 `.workbuddy/` 都在 `.gitignore` 內**（`.gitignore:396`、`:41`）。
   要交付就得 `git add -f` 或搬進 `scripts/`，否則修正只存在於本機。

---

## 6. 相關文件

- 可操作摘要：`docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html`（＋ `.md`）
- 白皮書：`docs/PREFILL250_THERMAL_TRANSIENT_20260916.html`
  §3.3（idle 掃描）、§10（powermetrics 因果鏈）、§11.14（條件式交付；**§11.14.7 已被 §11.15 推翻**）、
  **§11.15（可量測的條件）**。
- 儀器：`Backup/thermal_pressure_probe.sh`（2 Hz 序列）、`Backup/run_thermal_gate.sh`（閘門）、
  `Backup/run_lib_ab.sh`（ABBA 檔案互換）、`Backup/run_req2_retest.sh`（in-band 讀數）、
  `scripts/check/prefill_certifiability.py`（`--idle-before` **只在第一次啟動前生效**）、
  `scripts/check/powermetrics_gpu_freq{,_parse}.py`。
- 慣例：`agent_harness/CONVENTIONS.md` **B29**（條件必須可量測）、**B30**（事件時間戳）、
  **B31**（平行編輯 race）、**B32**（「沒有儀器」是工具的推論）。
- 同 repo 的 decode 側：skill `cgc-decode-attribution`。
