---
name: cgc-prefill-thermal-delivery
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）上決定一個 prefill t/s 數字能不能被引用。核心是一條 11 ms、不需要 root 的讀數 notifyutil -g com.apple.system.thermalpressurelevel（0=Nominal）＋臂自帶的 COLD-STATE/HOT-STATE 標籤。★2026-09-17 更正：這兩個條件都只是**必要條件**，**不**保證 ≥250（實測 COLD-STATE 下量到 242/227/249）。當使用者問「prefill 250 交付了嗎」「這個 t/s 能不能引用」「prefill 為什麼忽快忽慢」「能不能 conditional 交付」「怎麼量散熱條件」、要跑 prefill 驗收、或要比較兩個 prefill 數字時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-prefill-thermal-delivery/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-20 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
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

★★ **但「COLD-STATE」也不足以推出 ≥250 —— 2026-09-17 15:37 實測。**
`prefill250`／同 build／同 profile／同一份 2873-token prompt，在安靜 **1940 s**
（臂報告自己判 `COLD-STATE`、並引用 whitepaper §11.14 說「其 t/s 可以引用」）的窗口量到
**242.42 / 227.27 / 249.06 t/s —— 三個全部低於 250**，而發射前與每個請求**之前**都是 `0/NOMINAL`
（外掛 2 Hz 序列顯示 `15:37:15 → 15:38:24` 連續 Nominal，req1／req2 完全落在那段之內）。
⇒ **上面那張分離度表與 §3 的「可交付的述句」都只是「該批樣本」的觀測，不是門檻的保證。**
`COLD-STATE` 的授權要改讀成「**該臂的讀數可以引用**」，**不是**「該臂達標 ≥250」。
反向也一樣：同日 15:03 的臂標籤只是 `UNKNOWN-STATE`，卻得 **278.56 / 261.17 / 275.01**
⇒ **安靜秒數與等級都不預測這個 t/s**。兩臂的池 IO 幾乎相同（`pread_usec` 1501 vs 1548 s、
`us/job` 31733 vs 32533）⇒ 不是池路徑。未歸因的候選是**背景 GPU 客戶端**（無風扇 M4 的熱／功耗
包絡共享；發射時 load 1 分鐘值 2.93 vs 2.42，同日 15:39 量到 6.18）⇒ **閘門要在請求當下一起記錄
負載**；能裁決的直接變數是 `powermetrics` 的 GPU 時脈駐留（需 root），熱等級只是它的粗代理。
lesson `eng-mh-0054`。

**跑 A/B 時更要注意三件事**：
- **靜置時間不是你可以指定的值。** 它取決於前一臂留下的熱（實測同一晚兩個臂分別只等到 120 s 與 195 s）
  ⇒ 「每臂等讀數回 0」**不會**讓兩臂配對在相同的靜止條件上。可引用的 A/B 要固定靜置長度，或直接等 COLD。
  反面教材：2026-09-16 的 prefill A/B，四個臂裡**最慢的兩個正是只靜置 40 s 的兩個**，
  而被測變數（SPAC）的兩輪**符號相反** ⇒ 量到的是「安靜多久」。這種資料要**作廢**，不要解釋它。
- **失敗要 fail closed，不要「跑了然後貼標籤」。** 等不到就跳過該臂，並在總結裡留一個洞——
  一個熱態數字加上「熱態」標籤，比沒有數字更容易被下一個人誤引。
- **★ 一個 ABAB 序列的第一個臂不能與後面的臂交換（2026-09-17 實測）。** 同一個 `CGC_SPAC=0`
  配置在序列**位置 1** 量到 req1 ＝ **215.30**、在**位置 3** 量到 **287.80**（差 **72.50**），而兩臂的
  **池讀 bytes 完全相同**（2 784 305 152、`us/job` 只差 3%）⇒ 這個溢價**不在池的 IO 路徑**，是「第一次
  啟動」本身的代價（page cache／Metal pipeline 暖機）。後果：`off,on,off,on` 的 **pair 1 會被 pos1
  溢價污染成看起來像處置效果**（實測配對差 req1 ＝ +68.06，而 pair 2 ＝ −2.98，**符號相反**）。
  ⇒ **要 A/B，先跑一個丟棄臂**，讓每個回報臂都落在位置 ≥2；否則 req1 一律「未歸因」。
  這也是「安靜多久」被誤讀成「SPAC 有代價」的機制。

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
- **可交付的述句（2026-09-17 修正）**：可引用的是「**讀到 0 的那一臂**，其 req1–req3 讀數是
  **X / Y / Z**」＋ 熱標籤。**❌ 舊述句「其 req1–req3 全部 ≥250」已被 lesson `eng-mh-0054` 否證**
  （同一個條件集、同一天量到 242.42 / 227.27 / 249.06）⇒ 標籤背書的是「這個讀數的來歷」，
  **不是「這個門檻」**。
  **不可**說「250 隨時可重現」。
- **powermetrics 佐證（block 級）**：Nominal 2/2 → ≥250（276.25、300.43）；
  非 Nominal 4/4 → <250（227.27、151.28、182.39、145.79）。
- **★ `prefill250 + CGC_SPAC=1` 的 prefill 代價：不成立（2026-09-17 07:0x，同 build COLD 交錯
  off/on/off/on，四臂全 `COLD-STATE`）**：req1 off {215.30, 287.80} median 251.55 vs on {283.36, 284.82}
  median 284.09；req2 268.51 vs 285.95；req3 291.03 vs 295.11。**唯一的大落差是 pos1 溢價**
  （見 §1 第三條），req2／req3 的方向甚至與「代價」**相反**（on 快 1.4–6.5%）——但反向增益
  也**未取得交付地位**（n=2，req3 的 +4.08 落在 off 自身 3.54 範圍內）。池歸因：>99.8% 全 compulsory，
  off 永遠 `2604/0`、on 永遠 `2502/5`（**位置不變 ⇒ 可當 flag 真的套用了的第二個證據**），
  SPAC=on 少讀 3.9% bytes 但**不買也不付** prefill。**要裁決「反向增益」的下一設計＝丟棄臂開頭 ＋
  鏡射後半（`off,on,off | on,off,on`）**，資產在 `Backup/cgc_logs/spac_cold_ab/RESULT.md`。

---

## 4. 怎麼跑

**★ prefill 目前有兩個入口，而「哪一個是交付口徑」在 2026-09-17 仍 **未定**。** 引用前先指名入口：

| 入口 | 指令 | 形狀 | 已記錄的值 |
|---|---|---|---|
| **HTTP 驗收臂**（本 skill §1–§3 的交付規程就是為它寫的） | `IDLE_BEFORE=0 OUTDIR=… bash Backup/run_req2_retest.sh` | **2873-token prompt** 的 `prompt eval t/s`，每請求邊界帶熱讀數 | 278.56／261.17／275.01（`UNKNOWN-STATE`） |
| **llama-bench `pp2048`**（走**同一支** `llama_bench_matrix.py`；**自家形狀**） | `python3 scripts/check/prefill_certifiability.py --arm prefill250 --prompt 2048 --gen 16 --reps 3 --runs 5` | `-p 2048 -n 16 -d 0 -b 5632 [profile]`（`-b/-ub` 由 `run_server.sh CGC_DUMP_ENV=1` 取）；散熱靠 2 Hz `thermal_pressure.Sampler`——它是獨佔 GPU 的子行程，**沒有**每請求邊界可以掛讀數 | 276.25／300.43（Nominal ×2，≥250）；227.27／151.28／145.79／182.39（非 Nominal） |
| **llama-bench `pp512`**（**上游可比**，2026-09-17 追加） | `… --prompt 512 --gen 128 --depths 0 --reps 3 --runs 5 --batch 2048` | `-p 512 -n 128 -d 0 -b 2048 [cli]` ＝ 上游預設（`llama-bench.cpp:367-377`；標準列 `pp512`，`README.md:180-187`） | **尚無** |
| **`prod_matrix.py` 的四格**（**2026-09-17 17:43 第一次實跑**；`7ba001802` 的標準） | `python3 scripts/check/prod_matrix.py --profiles prefill250 --cells decode,decode-up,prefill-house,prefill-up --reps 3 --json … --md …` | 四格各自一次啟動；`prefill-house`＝`-p 2048 -n 16 -d 0 -b 5632`、`prefill-up`＝`-p 512 -n 128 -d 0 -b 2048`、`decode`／`decode-up`＝`-p 0 -n 128 -d 512 -b 512/2048`。**平台值＝丟掉 rep 1**（工具自算 `platform_ts`，`n_kept` 一併印） | 見白皮書 `docs/PROD_MATRIX_FIRST_RUN_20260917_1755.html`：decode 8.89／9.33、prefill 102.17／102.73（**四格 launch 全非 NOMINAL ⇒ 工具自己判為 HOT，不可當生產數字**） |

**★ 那四格是 HOT 樣本，而且是結構性的（2026-09-17 實測）。** 四格的 launch 戳記與 `wall_s` 相減，
格間空檔只有 **0.9／0.0／1.6 秒**，而 `Backup/run_req2_retest.sh:325` 的 `COLD_QUIET=1800`
（分類在 `:362-371`）要求「前一臂結束後安靜 ≥1800 s」才叫 `COLD-STATE`。以 0–2 秒間隔連續發射
⇒ **第 2–4 格必然繼承前一格的熱**（實測 launch 由 MODERATE 走到 HEAVY、launch 前 swap
由 4445.19 走到 6646.25 MiB）。⇒ **單次呼叫 `prod_matrix.py` 不可能拿到 NOMINAL 的 prefill 格**
（除非第一格剛好是冷的）。要可交付的數字，改成**一格一次呼叫、格間等冷**。

**★ 另外兩個入口的坑（同一次實測）：**

- **`--dry-run` 會建目錄**（`prod_matrix.py:249` 的 `rdir.mkdir` 在 `:259` 的 `if args.dry_run`
  **之前**）⇒ 乾跑一輪就在 `Backup/prod_matrix/` 留 28 個空目錄，命名與真跑的一模一樣。
  **乾跑請自帶 `--logdir /tmp/…`**（本輪以「空目錄數 124→124」證實建立點）。
- **`gate()` 只看 8080 與量測行程，不看「有人在重建」** —— 而本 repo 的比較全建立在同一個 build 上。
  實例：四格 17:48:36 結束，另一條線 17:48:30/31/34/37 換掉兩個 dylib 與兩支執行檔。
  **動手前自己補一條 `ps -Ao command= | grep -E "cmake|ninja"`**。
- **`--list` 的輸出受機器狀態影響**：機器被佔時它對 4 個 profile 印
  `resolve failed … (rc=1)`（讀起來像 profile 壞了）；機器空了 7 個全過。要引用那份對照表前先確認機器是空的。


- **兩個 cell 必須是兩次獨立 run**：`llama-bench` 同一行程內每個形狀共用模型與池 ⇒
  `-p 512,2048` 會讓第二格繼承第一格暖過的池，兩格不獨立。
- **`pp512` 只是「同一個標籤」，要真的可比還得 `-b/-ub` 也對上** ⇒ 用 `--batch 2048`。
  這件事本來做不到（`prefill_certifiability.py` 沒有 `--batch` 透傳，固定吃 profile 的 5632），
  **2026-09-17 已補**（`--batch/--ubatch/--dry-run`），並可用 `--dry-run` 零 GPU 驗證形狀。
- **`tag` 不含模型檔** ⇒ 每份產出都要記 `-m` 那一行。現行 harness 下 `--arm prefill250` 與
  `--arm prod25` **都載 `Nail-…-denseIQ4X.gguf`**（只有顯式 `CGC_SERVER_MTP=0` 才換檔）。

- ⚠️ **兩邊的數字看起來很近（276.25 vs 278.56），但那是兩個不同的量**（`-p 2048` vs 2873-token
  prompt、不同 `-b/-ub` 歷史值 6144 vs 5632）⇒ **不得並排、不得互相換算**。
- ⚠️ **llama-bench 那條自己的頭注就說它是抽籤不是規格**：`pp2048 @ -ub 6144` 四次獨立啟動得
  **276.59／198.84／176.18／122.68 ＝ 2.25× 離散**，而每次啟動**內部**三次樣本只差 2.08–15.41
  ⇒ 離散來自「行程之間的狀態」，不是量測雜訊。**所以「統一用 llama-bench」對 prefill 不是免費的**：
  它把問題從「熱條件不足」換成「跨啟動離散」，而後者還沒被解決。
- decode 已於 2026-09-17 統一在 llama-bench（見 skill `cgc-decode-attribution`）。

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
5. **`sysctl -n vm.loadavg` 在 idle 也不是 0**（實測 1.71–6.47）：常駐 Electron renderer
   （`Xcasca`／`Freebuff Helper (GPU)`／`WorkBuddy Helper (GPU)`／`WebKit GPU`／`WindowServer`），
   而無風扇 M4 是**共享熱／功耗包絡** ⇒ 背景負載會吃掉 GPU 的散熱餘裕。未歸零的混淆項。
   ★ 2026-09-17 更正它的地位：它不只是混淆項，它是「`COLD-STATE` 仍量到 227–249」那次否證的
   **第一候選**（見 §1）⇒ 閘門與臂報告要在**請求當下**一起記一筆負載，否則只能停在「未歸因」。
   列行程用 `ps -Ao pid=,etime=,command=`（可用）；純 `pgrep -f` 是**命令列文字比對**，
   要小心自己的指令文字被算成匹配（見第 7 條）。
   ★ 2026-09-17 21:0x 補：**`pgrep -x llama-server`（basename 精確比對）是這裡最穩的讀法** ——
   它既不是命令列文字比對（躲開第 7 條），也不受 `comm` 欄位語意影響。實測本機
   `ps -Ao pid=,comm=` 印的是**完整路徑**（`47353 /Users/…/build/bin/llama-server`），
   所以 `awk '$2=="llama-server"'` **永遠不命中** ⇒ 一種靜默的假陰性（我在 EN-80 因此兩次把
   「自己剛啟動、正在載入的行程」讀成「被系統殺了」）。完整對照見 `cgc-commit-gate` §1.5。
6. **`Backup/` 與 `.workbuddy/` 都在 `.gitignore` 內**（`.gitignore:396`、`:41`）。
   要交付就得 `git add -f` 或搬進 `scripts/`，否則修正只存在於本機。
7. **★ 你自己的指令文字會讓整臂被記憶體閘門擋掉（2026-09-17 實測，毀掉一個 arm）。**
   本 harness 把**每一條指令**跑成 `/bin/zsh -c … eval '<指令全文>'` ⇒ **指令全文在行程 argv 裡**。
   而 `run_req2_retest.sh:88` 的 `alive()` 與 `run_server.sh:534` 的 `cgc_existing_llama_server_count()`
   用的是**裸 `pgrep -f "build/bin/llama-server"`**（`run_server.sh:660-663` 早已記載「純 pattern 會打到
   命令列裡剛好出現 llama 路徑的上層 wrapper」並在 **preflight** 修成 ps 驗證版，但**沒改這兩處**）。
   實例：我一邊等臂、一邊輪詢 `pgrep -f 'build/bin/llama-server'`，那句指令本身就被閘門算成
   `other_llama_servers=1`。
   **症狀（認這三行就不會誤判成 SPAC/程式的問題）**：報告 `RESULT: ready=no last_request=0`、`[argv]` **空白**、
   `driver.log` 出現 `error: startup blocked by memory guard -> full-mtp: other_llama_servers=1>0`，
   而**同一支腳本**的預檢卻印 `[preflight] 無殘留 llama 行程`（ps 驗證版讀 0 ⇒ 兩個計數器互相矛盾就是這個病）。
   它還會讓 `alive()` 報「a llama-server is ALREADY resident」並白等 180 s。
   **為什麼 in-band 讀不到**：`pgrep` 只排除自己與**祖先** ⇒ 你的輪詢在自己的 shell 裡印 `idle`，
   對**別的**行程（那個跑閘門的腳本）卻是在廣播自己。**儀器對自己安慰、對別人廣播。**
   **紀律**：COLD 窗口期間，自己的指令文字**不得出現 `build/bin/llama-server`**。要探測行程就用
   bracket 技巧（`pgrep -f 'build/bin/llama[-]server'`）——它自己的文字不匹配該 regex。
   修法在 `run_server.sh:660` 已經寫好，就是叫那兩處改用 `cgc_preflight_pids`。
8. **停掉一個正在跑的臂，要驗證它真的停了，而且 kill 的 pattern 別打到自己的 shell。**
   `pkill -f 'run_spac_cold_ab.sh'` 的文字本身會被那個 pattern 匹配（第 7 條的同一個機制）⇒
   用 `run_spac_cold_ab[.]sh`。停完 `pgrep -fl` 逐支確認（實測留了一支 MARKER 載體沒死，
   它會在下一臂發射時變成 `other_llama_servers=1`）。

---

## 6. 相關文件

- 可操作摘要：`docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html`（＋ `.md`）
- 白皮書：`docs/PREFILL250_THERMAL_TRANSIENT_20260916.html`
  §3.3（idle 掃描）、§10（powermetrics 因果鏈）、§11.14（條件式交付；**§11.14.7 已被 §11.15 推翻**）、
  **§11.15（可量測的條件）**。
- 儀器：`Backup/thermal_pressure_probe.sh`（2 Hz 序列）、`Backup/run_thermal_gate.sh`（閘門）、
  `Backup/run_lib_ab.sh`（ABBA 檔案互換）、`Backup/run_req2_retest.sh`（in-band 讀數）、
  `Backup/run_spac_cold_ab.sh`（COLD 交錯 A/B，off/on/off/on；**它的 pre-flight 只驗 env dump，
  不驗行程** ⇒ 擋臂的仍是 `run_server.sh` 裡那道裸 pattern 的閘門）、
  `scripts/check/prefill_certifiability.py`（`--idle-before` **只在第一次啟動前生效**）、
  `scripts/check/powermetrics_gpu_freq{,_parse}.py`。
- SPAC COLD 交錯 A/B 的結果與儀器事故：`Backup/cgc_logs/spac_cold_ab/RESULT.md`、
  `Backup/cgc_logs/spac_cold_ab/RUN_NOTE.md`。
- 慣例：`agent_harness/CONVENTIONS.md` **B29**（條件必須可量測）、**B30**（事件時間戳）、
  **B31**（平行編輯 race）、**B32**（「沒有儀器」是工具的推論）。
- 同 repo 的 decode 側：skill `cgc-decode-attribution`。
