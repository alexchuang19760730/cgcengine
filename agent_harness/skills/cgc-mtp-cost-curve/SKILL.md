---
name: cgc-mtp-cost-curve
description: 在 flashkv-devserver（llama.cpp CGC fork，流式 MoE）上量測 speculative/MTP 的攤薄係數 m 與每步接受數 E，判斷「練 draft head 值不值得」。當使用者問「MTP 能不能加速／為什麼 MTP 虧／accept rate 要拉到多少才划算／cost(k)／攤薄」時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-mtp-cost-curve/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-20 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# 量 MTP 的 cost(k) 曲線

## 什麼時候用

- 「MTP 能不能給 2-3×？」「為什麼開了 MTP 反而變慢？」
- 「accept rate 拉到 85% 值得嗎？」「該先練 head 還是先改 batch 路徑？」
- 任何要把 `E`（每步接受幾個 token）與「verify 批次成本」分開的場合。

## 核心模型（先背起來）

```
S(k) = E(k) / cost(k)          cost(k) = 1 + m·k        （單位＝一個 plain decode step）
E    = (1 - a^(k+1)) / (1 - a)                          a = 單 draft token 接受率
硬上界：S <= (k+1) / (1 + m·k)      ⇒ m=1 時任何 accept 都救不了（S<=1）
```

**`m` 是唯一決定「練 head 有沒有用」的數字**：accept 拉滿也只是把 `E` 推到 `k+1`，
成本側照樣線性漲。所以結論的順序永遠是：先量 `m`，再談 accept。

## 怎麼量（工具已存在，別重寫）

```sh
python3 scripts/check/spec_cost_curve.py --self-test          # 26 項，零 GPU，先跑
python3 scripts/check/spec_cost_curve.py --profile prefill250 \
    --ks 0,1,2,3,5,7 --rounds 3 --prompt 512 --gen 128 --depth 0 \
    --json Backup/phase_decomp/spec_cost_curve_<tag>.json
# 要判定「成本是不是 expert 流送造成的」，加第二軸：
#   --budgets 8,6,4   ⇒ per-layer slots 143/107/71，在固定 k 下移動 MiB/step
```

`E` 與 `cost` 都是**直接讀數，不是擬合**：

| 量 | 來源 |
|---|---|
| `E = n_gen / rounds` | `LLAMA_BENCH_SPEC_DBG=1` ⇒ 每輪一行 `SPECDBG round: n_done=.. draft=..`（`llama-bench.cpp:2631`） |
| 有效 draft 深度 `k_eff` | 同一行 `draft=` 的平均 —— **不是** `--spec-draft-n-max`（MTP 模組會自己截斷） |
| verify 專家並集 | 快取 teardown 無條件印 `verify: calls=.. union=..`（`llama-expert-cache.cpp:2529`） |

warmup 的 generation 走非 spec 的 `test_gen(ctx,1)`（`llama-bench.cpp:2926`），所以輪數不會被 warmup 污染。
只有 `m` 是擬合的：最小二乘過原點，`cost-1 = m·k_eff`。

## 鐵律

1. **`--ks` 必須含 0**。一次 llama-bench 呼叫只能有一個 `--spec-draft-n-max` ⇒ k 曲線得分次呼叫；
   每個比值都用**同一輪自己的 k=0 基準**（工具會自動 ABBA 交錯、取輪內比值中位）。
   沒有 k=0 的曲線不能配對，工具會直接拒跑。
2. **絕對 t/s 通常不可引用**。18 支連續 bench 會把機器推到 HEAVY（基準臂自己掉 21%）。
   可引用的是輪內比值；報告要看 `clean` 欄與 `thermal(launch→worst)`。工具**還沒有 `--cooldown`**，
   若要乾淨絕對值，得自己分批跑並等 NOMINAL。
3. **`rc < 0`（訊號死亡）≠ 慢**。隔壁 session 的清場常以 `pkill -f 'llama-server|llama-bench'`
   形式出現（09-18 基準臂就這樣被 SIGKILL，rc=-9）。被殺的 run JSON 截斷、輪數偏短，
   混進中位會靜默偏掉曲線 ⇒ 工具會標記＋重試（預設 2 次）＋重試都死就排除。
   **assert／OOM 不重試** —— 那是「此 k 不可行」的結論。
4. **起跑前 preflight**（工具內建）：查 `llama-bench/llama-server/run_server.sh/http_duo.py/...`
   等競爭行程；有就 abort（`--force` 才硬上）。不要用 `pgrep llama` 自己判斷 —— 09-18 的競爭者
   是 `http_duo.py`，argv 裡沒有 `llama` 字串。
5. **union/call 是同一份資料裡最值錢的副產品**。它回答「verify 是不是真的把 draft token 的專家
   併成一個並集」：並集隨 k 明顯成長（實測 `9.36 + 3.12·k_eff`，單 token 一步＝8）＝ remap 有做；
   **恆為 8.00 ＝ 量到的是 exact-path fallback，那一輪整個作廢**（09-17 曾發生，09-18 已排除）。
   同時看 `verify-strict: zero_mapped_selected`，非 0 代表答案被靜默讀錯。

## 判讀

- `m >= 0.6`：幾乎沒攤薄 ⇒ 練 head 白花錢，去查 batch／並集路徑。
- `m <= 0.25`：攤薄良好 ⇒ accept 才是槓桿，值得練。
- 中間值：兩者都要。用 `S <= (k+1)/(1+mk)` 直接給天花板上界（實測 m=0.474 ⇒ a=0.92 只到 ~1.47×）。
- `E` 若隨 k 幾乎不動 ⇒ 加深度買不到 token，先查為什麼 `a` 這麼低（partial-restore 輪數也是線索）。

## 副產品：從 union 斜率反推 `p_route`，並估算「RSL 類」架構

`union(k_eff)` 的斜率就是「每個 draft token 平均帶來幾個**新**專家」（實測 3.61／滿 8）⇒
單專家邊際重疊率 `q = 1 − 斜率/8`（實測 0.55）。這是 `p_route` 的免訓練估計，不用去 dump 路由。

**但拿它估「draft 的 top-8 必須 ⊆ 已付費並集」（RSL-MTP）時會得到 0.70–0.97×、無一格 ≥1.0：**
`P(整個 top-8 都已在集合)` ≈ `q^8` = **0.008**（不是「略降」，是幾乎全砍）；放寬成「允許 ≤b 個新專家」
後 `Binomial(8, 1−q)` 顯示**接受率掉得永遠比成本快**（b=3：省 34% union／只留 47.5% 接受）。
而且「邊際成本 ∝ union」這個前提用兩點解 `(S_exp, c)` 會得到 **c<0**（四組配對全中）⇒ 否決。
⇒ **別再用 union 當槓桿去設計 draft 架構。** 詳見 `docs/RSL_MTP_GAIN_ESTIMATE_2026-09-18.md`。

## 每支 run 的 stderr 尾巴：成本主項就在那裡（先看這裡再決定怎麼優化）

`-p/-n/-d` 跑完後，teardown 會印出四行，**它們比 `m` 本身更能指出該修哪裡**：

```sh
grep -hE "read shape|decode/pool|layers_distinct_over_slots|miss attribution" \
  <workdir>/spec_cost_k*_r*_p*_n*_d*.stderr.log
```

1. `layers_distinct_over_slots=N` + `worst=layer L distinct=D slots=S` —— **N 層的 working set 超過
   該層 slot 配額**。實測 k=0..7：N = 0→4→15→17，`D/S` = 143/143→219/143。
   ⚠ **`k=0`（無 MTP）就已經 `143/143` 頂到配額 ⇒ 加任何 draft token 都立刻越界**，這解釋了為什麼
   `k: 0→1` 的跳變遠大於 union 計數的增長。
2. `hit%`：92.6%（k=0）→ ~50%（k≥1）。**崩點發生在 distinct 越界那一刻**，不是隨 k 漸進。
3. `read shape: bytes=`、除以步數 ⇒ `MiB/step`（實測 7.9 → 67）。回歸 `ms/step` 對 `MiB/step`：
   `ms = 89 + 3.35·MiB`（r²=0.70，等效 ~298 MiB/s）。⚠ **`MiB/step` 與 `k` 完全共線** ⇒ 這條回歸
   不能單獨定因；真正的機制證據是第 1 點那條**整數計數器**。
4. ⇒ ~~成本主項是 residency thrash~~ **此規則已被否決，見下方「第二軸」**。
   `layers_distinct_over_slots` 從 0 變非零只證明**有 thrash**，不證明**時間是它造成的**。

## 第二軸：pool budget（判定 thrash 到底值多少錢）

`--budgets 8,6,4` 會把 `-expert-cache <bytes>` 寫進 argv（per-layer slots 實測 143／107／71）。
**它是唯一能在 k 固定的情況下移動 `MiB/step` 的旋鈕**，用來打破上一條的共線。

09-18 結果（`prefill250 -p 512 -n 128 -d 0`，36 支）：

- 同 k 跨池的 **bytes→時間彈性只有 0.10–0.17**（k=1/3 甚至 −0.38）：
  k=0 時 bytes 4.68× 而時間只 1.25×；k=1 時 bytes 2.07× 而時間 **0.76×**。
- `k_eff` 單變數解釋 **84%** 的 `ms/step` 變異（59 ms/token）；bytes 只有 65%，
  且放進同一模型後 bytes 係數掉 3 倍（2.043 → 0.693 ms/MiB）⇒ bytes 大致是多餘的。
- `k_eff`（0.842）> `k`（配置的窗口，0.784）⇒ 成本跟著「真的 draft 出幾個 token」。
- draft 側只佔 union **3.0%** ⇒ 那 46–59 ms/token 不是 draft 自己去抓權重。
- ⇒ **`m` 的機械解釋是「每個 draft token 的固定代價」，修 residency 的上界只剩 ~15%**
  （超額 +269 ms 中 bytes 只解釋 ~40 ms；用 `8.5^0.145` 交叉驗算同一數字）。

⇒ **thrash／per-layer 配額／prefetch 開關降為次要。** 新的並列第一：

1. **`CGC_P_ROUTE=1`**（`llama-context.cpp:5929`、`llama-expert-cache.cpp:1799`）：探針**已在樹裡**，
   在 verify 步驟（`n_tokens>1`）直接印 `RSL p_route: ... i=1 X% i=2 ...` ＝
   `P(top8_{t+j} ⊆ top8_t)`，**不需要額外 forward**。跑一支 `--spec-draft-n-max 7` 就有答案，
   不必再用 union 斜率反推。
2. **n-gram draft**：同一個 k（verify 批次一樣是 k+1）但沒有 draft 前向。
   `m` 崩 ⇒ 成本是 draft 前向；`m` 不動 ⇒ 成本在 verify 的**每 token 路徑**（T=k+1 太小沒攤薄）。

**跑双轴前先修的三件事**（本輪全部踩到）：
① **run 順序與 budget 完全混淆** —— 每個 (round,k) 內永遠是 8→6→4，必須輪換／拉丁方；
② **漂移 −1.93 ms/run**（前 12 支 268 ms、後 12 支 188 ms）而熱態同時變壞 ⇒ 是 page-cache 預熱
   不是熱態，加 `idx` 協變量驗過係數不動（r² +0.001）；
③ 同格跨回合 max/min **1.03–1.74×** ⇒ 絕對 t/s 一律不可引用，只引用回合內比值與計數器。

**必查的環境事實**：`CGC_NO_PREFETCH=1` 是 `run_server.sh:2029` 的預設 ⇒ 上述全部是 prefetch 關閉下
量到的。要 A/B prefetch 就用 `CGC_PREFETCH_SRC=hist` 並 unset `CGC_NO_PREFETCH`（注意
`llama-expert-cache.cpp:1082` 記載的 deadlock class，且順序上排在 thrash 修好之後）。

## 別做

- **dynamic-k（自適應深度）**：用三輪資料算 oracle（每輪挑最好的 k）只得到 mean 1.035×／median
  1.068×，而這個上界還被噪音膨脹（同一 k=3 配置 r1=7.32 vs r2=11.49 t/s，1.57×）⇒ 真上界更低。
- **n-gram draft 的價值是診斷不是提速**：`--spec-type ngram-simple|map-k|map-k4v|mod|cache`
  （`common.h:177-181`）去掉 k 次 draft 前向 ⇒ 看 `m` 掉多少就知道 cost 裡有多少是 draft 前向。
  它省不掉 expert traffic：實測 draft 只佔 ~1%（`union=512` vs verify `union=47122` @ k=1）。

## 已知未解（別當成結論引用）

服務路徑（prod25）量到 MTP **+31%**、m≈0.21；llama-bench（prefill250）量到 **0.64–0.74×**、m≈0.474。
混淆項（profile／儀器／draft 深度／batch）都沒驗證，且 **prod25 在 llama-bench 上量不到**（零 GPU 可見）。
要解這個落差，用 `http_duo.py`（它保留下來就是為了研究儀器間差異）在服務路徑跑同一條曲線。

## 落盤

讀數要內嵌 `docs/*.md`：原始 json 放 `Backup/`，而 `Backup/` 被 `.gitignore:396` 排除 ⇒ 不能引用。
