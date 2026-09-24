# 任務 5（執行側）：平行跑環境被引擎自己的閘門擋下，而 4 GiB / prod25 baseline 建立完成

**日期** 2026-09-23 07:11–07:22 · 依 `docs/NEXT_ACTIONS_2026-09-23.md` 任務 5（【執行 Agent】設定 4GB cache
平行跑環境 + 建立新 baseline）· 載體 Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X · profile **prod25**
· pool **4 GiB**（`CGC_SERVER_EXPERT_CACHE_BYTES=4294967296`）· engine `server=054fb22f04a0`
`metal=128b6048870f`（與 04:14 那批五個指紋全部相同）· 產物 `Backup/t5_parallel/` ·
driver `Backup/phase_decomp/t5_parallel_baseline.py`

---

## 0. 一句話

**平行跑不是「還沒試」，是被 16 GiB 的實體邊界加上一份**設計好的**單行程閘門擋下來的**：第一支 4 GiB
server 一起來，第二支就得到 `error: 仍有 1 支 llama 行程，繼續啟動極可能 GPU OOM (ret=-3)`，而引擎自己
在啟動前就把算術印出來了 —— **一支** 4 GiB server 的靜態需求是 **17126 MiB vs 實體 16384 MiB**
（`OVERSUBSCRIBED by 742 MiB`），兩支是 **35.9 GB**。所以任務書的 `0.79 × 2 = 1.58` 裡，`0.79` 是真的、
**`× 2` 這一項不存在**：平行在可允許的配置下沒有第二個行程可以放。

同時（任務 5 的另一半）**4 GiB / prod25 的 baseline 建立完成，而且它是這條線上最可重現的一個臂**：
五次啟動的**每一個引擎側量都逐位元相同**（`accept 0.37908`、`mean_len 2.14`、`misses 102354`、
`evictions 102170`、`io_bytes 115747282944`、`slots 71`、`resident 3310.43 MiB`），只有**時間**在動，
而那 9.3% 的變動與 `thermal` 章**單調對齊** ⇒ baseline 應引用 NOMINAL 的那兩支：**8.30 t/s**。

---

## 1. 平行跑：逐字的拒絕，與兩層各自獨立的理由

### 1.1 實測

相位 B 時 S1（4 GiB, port 8080）在服務中，用完全相同的 env 對 port 8081 起第二支：

```
error: 仍有 1 支 llama 行程，繼續啟動極可能 GPU OOM (ret=-3)
```

失敗發生在**載入模型之前**（`admitted=false`，沒有任何位元組被配置）。對照組：S1 起跑時的閘門行是

```
[guard] memory_mode=dev class=full-mtp phys=16GB free=73% other_llama_servers=0
        req_phys=0GB req_free=40% req_other=0 admits=yes
```

`free=73% >= 40%` 通過、`phys=16 >= 0` 通過 —— **唯一被擋的是 `other_llama_servers`**。

### 1.2 兩層機制（都不是「資源暫時不夠」）

1. **閘門層（設計）**：`cgc_memory_guard_req()` 對**每一個** class 都回 `req_other=0`
   （`full-mtp-known-profile: 0 35 0`、`full-mtp: 0 40 0`、`fallback-mtp: 0 20 0`、`baseline: 0 15 0`；
   run_server.sh:582-591）。也就是說**這台機器不放行任何第二支 llama 行程，與剩餘記憶體無關**。
   外面還有一道 `[防護 1]` preflight STALE 硬檢查（:759-794）會先擋，所以它不是靠某個 agent 記得。
   ⇒ 任何「平行跑兩個」的計畫都要先改這個共用常數，而任務書同時寫著「不要偷偷放寬共用常數」。
2. **物理層（算術）**：引擎在載入前自己印

   ```
   [budget] 靜態需求 17126 MiB（model resident 13030 + pool 4096） vs 實體 16384 MiB
   [budget] OVERSUBSCRIBED by 742 MiB：這個啟動只能在 macOS 記憶體壓縮 + swap 之上執行。
   ```

   `load_mode=none`（= `--no-mmap`）⇒ 模型是**匿名頁、行程之間不可共享**；兩支就是
   `2 × 13.66 + 2 × 4.29 = 35.9 GB`。唯一的共享路徑是 `--load-mode mmap`，而它在本機已被量過三次
   （`4096/5120/6144` 全部 0/3，死在載入後第一個 `ggml_metal_synchronize`，status 5 OOM），
   引擎自己也把它標成「**不是**槓桿，不要再試」。

### 1.3 判定：任務 5 的平行前提不成立

`0.79 × 2 = 1.58` 這個算術只有在「第二支放得進去」時成立。實測是第二支連啟動都不允許，所以可拿到的
是 `0.79 × 1 = 0.79` —— **把池降到 4 GiB 去換平行，在此機器上是純虧**。要真的拿到 1.58，得先動其中一件
（放寬 `req_other`、或讓兩支共享模型頁），兩件都在我的權限與風險邊界之外，所以我停在「把拒絕與算術
留下來」而不是自己去繞過它（繞過的形狀是 35.9 GB 的匿名頁疊在 16 GiB + 5.3 GB swap 上，而那正是
這個 repo 記過的 kernel panic 根因）。

---

## 2. 4 GiB / prod25 baseline（任務 5 的另一半）

五次啟動、同一個 env、只有 port 與 tag 不同。`--rounds 5 --warmup 1 --n-predict 160`（`decode_bench.py`，
交付 regime）。**引擎側與客戶端側分開列**，因為它們的答案不一樣：

| tag | t/s | t/s min | step ms | accept | mean_len | hit% | misses | evictions | compulsory | slots | resident |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rep1 | 8.29 | 8.22 | 258.1 | 0.37908 | 2.14 | 64.1 | 102354 | 102170 | 7894 | 71 | 3310.43 |
| b2 | **8.31** | 8.10 | 257.5 | 0.37908 | 2.14 | 64.1 | 102354 | 102170 | 7894 | 71 | 3310.43 |
| b3 | 8.17 | 7.92 | 261.9 | 0.37908 | 2.14 | 64.1 | 102354 | 102170 | 7894 | 71 | 3310.43 |
| b4 | 8.14 | 7.92 | 262.9 | 0.37908 | 2.14 | 64.1 | 102354 | 102170 | 7894 | 71 | 3310.43 |
| b5 | 7.60 | 7.42 | 281.6 | 0.37908 | 2.14 | 64.1 | 102354 | 102170 | 7894 | 71 | 3310.43 |

- **median 8.17 t/s（262 ms/step），範圍 7.60–8.31（9.3%）**
- 客戶端的每一欄都逐位元相同，包括 `io_bytes = 115747282944`（107.8 GiB）與 `io_jobs = 291567`
- 只有時間在動，而它與 `thermal` 章對齊：`NOMINAL/NOMINAL/MODERATE/MODERATE/HEAVY`
  → `8.29 / 8.31 / 8.17 / 8.14 / 7.60`。**單調**。
- ⇒ **要引用的 baseline 是 NOMINAL 那兩支：8.29 與 8.31（median 8.30）**；另外三支是熱的，
  這正是「不要在熱機上跑」那條規則本身。

### 2.1 跨 session 重現：這個臂是這條線上最穩的一個

把今天**全部** prod25 的池臂放在一起（early 半夜四次 + 這次五次），每個引擎側量仍然逐位元相同：

| 池 | 逐支 t/s | median | 極差 |
|---|---|---:|---:|
| **4 GiB**（n=9） | 7.60 · 8.14 · 8.17 · **8.19 · 8.23** · 8.29 · 8.31 · 8.35 · **8.50** | **8.23** | **1.118×** |
| 8 GiB（n=5） | **7.35** · **10.69** · 10.78 · 10.84 | 10.69 | 1.47× |

`misses / evictions / slots / resident` 兩組各自**九支、五支都逐位元相同**（4 GiB：102354 / 102170 /
71 / 3310.43；8 GiB：39948 / 39836 / 143 / 6430.62）。所以：

- **同池跨啟動的差不是工作量差，是每次事件的時間差** —— 這與 `docs/K3_SWING_ANALYSIS` 的結論一致。
- 8 GiB 那 1.47× **由單一離群值驅動**（`7.35`，是 04:06 那支 purge 後的第一支）；其餘四支只在
  **1.4%** 內。上一輪我把 8 GiB 寫成「站不住」，這句話要**修正**：它是**通常很穩、偶爾掉一次**，
  而 4 GiB 是**每次都慢 23%、但每次都在同一個 8 附近**。
- 池的價格（prod25，median vs median）：**8.23 / 10.69 = −23.0%**。與 `CACHE_SIZE_AB` 的 −22.6% 一致。

---

## 3. 這條線的下一步（不是這個 driver）

平行被擋掉之後，「總吞吐量」這件事只剩兩個不含第二個行程的做法：

1. **同一個 server 內並行請求** —— 不必第二支 server、不必第二份模型。這一格從未被量過，而它與任務 5
   想買的東西是同一件（`server` 目前 `-np 1`）。這是我建議的下一步。
2. **把 4 GiB 的 −23% 買回來**（`capacity` 佔這個臂 miss 的 **92.3%**、`compulsory` 只有 7.7%
   ⇒ 在這個臂上「多幾個槽」幾乎一對一換位元組，與 09-20 那份「74.4% capacity」的普查同向）。
   這才是真的把 4 GiB 變成可用工作區間的路。

---

## 4. 過程中修掉的三個工具缺陷（兩個會產出看起來很正常的錯數字）

1. **profile 用錯**：driver v1 用 `CGC_SERVER_PROFILE=prefill250`（那是 `decode_sweep.py` 的*預設*值），
   而這條線的交付 regime 是 **prod25**（`decode_sweep --profile prod25`；`CACHE_SIZE_AB` 的抬頭也寫著）。
   後果具體：prefill250 的取樣不是貪婪（每輪 `predicted_n = 99/96/96/96/101/160`、三個答案 md5），
   prod25 是（每一輪都 108、一個 md5）。它量出 **6.77 t/s** —— 一個看起來很合理的 baseline，
   只是屬於**另一個臂**（`misses 36613 / hit 47.2%` vs prod25 的 `102354 / 64.1%`）。
2. **counter join 用 row tag 當 key**：teardown 後才寫的那一列（`rep1-final`）去找一個不存在的 ctrl log，
   於是**靜默地沒有計數器** —— 長得跟「這一輪沒有資料」一樣，而不是「查錯了檔」。
   現在計數器是獨立的一條流，用 `logtag`（真的啟動的那一支）當 key。
3. **`srv_log` 切字**：`[log]   <path>（tail -f 同路徑）` 只按空白切，路徑尾巴黏著 `（tail` ⇒
   teardown 找不到檔案，於是不等 `final stats` 就 `kill -9`。修好後 5/5 都拿到完整 teardown 區塊。
   （同樣的形狀 `decode_sweep.py:LOG_LINE_RE` 是對的，它排除了 `（`；錯的是我重寫的那一份。）

---

## 5. 誠實邊界

- **量測閘與啟動器仍然不一致**：`server_window` 這五次判 QUIET（reclaimable 8097–8308 MB），而它自己的
  `NEED_MB=8000` 就在那條線上 —— 兩支在 8000 旁邊（8097/8229/8087/8278/8308），門檻只要再高一點就會翻面。
  我沒有動 `NEED_MB`。
- 8 GiB 的兩支早期讀數（`10.69/10.78`）是 `decode_sweep --profile prod25` 出來的，與這次同一顆 binary
  （五個指紋逐位元相同），所以 §2.1 的跨池比較是同語言的；**但跨 session 的絕對值仍不該混用**，
  上面每一列都帶自己的啟動熱章。
- 相位 B 的失敗是**閘門拒絕**，所以「兩支同時跑會不會互相干擾」這個問題**沒有被回答** ——
  不是「干擾很小」，是「量不到」。要回答它必須先改 §1.2 那一層。
- `accept` 在此臂是常數（0.37908, n=5），所以 4 GiB 這一臂的工作量是**確定的**；
  `prefill250` 那一臂不是（§4.1）。
- 未 commit。產物 `Backup/t5_parallel/`（`baseline_prod25.json` + `bench_*.json` + `launches.jsonl`），
  腳本 `Backup/phase_decomp/t5_parallel_baseline.py`；**零殘留行程**。
