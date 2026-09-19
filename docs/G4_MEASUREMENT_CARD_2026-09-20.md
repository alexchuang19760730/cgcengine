# G4 的待跑卡（2026-09-20 02:0x）—— 以及「量不出來」那句話的更正

作者：引擎歸因線。**這張卡的用途**：窗口一開就照著跑，不要重新推導。

---

## 0. 先更正一句話（我上一輪寫錯的）

上一輪我在 `targets.json` 的 `resolution` 裡寫「G4 的 −12.8% **NOT MEASURABLE**」。
**那句話用的統計量是錯的**：我算的是**跨 run** 的散佈（Ornith 的四台 server），
而「能不能分辨」取決於**單一 run 內、中位數的標準誤** `1.253·σ/√n`。

兩者不是同一件事：

| 比較方式 | 觀測到的 3SE | 能不能判 12.8% |
|---|---:|---|
| **兩台先後 server、各 ~45 步** | 16–25% | ❌ 不行（09-18 那次就是這個） |
| **一台 server、~180 暖步** | **8.0%** | ✅ 可以 |
| 一台 server、暖態後半 ~90 步 | **6.3%** | ✅ 可以 |

數字直接來自既有 log（`CGC-DECPROF` 的 `union_sum`，`layers=40` 的完整步）：

| log | ntok | n | union 中位 | σ | SE(中位) | 3SE |
|---|---:|---:|---:|---:|---:|---:|
| `llama_server_20260919_210013.log`（E2b capB，全段） | 4 | 179 | 132.13 ms | 37.69 | 3.53 | **8.0%** |
| 同上，暖態後半（step 482–946） | 4 | 89 | 131.85 ms | 20.95 | — | **6.3%** |
| `llama_server_20260918_043255.log`（Ornith ctl_fwd） | 1 | 48 | 89.35 ms | 26.15 | 4.73 | 15.9% |
| `llama_server_20260918_043439.log`（Ornith on_fwd） | 1 | 48 | 132.89 ms | 42.52 | 7.69 | 17.4% |

**⇒ 結論：G4 的 −12.8% 量得出來，但只能用「一個夠長的暖 run」量，不能用「兩台先後 server」。**
`resolution` 保留原文（那個錯誤留在紙上比較有用），以本節為準。

---

## 1. 前置檢查（**不做的話窗口一定白跑**）

1. **union 需要兩個旋鈕**：`CGC_DECODE_PROFILE=1` **與** `CGC_GPU_TIMING=1`。
   實測 2026-09-20：最近 13 份 server log **有 12 份完全沒有 DECPROF**
   ⇒ 配對集會是空的，然後被讀成「沒有效應」。
   檢查一行：
   ```sh
   python3 scripts/check/paired_union_ab.py --pick "$(date -v-30M +%s)"   # 逐份報 union 步 n/decprof 步 m 並指名缺哪個旋鈕
   ```
2. **機器必須真的空**（他的閘門 = 8080 無 listener ＋ 無別的 llama 行程 ＋ `vm_free_mb() ≥ 8000`）：
   ```sh
   python3 -c "import importlib.util as u; s=u.spec_from_file_location('h','scripts/check/decode_window_harness.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(round(m.vm_free_mb(),1),'MB')"
   ```
   ⚠️ 2026-09-20 01:5x 實測 **6498 MB**（active 5.16 GB ＋ wired 1.84 GB ＋ 壓縮 2.11 GB 不可回收）
   ⇒ **不是「等別人做完」就能解決的**，要機器端釋放記憶體，或把 `--need-mb` 降到 ~6500。
3. **不要用 `decode_sweep.py` 當配對工具**：它是**臂外層**（一個臂一台 server、跑完 N 輪才換臂）
   ⇒ 兩臂必然是先後兩台 server，落進上表第 1 列（3SE 16–25%）。
4. **不要用 `paired_ab.py` 收 union**：它驅動的是 **llama-bench**（`cell=decode -p 0 -n 128 -d 512`），
   不產生 `llama_server_*.log`；而且它**不記錄每 rep 的 log 路徑**。

---

## 2. 待跑卡 A（**G4 的下一個窗口，如果排到**）

目標：量「某個候選」對 `union_sum` 的效應，判準 3SE ≤ 8%。

> ⚠️ **修訂（2026-09-20 02:2x）—— 先讀這段，再讀下面的指令。**
> 1. **它會無閘門啟動 server。** `http_duo.py` **不在**走 `server_window` 閘門的 13 支工具之列
>    （`phase_split_ab`／`pool_curve`／`mtp_accept_ab` … 在，`http_duo`／`profile_duo`／
>    `decode_sweep` **不在**）。照這張卡原樣跑，等於在別人的 waiter 排隊時直接起 server ——
>    就是 §EN-278 那次事故。要跑就得自己包一層：
>    `python3 scripts/check/server_window.py wait --need-mb <F> -- python3 scripts/check/paired_union_runner.py run …`
> 2. **「量儀器底」這個用途已被取代，不要為它排窗口。** 佇列裡那一輪帶著
>    `CGC_DECODE_PROFILE=1` ＋ `CGC_GPU_TIMING=1`（launcher 自己印的 `extra=[...]`），
>    **它的 server log 就有 `union_sum`** ⇒ 用 `scripts/check/union_floor.py --newest 3` 直接讀。
>    本節留著是為了**要測一個候選**時用的，不是為了量底。
> 3. 現成的答案：E2b `llama_server_20260919_210013.log`（ntok=4、n=179）⇒ 中位 132.13 ms、
>    σ 37.69、**3SE 8.0%**（後半段 6.3%）—— **單 run 足以分辨 G4 的 12.8%**。

**工具已經有了**：`scripts/check/paired_union_runner.py`（2026-09-20 新增，與 `paired_union_ab.py`
同批）。它是**唯一**同時滿足「server 端」與「記錄每 rep 的 log 路徑」的執行器 ——
另外三支各缺一半：`paired_ab.py` 驅動 llama-bench（無 server log、也不記路徑）、
`decode_sweep.py` 是臂外層（兩臂必然是先後兩台 server）、`http_duo.py` 只有單臂（但它**記
`server_log`**，所以本工具用它當每個 rep 的引擎）。

設計上它**只當排程器**：每個 rep 呼叫 `http_duo.py --port auto`（⇒ 不撞別人的 8080），
**不重打啟動與 env 邏輯** —— `run_server.sh` 的 allowlist 只有一份，重打一份就是旋鈕靜默失效的來源。
`--self-test` 6 項（ABBA 排程、SE 公式、三條拒絕路徑、fixture 可配對步數），
另外 import 時就斷言 `http_duo.py` 真的在解析出來的路徑上。

```sh
# (1) 跑 ABBA 交錯配對：每對換執行序（A-B / B-A），單調漂移在 AB/BA 估計量裡抵消。
#     --a-env / --b-env 預設相同 ⇒ NULL 模式（儀器底）。要測候選就把它們設成兩個不同的旋鈕。
python3 scripts/check/paired_union_runner.py run \
  --pairs 8 --reps 4 --profile prod25 \
  --out Backup/phase_decomp/g4_null_manifest.json
#     ⚠️ 缺 CGC_DECODE_PROFILE 或 CGC_GPU_TIMING 時它會警告（那一臂的 log 不會有 union）。
#     ⚠️ 步數決定單 run 的精度：E2b 的經驗是 ~180 個 ntok=4 的完整步才到 3SE 8%
#        ⇒ 先把 --reps 加大（同一個 server 內續跑，成本只是時間），再考慮加對數。

# (2) 立刻確認這一跑真的有 union（不是空的）
python3 scripts/check/paired_union_ab.py --pick "$(date -v-20M +%s)"

# (3) 統計：每對一個 Δ、按執行序分組、AB/BA 扣漂移、3SE 對照 12.8%
python3 scripts/check/paired_union_runner.py eval \
  --manifest Backup/phase_decomp/g4_null_manifest.json
```

`eval` 會同時印三件事，缺一不可：**效應 (A−B)**、**漂移**（未交錯的設計會把它當成效應報出來 ——
Ornith fixture 上漂移是 **+28.12%** 而真效應只有 +5.43%）、以及**「欲達 3SE 需 n_pairs ≈ X」**。
**漂移若與效應同量級，這個 run 不足以下結論。**

**判準（事前寫下，免得事後挑）**：把這一跑的 `union_sum` 中位數與 E2b 的 **132.13 ms**
（`llama_server_20260919_210013.log`，ntok=4、n=179、3SE 8.0%）比。
比值 ≤ 0.872（−12.8%）且在 3SE 之外 ⇒ 候選成立；在 3SE 之內 ⇒ **NOT MEASURABLE**（不是「沒有變化」）。

⚠️ **跨 run 比較要認**：兩個 run 之間仍有殘留漂移（Ornith 的 AB/BA 量到 +9.95%）。
所以**任何跨 run 的結論都要在同一批裡帶一個對照臂**，或用 AB/BA 扣掉漂移。
**單一 run 的 3SE 只回答「這台機器在一個 run 內能多精確」，不回答「換一台 run 會不會整體位移」。**

---

## 3. 待跑卡 B（**G6 的窗口，它優先**）

G6 **不依賴 G4**（`union <= 38*mean_len` 的兩邊各自可量）⇒ 照他原本的計畫跑即可。
唯一要提醒他的一句：**他的 run 也過不了 8000 MB 的閘門**（實測 6498 MB，且零 llama 行程），
所以與其等他，不如先決定 `--need-mb` 要不要降、或先釋放記憶體。

---

## 4. 這張卡自己承認的兩個弱點

1. ~~真正乾淨的做法是 AB/BA，但那需要一個會記錄每 rep log 路徑的 server 端配對執行器，而它不存在。~~
   **2026-09-20 已補上**：`scripts/check/paired_union_runner.py`。它把「效應」與「漂移」分開報，
   所以 §4 的第 1 個弱點從「缺工具」降級成「樣本數」——`eval` 會直接印「欲達 3SE 需 n_pairs ≈ X」
   （Ornith fixture 反推是 ≈68 對，但那是短 run；單 run 拉的長度可以換掉對數，見 §0）。
2. §0 的表用的是**既有 log 的中位數**，不是同一批配對樣本；它的用途是**定出可分辨門檻的量級**，
   不是給出一個判決。
