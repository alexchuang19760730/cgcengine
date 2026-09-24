# Build provenance 與 dylib hazard — 2026-09-21 11:5x

回答兩個指令：①「把無關改動 stash／commit」②「提醒他們 `cmake --build` 會蓋掉別條線的 dylib」。

## 0. 裁決（先講結論）

| 指令 | 做不做 | 理由 |
|---|---|---|
| 現在 `git stash` | **不做** | 沒必要（S0-2 已跑完且全同 build）＋ 會把別條線的在編 WIP 從工作樹抽走 ＋ 「無關」那批跟 S1 的處置長在**同一個檔案**裡，切不乾淨 |
| 現在 `git commit` | **不做** | pre-commit 檢查 8 會擋（binary 比 source 舊）；繞過它等於把「binary 與原始碼不符」寫進版控 |
| 提醒別條線 | **做**（本文 §5） | 閘門存在且有效，但**後門也存在**，而他們下一步 Stage 2-a 一定要重建 |

**真正該做的是換順序**：不是「先清樹再做事」，而是 **先做事 → 要重建的那一刻才清樹 → 重建 → commit**。
因為 S0-2 的三個旋鈕都已經在現有 binary 裡（§1.3），今天剩下的活**完全不需要重建** ⇒ 現在清樹沒有任何收益，只有風險。

---

## 1. 目前 binary 到底是什麼（全部實測，非推測）

### 1.1 身分

```
git HEAD : b54b83e809346f9b9b91a27e717033d1d0435cdd
指紋（委派 scripts/check/decode_sweep.py:build_fingerprint()，8 鍵）
  libggml            8f4abce76052
  libggml-base       fea88a48e636
  libggml-blas       52de23d5c405
  libggml-cpu        70ebbd7c709a
  libggml-metal      544b6c94bc7a
  libllama           7376c9769bd6
  libllama-common    6e5699e0b156
  server             054fb22f04a0
```

產物 mtime：`libggml-base` = 09-20 **21:42:55**；
`libllama` / `libllama-common` / `libllama-server-impl` / `llama-server` = 09-20 **23:05:06**。

### 1.2 哪些改動編進去了、哪些沒有（用 mtime vs 23:05 判）

| 檔案 | 改動時間 | 在 binary 裡 | 內容（diff 標記） |
|---|---|---|---|
| `common/speculative.{cpp,h}` | 09-20 21:04 | ✅ 在 | M4 draft-distribution（+67） |
| `tools/server/server-context.cpp` | 09-20 21:04 | ✅ 在 | +6 |
| `ggml/src/ggml-backend.cpp` | 09-20 21:42:50 | ✅ 在 | hook-sub per layer（+115） |
| `src/llama-expert-cache.h` | 09-20 22:28 | ✅ 在 | M-F5 ADVISE_ASYNC ＋ L2 ring（+74） |
| `src/llama-expert-cache.cpp` | 09-20 22:36 | ✅ 在 | NO_RDADVISE／NO_MERGE ＋ L3 M0（+380） |
| **`src/llama-context.cpp`** | **09-21 00:08** | ❌ **不在** | **+278**：L3 M0 ×3、L2 ×2、**hooksub-path fix／hooksub reachable on the delivery path ×3** |

⚠️ **由此得到一個之前沒人寫下的事實**：目前這顆 binary 的 **hooksub 是「半接」的**
—— `ggml-backend` 側在，`llama-context` 側那三條「讓 hooksub 在交付路徑上可達」的修正**不在**。
任何讀逐層 hooksub 輸出的人（本線的 KIND×OP 儀器就是）都要記這筆。

### 1.3 三個 S0-2 旋鈕已在 binary 裡 ⇒ 今天不必重建

`strings src/llama.cpp/build/bin/libllama.0.0.279.dylib` 三個都命中：
`LLAMA_EXPERT_CACHE_NO_RDADVISE`、`LLAMA_EXPERT_CACHE_NO_MERGE`、`LLAMA_EXPERT_CACHE_ADVISE_ASYNC`
（源碼在 `llama-expert-cache.cpp:3495/3497/3928`，mtime 22:36 < 23:05 ✓）。
⇒ 與 §EN-375 一致：**S0-2 十臂全部同 build**，且下一輪 Stage 0 的其餘項目也不需要重建。

### 1.4 但 provenance 沒有被記錄下來

`agent_harness/engine_loop/build.json` **不存在** ⇒ `rebuild.sh --check` 目前回
「沒有任何已記錄的指紋可比」。也就是說上面那 8 鍵只活在本文裡。要讓它可機檢：

```sh
# 把現在這一批的指紋寫成 build.json（不建置；之後 rebuild.sh --check 才有得比）
cd /Users/alexchuang/Documents/flashkv-devserver
python3 -c 'import importlib.util,json,datetime,sys
spec=importlib.util.spec_from_file_location("ds","scripts/check/decode_sweep.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
json.dump({"fingerprint":m.build_fingerprint(),
           "recorded_at":datetime.datetime.now().isoformat(timespec="seconds"),
           "git_head":"b54b83e809346f9b9b91a27e717033d1d0435cdd",
           "note":"recorded without building -- binary mtime 2026-09-20 23:05:06"},
          open("agent_harness/engine_loop/build.json","w"), ensure_ascii=False, indent=2)'
bash agent_harness/engine_loop/runners/rebuild.sh --check
```
（這會在 `agent_harness/` 下新建一個檔 —— 那一層歸別條線，所以我沒替他們建，要建請用上面這段。）

---

## 2. 為什麼現在不該 `git stash`

1. **沒有收益**：清樹的唯一目的是讓「下一次建置」的產物可歸因。而下一件需要建置的事是
   Stage 2-a（要寫 C++），在那之前不會重建。§EN-375 已確認本輪所有比值同 build。
2. **會傷到人**：stash 是把檔案**從工作樹抽掉**。另一條 session 正在編這批檔案
   （`llama-context.cpp` 00:08 還在動），抽掉等於把他們的地板抽走，而 stash 的
   復原（pop）不是他們會預期發生的動作。
3. **切不乾淨**：想 stash 的「無關」批次（L2 ring、L3 M0、M4 parity）與 **S1 的處置 M-F5
   住在同一個檔案裡** —— `llama-expert-cache.h` 同時有 `[CGC 2026-09-20 M-F5]` 與
   `[CGC 2026-09-20 L2]`；`llama-context.cpp` 同時有 L3 M0、L2 與 hooksub 修正。
   用 `git stash push <file>` 會連 S1 的旋鈕一起拿掉；用 `git stash push -p` 在 872 行／7 檔
   上盲挑 hunk，是拿別人的 WIP 玩猜謎。
   ⇒ **真要切，只能由作者本人切。**

## 3. 為什麼現在 `git commit` 也不便宜

`pre-commit` 是 **common-dir hook**（所有 worktree 共用那一顆）＝ `scripts/check_build_tracked.sh`。

- **檢查 8（原始碼↔產物同步）會擋**：改了 llama 原始碼的 commit 必須 (a) build/bin 產物一起
  staged、(b) **binary mtime 不得老於 staged 原始碼**。這裡 binary = 23:05，而
  `llama-context.cpp` = 00:08 ⇒ **(b) 失敗**。
  `ALLOW_STALE_BIN=1` 可以繞，但那正是這支 hook 存在要防的疏失 D
  （production 載入與原始碼不符的 dylib）。**別為了「清樹」這個形式目標去開這個洞。**
- **檢查 11（replay benchmark regression）**：3 個 profile × 12 指標，跟上一 commit 比。
  ⇒ commit 很可能**自己起 server 跑一輪量測**，佔 GPU、且跟別條線搶。這不是順手的事。

---

## 4. Stage 2-a 真的要重建時，的安全順序

```sh
cd /Users/alexchuang/Documents/flashkv-devserver

# 1) 問閘門（不建置、不寫檔）。要看到「閘門 : 通過」
bash agent_harness/engine_loop/runners/rebuild.sh --dry-run

# 2) 通過才建；建完它會寫 build.json（provenance 從此可機檢）
bash agent_harness/engine_loop/runners/rebuild.sh -j 8

# 3) 這時才 commit：binary 比 source 新 ⇒ 檢查 8 會過
git add src/llama.cpp/src src/llama.cpp/common src/llama.cpp/ggml \
        src/llama.cpp/tools src/llama.cpp/build/bin
git commit -m "..."

# 4) 若想完全不碰別人的工作樹：開一顆乾淨 worktree 編（注意：乾淨樹沒有 M-F5，
#    ADVISE_ASYNC 旋鈕不在那裡 ⇒ 只能拿來跑基線，不能跑 S1）
git worktree add ../cgc-s2a-clean b54b83e80
```

---

## 5. 給另一條 session 的提醒（dylib hazard）

**規則：不要直接 `cmake --build src/llama.cpp/build`。** 建置產物納入版控，所以這一行是
對「機器上所有正在跑的實驗」的一次寫入 —— 2026-09-17 實測過：別條線的 server 明明在聽 8080，
build 還是跑了，他們那一輪 A/B **橫跨兩個 build**。

### 5.1 閘門已經存在，而且是真的會 abort

`agent_harness/engine_loop/runners/rebuild.sh`（不是只印出來，是 `exit 1`；2026-09-17 13:18
真的擋下一次）。它擋：

- port 8080 有 listener
- 量測行程：`run_ids_dst_capture|decode_sweep|knifeedge_matrix|ab_interleave|m123_oracle_gate|refill_idle_sweep`
- **殘留引擎**：候選由 `pgrep -f` 蒐集、**身分由 `basename(ps -o comm=)` 判定**
  （`llama-server llama-cli llama-bench llama-simple llama-perplexity`）⇒ 命令列提到名字的
  wrapper 不會被誤判，真的引擎跑不掉
- 另一個 `cmake`/`ninja` 正在跑

### 5.2 ⚠️ 但有兩個後門完全沒過閘門

```
scripts/build_fork_llama.sh:224   cmake -B "$BUILD_DIR" ...
scripts/build_fork_llama.sh:238   cmake --build "$BUILD_DIR" -j"$JOBS"
scripts/build_native_cgc_llama.sh （同樣有 cmake）
```
**這三行就是事故現場。** 要走就走 `rebuild.sh`。

### 5.3 閘門名單是舊的（實務上仍擋得住，但要知道）

`meas` 那段名單不含 `perjob_s02.py`、`harness.py`、`profile_duo`、`prod_profile`、`http_duo`、
`cb_headroom_probe`。不過這些 driver 都會 spawn `llama-server`／`llama-bench`，
**會被第二段的執行檔 basename 抓到** ⇒ 實務上仍擋得住；
唯一會漏的是「driver 已啟動、引擎尚未 spawn」的那個空檔。

### 5.4 重建的代價要先講清楚

重建後，**23:05 那批 binary 的所有數字全部變成不可與新 build 直比**，包含：
- §EN-375 的 S0-2 十臂（`NO_MERGE` +0.018% 那個強結論）
- §EN-373 我從 01:31 np4 log 算出來的聚合 3.71 t/s
- 今天任何一輪 prefill／decode 讀數

⇒ **先把還能用現有 binary 做的量測做完，再重建。** 這也是 §0 那張表把順序反過來的原因。

---

## 6. 給 operator 的一句話

「無關改動 stash／commit」這個動作我建議**改判為「待重建時執行」**，而不是現在做：
現在做沒有收益（不需要重建）、有實質風險（抽別人的 WIP、切不乾淨、commit 會被 pre-commit
檢查 8 擋或逼你開 `ALLOW_STALE_BIN` 的洞）。真正該先做的是 §1.4 把指紋記下來，
以及 §5 這份提醒送到別條線手上。
