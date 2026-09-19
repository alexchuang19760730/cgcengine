# 引擎凍結 ⊕ PLAIN_MATCH 重跑（2026-09-19）

## 為什麼

同一個探針、同一個 env、同一份 argv：

| 時間 | `off-plain`（MTP=0、不帶任何 equaliser）輸出 | `on` 輸出 |
|---|---|---|
| 02:57 | 262 字（與 `on` 相同 ⇒ PLAIN_MATCH **TRUE**） | 262 字 |
| 04:0x | 278 字（⇒ PLAIN_MATCH **FALSE**） | 262 字 |

兩次之間唯一的已知差異是 **libllama 在 03:39:45 被重建**，而被連結的舊產物（`0.0.275`/`0.0.277`）在 04:0x 時已經不在磁碟上。所以那不是「引擎變了」也不是「引擎沒變」，而是一個**無法事後二分的身分問題**。

`libllama.0.0.279.dylib` 是**版本號不是身分**：重建之後它還是叫 0.0.279。這份文件把 (原始碼 ↔ 產物) 變成一對被記錄、可複驗的東西，然後在那顆 build 上把 MTP on / off / on-exact 重跑一次。

---

## 1. 凍結機制：`scripts/check/engine_freeze.py`

```
freeze --tag X     寫一份 JSON：(原始碼 digest) ⊕ (產物 digest) ⊕ (決定產物的編譯旗標) ⊕ HEAD ⊕ 髒檔清單
verify --tag X     逐節複驗，把漂移歸因成「原始碼動了」還是「產物被換掉了」，而不是塌成「數字變了」
```

- 原始碼 digest 是 **sha256 over `git ls-files -co --exclude-standard -- src/llama.cpp`（排除 `build/`）**：tracked **加上** untracked-but-not-ignored。這棵樹一直帶著平行 session 的未提交編輯，所以純 `HEAD` 雜湊會描述一個**沒有任何人建置過的**原始碼狀態。
- 每個檔案餵入 `路徑 + 大小 + 內容`：改名、截斷、編輯都會移動 digest，而且彼此不可混淆。
- 產物逐一點名（`llama-server`、`libllama.0.0.279`、`libllama-common`、`libggml-metal`、`libggml-base`），缺一顆會顯示 `MISSING` 而不是從 digest 裡消失。
- 編譯旗標（`CMAKE_CXX_FLAGS=-DMTP_SUPPORT` 等）在凍結**裡面**：同樣的原始碼配不同旗標是不同的 build。

**它不主張什麼**：不做乾淨目錄的 hermetic 重建（這個 build dir 與平行 session 共用，而且它就是生產產物）。它主張的正好是驅動需要的東西 ——**這顆產物就是這份原始碼狀態的建置，而且你可以證明我沒有偷偷動掉任何一半** —— 而 `verify` 讓這個主張可被否證。

**已知的偏向（刻意留下的）**：原始碼 digest 是**過度涵蓋**的（整個 `src/llama.cpp` 扣掉 `build/`，含 docs/tests）。所以一個只改 `src/llama.cpp` 底下文件的編輯也會報 `source DRIFT`。這是有意的方向選擇 —— 寧可對**不影響產物的編輯**過度報警，也不要在任何**真正餵進編譯的輸入**上說 MATCH。而且不會因此塌成模糊：`verify` 是**分節**輸出的，`source DRIFT + artifacts MATCH` 自己就說明了是哪一半動了。反過來（`source MATCH + artifacts DRIFT`）則永遠是「有人換掉了產物」。

### 本次凍結

```
Backup/engine_freeze/pre_plainmatch_0919.json
source  sha256=9bf41903e7e69b5789d3f6ae  files=3381  bytes=163379581
head    98b44c8c6   dirty_sources=4
          src/llama.cpp/src/llama-context.cpp
          src/llama.cpp/src/llama-expert-cache.cpp
          src/llama.cpp/src/llama-expert-cache.h
          src/llama.cpp/tools/llama-bench/llama-bench.cpp
flags   Release / -DMTP_SUPPORT / LLAMA_BUILD_SERVER=ON
```

---

## 2. 可重現性探針（這是「可重現的 build」的證據）

先看時間鏈，證明產物不是過期的：原始碼 `03:39:07` → 目的檔 `03:39:35` → `libllama 03:39:45`。

然後**強制**重建：`touch` 那 4 個髒檔 → `cmake --build ... -j6`。

| | |
|---|---|
| 重編譯 TU | **164** 個（改到 `llama-expert-cache.h`，依賴它的全數重建） |
| 重新連結 | `libllama-common`、`libllama-server-impl`、`llama-server`、`libllama.0.0.279` |
| mtime 變化 | `03:39:45` → `08:32:59` |
| **5 顆產物的 sha256** | **全部逐位元不變** ⇒ `FROZEN STATE HOLDS` |

⇒ 兩件事同時成立且都是量出來的：**(a)** 產物就是這份原始碼的建置；**(b)** 這個 build 對這份原始碼狀態是**決定性的**（重編重鏈後位元相同），所以「同一顆 build」不是靠 mtime 猜的。

---

## 3. 這顆 build 同時通過 M1/M2/M3 閘門

| 項目 | 值 |
|---|---|
| 閘門 | `summary_anchor0919b.json`：**M1 9/9、M2 9/9、M3 9/9**，`comparable=True`、`config_diffs=[]` |
| build | `libllama md5[:16] = aab787412e572550` |
| freeze 的同一顆 | 同值（sha256 `1b7c6d1bae17ba9f4dd9d366`） |

也就是說：**閘門 PASS 的產物與下面量速度的產物是同一顆 binary**，而且這顆 binary 可從凍結的原始碼重建出來。這是這一輪真正新增的東西 —— 以前這兩句話只能靠「我記得我沒重建」來成立。

---

## 4. 凍結 build 上的 6 臂（鏡像順序）

命令：

```
plain_match_ab.py --order on,off,on-exact,on-exact,off,on --reps 2 \
                  --profile prod25 --pool-gb 8 --port 8097 \
                  --json Backup/phase_decomp/plain_match_frozen.json
```

順序是**鏡像**（ABCCBA）：兩個 `on` 在第 0、5 位，兩個 `off` 在第 1、4 位，兩個 `on-exact` 在第 2、3 位 —— 單調漂移（熱／swap）無法解釋臂間差異。

| # | 臂 | 字元數 | 臂內決定性 | draft 接受 | engine digest |
|---|---|---|---|---|---|
| 0 | `on` | 262 | ✔ | 76/149 | aab787412e572550 |
| 1 | `off` | **278** | ✔ | – | aab787412e572550 |
| 2 | `on-exact` | 262 | ✔ | 76/149 | aab787412e572550 |
| 3 | `on-exact` | 262 | ✔ | 76/149 | aab787412e572550 |
| 4 | `off` | **278** | ✔ | – | aab787412e572550 |
| 5 | `on` | 262 | ✔ | 76/149 | aab787412e572550 |

6 次啟動的 engine digest **只有 1 個相異值**。

### 判定

| 主張 | 結果 |
|---|---|
| `on` vs `off` | **PLAIN_MATCH FALSE**：`57/262` 位置不同（21.76%），首分歧在第 205 字 |
| 載體位元組 | 兩臂 `realpath`／`size`／head-tail hash 相同（13,663,116,512 B，`6d34d8cc3c38fce1`） |
| 臂內決定性 | 每一臂的兩次啟動逐字元相同 |
| accept | 每一個 MTP-on 臂都是 `76/149 = 51.0%` |
| `on` vs `on-exact` | **0/262 不同** ⇒ **FAST PATH EXONERATED**：關掉 VERIFY/DRAFT 的 ZERO-slot 快路徑什麼都沒改變，所以分歧**不是** pool 近似，而在 **batch/verify 路徑本身** |

**結論在固定 build 上成立，而且現在是**可歸因**的**：MTP-off 的 `278` 與 MTP-on 的 `262` 從鏡像兩端各自重現，共用一顆 build。

---

## 5. 現在能主張什麼、不能主張什麼

**能主張**
- 這顆 build 對 v6 參考是 M1/M2/M3 逐位元相同的（9/9/9）。
- 這顆 build 就是凍結原始碼的建置，且對該狀態決定性（重編重鏈後位元不變）。
- 在同一顆 build 上，MTP-on 與 MTP-off 的 greedy 輸出**系統性不同**（57/262，首分歧 205），且這與 fast path 無關。

**不能主張**
- **02:57 那個 `TRUE` 仍然無法解釋**，而且它的產物已不在磁碟上。它是凍結**之前**的唯一資料點，任何引用它的句子都必須標明「產物不可考」；本輪的結論不依賴它（是獨立重測出來的）。
- 上面所有比較是**渲染文字**層級（這顆 build 不回傳 `tokens`）。文字差異**一定**是 token 差異（反向不成立，只會掩蓋），且比較窗口是 `min(262,278)=262`。
- 單一 prompt、單一池子（8 GiB）、greedy（`CGC_FORCE_TEMP0=1` 由伺服器自己釘）。**不覆蓋 temp>0 的拒絕取樣。**
- 決定性主張是「就地重建」，不是乾淨目錄的 hermetic 重建。

---

## 6. 檔案

| 用途 | 路徑 |
|---|---|
| 凍結工具 | `scripts/check/engine_freeze.py` |
| 凍結記錄 | `Backup/engine_freeze/pre_plainmatch_0919.json` |
| 重建日誌 | `/tmp/rebuild_noop.log`、`/tmp/rebuild_probe.log` |
| 產物備份（重建前） | `/tmp/engine_bak_pre_plainmatch_0919/` |
| M1/M2/M3 閘門 | `Backup/m123_oracle_gate/summary_anchor0919b.json` |
| 六臂原始資料 | `Backup/phase_decomp/plain_match_frozen.json` |
| 驅動 | `scripts/check/plain_match_ab.py` |

## 7. 使用規則（這一輪真正要留下的東西）

> 引用任何速度或身分數字之前，先跑 `engine_freeze.py verify --tag <該輪的 tag>`。
> `source`/`flags`/`artifacts` 三節都 MATCH 才代表「這數字是關於這顆 build」；
> 任何一節 DRIFT，就該標成不可歸因而不是拿來比較。

驅動端負責記 `engine_digest`（產物），`engine_freeze.py` 負責記原始碼與兩者的配對，閘門負責記判定 —— 三者用同一個 build 名稱串起來。

機器清乾淨：0 個 llama-server、0 個驅動。改動未 commit。
