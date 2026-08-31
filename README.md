# cgcengine0823 — CGC llama.cpp fork

llama.cpp fork，實作 **CGC bounded-residency 專家快取**：MoE 專家權重不常駐 GPU，
以 bounded pool（預設 4GiB）+ 檔案按需 pread 填充，讓 35B MoE（13GB+ GGUF）在
**16GB 統一記憶體**機器上 `-ngl 99` 全卸載。MTP 生產線實測 **26.5 t/s**
（accept 99.2%、hit 88.9%）。

- 生產 binary：`llama-simple`（非 MTP）與 `llama-speculative-simple`（MTP），
  **build 產物隨 git 版控**（`src/llama.cpp/build/bin/`）—— macOS checkout 後免重建可直接跑。
- 詳細文件見文末「文件地圖」。

---

## 快速上手（macOS）

### 0. 環境

- macOS（Apple Silicon / Metal）— 全功能平台（L4 pool / watchdog / MMV_FUSE）
- Python 3 + `huggingface_hub`（拿模型用）：`pip install -U "huggingface_hub[cli]"`
- Windows 夥伴請先看下方「平台差異」——8GB 機不能開 cache，行為不同

### 1. Clone

```bash
git clone https://github.com/alexchuang19760730/cgcengine0823.git
cd cgcengine0823
```

### 2. 拿模型（GGUF 不進 git —— 11–14GB 超過 GitHub 單檔 100MB 上限）

模型正式備份在 Hugging Face：`Alexchuang/cgcengine-models`。
git 內的錨點是 [`models/gguf/MANIFEST.md`](models/gguf/MANIFEST.md)（5 模型清單 / 角色 /
SHA256 / curl 直下法）與 `models/gguf/SHA256SUMS`。

```bash
# 按需下載（跑哪條線下哪個；MTP 生產線用 denseIQ4X 載體）：
hf download Alexchuang/cgcengine-models Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf --local-dir models/gguf
hf download Alexchuang/cgcengine-models Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --local-dir models/gguf

# 驗證（hf download 本身有服務端 hash 校驗，此為離線複查）：
cd models/gguf && shasum -a 256 -c SHA256SUMS && cd ../..
```

忘了這步也沒關係：跑 `run_n30cache.sh` 撞到模型缺失時，錯誤訊息會直接印出對應的下載指令。

### 3. 跑（git 內 binary 免重建）

```bash
# 非 MTP（llama-simple，~7 t/s 為結構性基準）：
./scripts/run_n30cache.sh -m qwen36 -n 64 -p "The capital of France is"

# MTP 生產線（llama-speculative-simple，26.5 t/s）：
./scripts/run_n30cache.sh -m qwen36 --mtp --dense-iq4x -n 64 -p "The capital of France is"

# gemma4 家族（-ngl 30）：
./scripts/run_n30cache.sh -m gemma4 -n 64 -p "..."
```

要自建（改了原始碼 / 非 macOS）：

```bash
scripts/build_cgc_llama.sh          # cmake --build，產物進 src/llama.cpp/build/bin
```

### 3b. 跑 HTTP 服務（llama-server，給 Windows/其他夥伴遠端測試）

```bash
./scripts/run_server.sh             # 0.0.0.0:8080，OpenAI 相容，log 進 Backup/cgc_logs/
```

- 夥伴端：程式直接指 `http://<Mac LAN IP>:8080/v1`（腳本啟動完成會印連線卡）
- 本機 curl 測 localhost **必帶 `--noproxy '*'`**（本地代理會攔 127.0.0.1 → 502 空回應）
- 內建防護：啟動前清殘留行程、記憶體 <25% 拒跑（16GB 機疊 13GB 模型會 kernel panic——**server 運行期間本機禁跑任何 13GB 級操作**）
- 非 MTP 基線 ~7 t/s（結構性）；注意 Qwen3 為思考模型，簡短問答的答案在 `reasoning_content` 欄位

### 4. 驗收（steady-state 回歸基準）

```bash
./scripts/run_n30cache.sh -m qwen36 --mtp --dense-iq4x --steady
```

通過標準：**≥26 t/s、accept ≥99%、hit ≥88%**（固定 seed 42 + 長 prompt + 自動 watchdog）。

---

## 常用環境變數（run_n30cache.sh 已打包生產值）

| 用途 | 變數 |
|---|---|
| pool 預算 | `N30CACHE_BUDGET`（預設 4GiB；8GiB 實測**反效果**，別加大） |
| 填充執行緒 | `N30CACHE_WORKERS`（8 = 甜蜜點） |
| 每層槽位配置 | `N30CACHE_LAYER_CAPS`（steady 預設 `40-40:256`；`=0` 關閉） |
| watchdog | `--steady` 自動開；`N30CACHE_WATCHDOG=0` 顯式關 |
| seed | `N30CACHE_SEED`（bit-identity 對照必設固定值） |

## 平台差異

| 平台 | 狀態 |
|---|---|
| macOS（Apple Silicon） | 全功能（本 README 主線）；`--no-mmap` 已硬編碼防 16GB 冷頁風暴 |
| Windows 8GB | **cache/L4/watchdog 不適用**（無 Metal backend）；`--no-mmap` 在 8GB 機會 OOM，跑 Windows 路徑前先看三平台白皮書；MSYS2 無 `/usr/bin/time` |
| HarmonyOS / 端側 | `deploy-harmonyos/`（部署包自包含 dylib 全家） |

## 貢獻者須知（commit 紀律）

1. **裝 pre-commit hook**（一次）：
   ```bash
   scripts/check_build_tracked.sh --install-hook
   ```
2. 8 項檢查每次 commit 自動跑，其中兩項是**防呆鐵律**：
   - **T-P7 死鎖防護守衛**：`llama-expert-cache.cpp` 的 `batch_owned` 精確 mask 與
     FATAL 看門狗不得被移除（2026-08-30 hist-prefetch 死鎖教訓，commit `4a746c724`）
   - **T-P8 原始碼↔binary 同步**：改了 llama 原始碼 → 同一 commit 帶 build 產物且
     binary 不准比原始碼舊（保住「checkout 免重建」不變量）
3. 分支紀律：開發在 `dev`；晉升一律 `git switch main && git merge --ff-only dev`（hook 擋 main 直接 commit）
4. 新增/更換模型：`shasum -a 256` 更新 `MANIFEST.md`+`SHA256SUMS` →
   `hf upload Alexchuang/cgcengine-models <file> <file>` → 同一 commit
5. 發版前固定驗收：若本輪涉及 `deploy-harmonyos` / macOS bundle，同步執行
   `BIN_DIR=src/llama.cpp/build/bin RUN_DEPLOY_HARMONYOS_ACCEPT=1 ./scripts/check_build_tracked.sh --repo "$PWD"`
   ，確認 deploy bundle 可重建且 `llama-server` / `llama-simple` 可啟動。

## 文件地圖

| 文件 | 內容 |
|---|---|
| [`moeexpert/doc/CGC_生產級Release_2026-08-30.html`](moeexpert/doc/CGC_生產級Release_2026-08-30.html) | **生產級 release**：開發介紹 / 測試覆蓋 / 指標說明 / release note |
| [`moeexpert/doc/CGC_測試計劃_白皮書_2026-08-30.html`](moeexpert/doc/CGC_測試計劃_白皮書_2026-08-30.html) | 測試計劃（L0–L3、T-P/T-A/T-E 全項 + 排查表） |
| [`moeexpert/`](moeexpert/) | 白皮書群（race conditions / CPOT 工具 / 三平台） |
| [`models/gguf/MANIFEST.md`](models/gguf/MANIFEST.md) | 模型資產錨點（下載 / 驗證 / 在地重生） |
