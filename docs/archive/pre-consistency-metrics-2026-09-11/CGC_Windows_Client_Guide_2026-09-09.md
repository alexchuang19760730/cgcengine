# CGC Expert Cache 推理服務 — Windows 端使用與測試指南

> **版本**: demo/sweet-spot-windows-fix (基於 production @ e81b04cd6)  
> **更新日期**: 2026-09-09  
> **適用對象**: Windows 端開發者、測試人員、最終用戶

---

## 1. 服務概述

本服務是基於 llama.cpp 的 CGC Expert Cache 優化推理服務，運行在 Mac M4 Air (16GB) 上，提供 OpenAI 相容的 API 介面。

### 1.1 核心規格

| 項目 | 規格 |
|------|------|
| **模型** | Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X (13GB) |
| **模型架構** | MoE (Mixture of Experts), 35B 總參數, ~3B 激活參數 |
| **量化** | IQ3_XXS (主體) + denseIQ4X (部分層) |
| **Expert Cache** | 8GB pool, 每層 ~143 slots |
| **上下文長度** | 8192 tokens |
| **GPU 層數** | 99 (全部離線) |
| **MTP 投機解碼** | draft-mtp, n_max=3 |
| **KV Cache** | Q8_0 量化 (節省 ~0.56GB) |

### 1.2 性能基準 (甜蜜點配置)

| 指標 | 數值 | 備註 |
|------|------|------|
| **確定性測試** | 90-100% 正確率 | 15+27 探針, 10 連發 |
| **Decode 速度** | 14-25 t/s | 取決於 profile 和機器狀態 |
| **Prefill 速度** | 8-15 t/s | 短 prompt |
| **Draft Accept** | 45-95% | 健康生成 ~55-95%, 退化生成偏低 |
| **7-profile 品質** | 3-5/7 達標 | 取決於機器記憶體壓力 |

---

## 2. 連線資訊

### 2.1 API 端點

| 項目 | 值 |
|------|-----|
| **Base URL** | `http://192.168.101.90:8080/v1` |
| **API Key** | 任意字串 (如 `sk-llama-local-cgc`) |
| **Model** | 留空或任意 (server 只有一個模型) |
| **Health Check** | `http://192.168.101.90:8080/health` |
| **Models List** | `http://192.168.101.90:8080/v1/models` |

### 2.2 網段選擇

取決於 Windows 如何連接 Mac：

| Windows 位置 | 使用 IP |
|-------------|---------|
| 同一實體 LAN (Wi-Fi/網路線) | `192.168.101.90` |
| Windows VM (UTM/Parallels 橋接) | `192.168.2.1` |

### 2.3 連線驗證

在 Windows 端開啟 cmd 或 PowerShell：

```powershell
# 1. Ping 測試
ping 192.168.101.90

# 2. Health Check
curl http://192.168.101.90:8080/health

# 3. 模型列表
curl http://192.168.101.90:8080/v1/models
```

預期輸出：
```json
{"status":"ok"}
```

---

## 3. 支援的用戶端

### 3.1 OpenAI 相容用戶端

任何支援 OpenAI API 的用戶端都可以使用：

| 用戶端 | 設定方式 |
|--------|---------|
| **Claude Code CLI** | 環境變數 `ANTHROPIC_BASE_URL` 或設定檔 |
| **OpenWebUI** | 設定 → 連線 → OpenAI API → Base URL |
| **Continue.dev** | VS Code 外掛，設定 `models.providers` |
| **Cursor** | 設定 → Models → OpenAI 相容 |
| **自訂程式** | 直接呼叫 REST API |

### 3.2 Claude Code CLI 設定範例

```bash
# 設定環境變數
export ANTHROPIC_BASE_URL=http://192.168.101.90:8080/v1
export ANTHROPIC_API_KEY=sk-llama-local-cgc

# 啟動 Claude Code
claude --bare
```

或在 `~/.claude/settings.json` 中設定：

```json
{
  "apiKey": "sk-llama-local-cgc",
  "baseURL": "http://192.168.101.90:8080/v1"
}
```

---

## 4. API 使用範例

### 4.1 基本聊天完成 (curl)

```powershell
curl http://192.168.101.90:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "你好，請介紹一下自己"}
    ],
    "max_tokens": 200
  }'
```

### 4.2 Python 範例

```python
import requests

url = "http://192.168.101.90:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "default",
    "messages": [
        {"role": "system", "content": "你是一個有幫助的助手。"},
        {"role": "user", "content": "用 Python 寫一個快速排序函數"}
    ],
    "max_tokens": 500,
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result["choices"][0]["message"]["content"])
```

### 4.3 串流 (Streaming) 範例

```python
import requests

url = "http://192.168.101.90:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "default",
    "messages": [{"role": "user", "content": "寫一首關於秋天的詩"}],
    "max_tokens": 300,
    "stream": True
}

response = requests.post(url, headers=headers, json=data, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

---

## 5. 配置參數說明

### 5.1 服務端強制參數 (CGC_FORCE_TEMP0)

由於 IQ3_XXS 量化模型在 temperature > 0 時會出現品質崩潰（echo、think-loop、錯誤答案），服務端**強制覆蓋**以下參數：

| 參數 | 強制值 | 說明 |
|------|--------|------|
| **temperature** | `0.0` | 貪婪解碼，確保確定性 |
| **seed** | `0` (若未提供) | 固定隨機種子 |

> **注意**: 即使用戶端傳入 `temperature=0.7`，服務端也會強制改為 `0.0`。這是為了保證 IQ3_XXS 模型的品質穩定性。

### 5.2 建議用戶端參數

| 參數 | 建議值 | 說明 |
|------|--------|------|
| **max_tokens** | 依需求 | 短查詢 30-50, 代碼生成 200-500 |
| **top_p** | 0.8 (服務端預設) | 核心取樣 |
| **top_k** | 0 (服務端預設) | 禁用 top-k |
| **presence_penalty** | 0.0-0.3 | 抑制重複，>0.5 可能導致數字幻覺 |
| **frequency_penalty** | 0.0 | >0 可能導致數字幻覺，不建議使用 |
| **repeat_penalty** | 1.0-1.1 | >1.3 可能導致數字幻覺 |

> **重要**: IQ3_XXS 模型對 penalty 參數非常敏感。`presence_penalty >= 0.5` 或 `frequency_penalty > 0` 或 `repeat_penalty >= 1.3` 都可能導致數字幻覺（如 "1.000000..."）或輸出異常。

### 5.3 不建議使用的參數

| 參數 | 原因 |
|------|------|
| **frequency_penalty > 0** | 導致數字幻覺 |
| **presence_penalty >= 0.5** | 導致數字幻覺 |
| **repeat_penalty >= 1.3** | 導致數字幻覺 |
| **temperature > 0** | 服務端強制覆蓋為 0 |
| **logit_bias** | 未經測試，可能導致異常 |

---

## 6. 測試用例

### 6.1 基本功能測試

#### 測試 1: 英文短查詢

```powershell
curl http://192.168.101.90:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "2+2 is? Answer with one word."}],
    "max_tokens": 30
  }'
```

**預期結果**: 輸出包含 "4"，無 echo，無 `<think>` 標籤。

#### 測試 2: 中文短查詢

```powershell
curl http://192.168.101.90:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "15+27等於多少？只回答數字"}],
    "max_tokens": 30
  }'
```

**預期結果**: 輸出包含 "42"，無 echo。

#### 測試 3: 代碼生成

```powershell
curl http://192.168.101.90:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a Python function to calculate fibonacci."}],
    "max_tokens": 200
  }'
```

**預期結果**: 輸出 Python 代碼，包含 `def` 和 `return`，無循環。

#### 測試 4: 多輪對話

```powershell
curl http://192.168.101.90:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi there! How can I help you?"},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 100
  }'
```

**預期結果**: 輸出包含 "4"，上下文連貫。

### 6.2 性能測試

#### 測試 5: 速度測試

```python
import requests
import time

url = "http://192.168.101.90:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 短 prompt 測試
data = {
    "model": "default",
    "messages": [{"role": "user", "content": "寫一個 Python 的 Hello World"}],
    "max_tokens": 100
}

start = time.time()
response = requests.post(url, headers=headers, json=data)
elapsed = time.time() - start

result = response.json()
tokens = result["usage"]["completion_tokens"]
print(f"生成 {tokens} tokens, 耗時 {elapsed:.2f}s, 速度 {tokens/elapsed:.2f} t/s")
```

**預期結果**: Decode 速度 14-25 t/s（取決於機器狀態）。

### 6.3 品質測試

#### 測試 6: 7-profile 基準測試

在 Mac 端執行：

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
python3 scripts/check/replay_server_profile.py \
  --base-url http://127.0.0.1:8080 \
  --all-profiles \
  --warmup \
  --runs 3 \
  --seed 0 \
  --reference scripts/check/replay_bench_reference_v2.json \
  --bench-output /tmp/cgc_benchmark.json
```

**7 個 profile**:
1. **qa-zh** — 中文短問答
2. **longform-zh** — 中文長文本
3. **coding** — 代碼生成
4. **math** — 數學計算
5. **reasoning** — 邏輯推理
6. **writing** — 寫作
7. **translation** — 翻譯

**預期結果**: 3-5/7 profile 品質達標（≥0.9），取決於機器記憶體壓力。

---

## 7. 已知問題與限制

### 7.1 模型量化限制

| 問題 | 說明 | 影響 |
|------|------|------|
| **IQ3_XXS 量化精度** | 3-bit 量化，精度較低 | 簡單數學可能出錯（如 15+27=44） |
| **temperature 強制為 0** | 服務端強制覆蓋 | 無法使用隨機性生成，輸出確定但可能單調 |
| **penalty 參數敏感** | presence/frequency/repeat penalty 過高會導致數字幻覺 | 不建議使用非零 penalty |

### 7.2 服務端限制

| 問題 | 說明 | 影響 |
|------|------|------|
| **單一模型** | server 只載入一個模型 | 無法在多模型間切換 |
| **無並發** | `--np 1`，單一請求佇列 | 同時只能處理一個請求 |
| **記憶體限制** | 16GB Mac，模型 ~8GB + Expert Cache 8GB | 記憶體壓力大時可能 OOM 或降速 |
| **無認證** | 不需要 API Key | 區網內可直接存取，請勿暴露到公網 |

### 7.3 常見異常輸出

| 異常現象 | 可能原因 | 解決方案 |
|---------|---------|---------|
| **數字幻覺** ("1.000000...") | presence_penalty >= 0.5 或 frequency_penalty > 0 | 將 penalty 設為 0 |
| **echo 循環** (重複 prompt) | 記憶體壓力過大或模型退化 | 重啟 server，釋放記憶體 |
| **`<think>` 標籤** | chat template 或模型輸出 | 忽略標籤，或在用戶端過濾 |
| **`<im_end>` 標籤** | chat template 邊界問題 | 忽略標籤，或在用戶端過濾 |
| **代碼循環** (重複函數定義) | prefill 後模型進入循環 | 增加 max_tokens，或使用更具體的 prompt |
| **速度驟降** (<10 t/s) | 記憶體壓力過大，開始使用 swap | 關閉其他應用，重啟 server |

---

## 8. 故障排除

### 8.1 連線問題

#### 問題: 無法連線到 server

**檢查步驟**:
1. 確認 Mac 端 server 正在運行：
   ```bash
   ps aux | grep llama-server
   ```
2. 確認防火牆沒有擋住 8080 埠：
   ```bash
   # Mac 端
   curl http://127.0.0.1:8080/health
   ```
3. 確認 Windows 和 Mac 在同一網段：
   ```powershell
   # Windows 端
   ping 192.168.101.90
   ```
4. 確認使用正確的 IP（實體 LAN 用 192.168.101.90，VM 用 192.168.2.1）

### 8.2 品質問題

#### 問題: 輸出品質差，有 echo 或循環

**解決方案**:
1. 檢查 Mac 端記憶體狀態：
   ```bash
   top -l 1 | grep PhysMem
   ```
   如果 Available < 4GB，建議關閉其他應用或重啟 Mac。

2. 重啟 server：
   ```bash
   pkill -INT -f llama-server
   cd /Users/alexchuang/Documents/flashkv-devserver
   ./scripts/run_server.sh
   ```

3. 檢查用戶端參數，確保沒有使用過高的 penalty：
   ```json
   {
     "presence_penalty": 0.0,
     "frequency_penalty": 0.0,
     "repeat_penalty": 1.0
   }
   ```

### 8.3 速度問題

#### 問題: 生成速度慢 (<10 t/s)

**解決方案**:
1. 檢查記憶體壓力，關閉其他應用
2. 減少 max_tokens（短查詢用 30-50）
3. 減少上下文長度（如果用戶端有快取，清除舊對話）
4. 重啟 server 釋放記憶體碎片

### 8.4 API 錯誤

#### 問題: 收到 500 或 503 錯誤

**可能原因**:
- server 正在處理其他請求（單一佇列）
- 記憶體不足，OOM
- 模型載入失敗

**解決方案**:
1. 等待幾秒後重試
2. 檢查 Mac 端 server 日誌：
   ```bash
   tail -50 /Users/alexchuang/Documents/flashkv-devserver/Backup/cgc_logs/llama_server_latest.log
   ```
3. 重啟 server

---

## 9. 進階設定

### 9.1 自訂系統提示詞

在 API 請求中加入 system message：

```json
{
  "messages": [
    {"role": "system", "content": "你是一個專業的 Python 開發者，只輸出代碼，不要解釋。"},
    {"role": "user", "content": "寫一個快速排序函數"}
  ]
}
```

### 9.2 使用 assistant_prefill

引導模型開始生成：

```json
{
  "messages": [
    {"role": "user", "content": "寫一個 Python 的 fibonacci 函數"}
  ],
  "chat_template_kwargs": {
    "assistant_prefill": "```python\ndef fibonacci(n):\n    "
  }
}
```

> **注意**: `assistant_prefill` 可以有效引導模型進入正確的生成模式，減少循環機率。

### 9.3 串流生成

設定 `stream: true` 以獲得更好的用戶體驗：

```json
{
  "stream": true,
  "messages": [{"role": "user", "content": "寫一篇關於人工智慧的文章"}]
}
```

---

## 10. 聯絡與支援

### 10.1 取得 server 狀態

在 Mac 端執行：

```bash
# 健康檢查
curl http://127.0.0.1:8080/health

# 查看 server 日誌
tail -f /Users/alexchuang/Documents/flashkv-devserver/Backup/cgc_logs/llama_server_latest.log

# 查看記憶體使用
ps aux | grep llama-server | grep -v grep
```

### 10.2 重啟 server

```bash
# 優雅關閉
pkill -INT -f llama-server

# 等待 3 秒
sleep 3

# 重新啟動
cd /Users/alexchuang/Documents/flashkv-devserver
./scripts/run_server.sh
```

### 10.3 相關文件

- **技術白皮書**: `docs/ExpertCache_關鍵版本技術白皮書_2026-09-08.html`
- **TPOT 延遲分析**: `docs/CGC_TPOT_延遲分解與優化路線圖_2026-09-07.html`
- **品質漂移修復方案**: `docs/CGC_品質漂移修復方案開發建議書_2026-09-07.html`
- **測試腳本**: `scripts/check/replay_server_profile.py`
- **Server 啟動腳本**: `scripts/run_server.sh`

---

## 11. 已修復的問題與最佳實踐 (2026-09-09 更新)

### 11.1 已修復的問題

本版本 (`0bf3a8e5c`) 修復了 Windows 端兩次回報的品質退化問題：

| 問題 | 根本原因 | 修復方案 | 狀態 |
|------|----------|----------|------|
| `<think>` 標籤作為 literal text 輸出 | `--reasoning off --reasoning-format deepseek` 矛盾組合 | 改為 `--reasoning-format none` | ✅ 已修復 |
| 短查詢 echo 循環 (T1/T2) | `CGC_FORCE_TEMP0=1` 強制 temperature=0，greedy 解碼進入循環 | 改為 `CGC_FORCE_TEMP0=0`，允許用戶端設置 temperature | ✅ 已修復 |
| coding profile fibonacci 循環 | coding profile 缺少 `presence_penalty=1.5` | 添加 `presence_penalty=1.5` | ✅ 已修復 |
| `/props` 顯示 MTP 沒開 | llama.cpp `/props` 顯示預設值而非實際運行值 | MTP 實際上是開的（`--spec-type draft-mtp`），文檔說明 | ✅ 已澄清 |

### 11.2 測試結果驗證

| 測試用例 | 修復前 | 修復後 | 說明 |
|----------|--------|--------|------|
| **T1: Short EN Q&A** | ❌ echo 循環 | ✅ PASS (2+2=4) | 12-15 t/s |
| **T2: Short CN Q&A** | ❌ Content 為空 | ✅ PASS (15+27=42) | 15-20 t/s |
| **T3: Code Gen** | ❌ 代碼 fence 循環 | ⚠️ 需要顯式 prefill | 見下方最佳實踐 |
| **T4: Logic** | ❌ 只輸出 "1.\n" | ⚠️ IQ3_XXS 限制 | 見下方說明 |
| **T5: Long Context** | ❌ `<think>` 標籤 | ⚠️ 需要顯式 prefill | 見下方最佳實踐 |

### 11.3 最佳實踐：顯式添加 assistant_prefill

**IQ3_XXS 量化模型的固有限制**：對於代碼生成和長文本生成，建議在請求中顯式添加 `chat_template_kwargs.assistant_prefill` 來引導模型，避免進入循環。

#### 代碼生成示例

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "Write a Python function to calculate fibonacci."}],
  "temperature": 0.3,
  "max_tokens": 200,
  "chat_template_kwargs": {
    "assistant_prefill": "```python\ndef fibonacci(n):\n    "
  }
}
```

**驗證結果**：Decode 20.92 t/s，輸出包含完整的 fibonacci 函數定義和 return 語句 ✅

#### 長文本生成示例

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "請用一段中文說明為什麼巴黎會成為法國的政治與文化中心。"}],
  "temperature": 0.3,
  "max_tokens": 300,
  "chat_template_kwargs": {
    "assistant_prefill": "巴黎之所以成為法國的政治與文化中心，主要是因為"
  }
}
```

### 11.4 推薦的 temperature 設置

| 場景 | 推薦 temperature | 說明 |
|------|-----------------|------|
| **短查詢 (Q&A)** | 0.1-0.3 | 避免 greedy 循環，同時保持準確性 |
| **代碼生成** | 0.3 + 顯式 prefill | prefill 是關鍵，temperature 次要 |
| **長文本生成** | 0.3 + 顯式 prefill | prefill 是關鍵，避免 `<think>` 標籤 |
| **創意寫作** | 0.5-0.7 | 可能需要更高的隨機性，但 IQ3_XXS 品質會下降 |

**注意**：不要設置 `presence_penalty > 0.3` 或 `frequency_penalty > 0`，這會導致 IQ3_XXS 模型出現數字幻覺（如 "1.000000..."）。

### 11.5 auto_anchor 的已知問題

`CGC_SERVER_AUTO_ANCHOR=1` 已配置，但目前對於 code-like prompts 沒有有效注入 prefill。這是一個已知問題，後續版本會修復。在此期間，請使用顯式的 `chat_template_kwargs.assistant_prefill`。

---

## 附錄 A: 快速開始 checklist

- [ ] 確認 Mac 端 server 正在運行 (`ps aux | grep llama-server`)
- [ ] 確認 Windows 可以 ping 通 Mac (`ping 192.168.101.90`)
- [ ] 確認 Health Check 正常 (`curl http://192.168.101.90:8080/health`)
- [ ] 設定用戶端 Base URL 為 `http://192.168.101.90:8080/v1`
- [ ] 設定 API Key 為任意字串
- [ ] 確保用戶端沒有設定 `presence_penalty > 0.3` 或 `frequency_penalty > 0`
- [ ] 執行基本測試（2+2 查詢）
- [ ] 執行代碼生成測試
- [ ] 確認輸出無 echo、無 `<think>` 標籤、無數字幻覺

---

## 附錄 B: 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| **demo/sweet-spot-windows-fix** | 2026-09-09 | 基於 production @ e81b04cd6，修復 coding profile presence_penalty，新增 Windows 端使用指南 |
| **production @ e81b04cd6** | 2026-09-08 | 白皮書補 §0.5 pool 曲線重測，8GB 甜蜜點 16.1 t/s |
| **production @ abda5f8c3** | 2026-09-08 | Expert Cache 關鍵版本技術白皮書 2026-09-08 |
| **v1.0.0-production @ 9dde528a5** | 2026-09-07 | CGC Expert Cache 25.17 t/s，三 profile 質量 baseline |

---

**文件結束**

> 如有問題，請參考技術白皮書或聯繫開發團隊。
