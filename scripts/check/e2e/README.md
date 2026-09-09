# CGC End-to-End (E2E) Quality Test Framework

## 概述

這是一個端到端品質測試框架，用於解決這十天遇到的所有測試方法缺陷問題。

**核心設計原則**：
1. **裸請求測試**：不帶 assistant_prefill、不帶 stop tokens、用預設參數，模擬真實客戶端（Windows/Claude Code CLI）
2. **upstream 對比**：同時測試我們的 server 和 upstream 原版 server，區分「我們的修改問題」vs「模型/量化本身問題」
3. **agent 逐條驗收**：由 AI agent 逐條檢查每個 prompt 的輸出品質，不是只檢查 anchor substring
4. **問題覆蓋**：覆蓋這十天遇到的所有 24 個已知問題

## 文件結構

```
scripts/check/e2e/
├── README.md                          # 本文檔
├── e2e_quality_test.py                # 端到端品質測試腳本（裸請求 + upstream 對比）
├── agent_acceptance_checker.py        # Agent 逐條驗收檢查器
└── issue_coverage.py                  # 問題覆蓋清單（這十天遇到的所有問題）
```

## 快速開始

### 1. 只測我們的 server

```bash
python3 scripts/check/e2e/e2e_quality_test.py \
  --server http://127.0.0.1:8080 \
  --output /tmp/cgc_e2e_results.json
```

### 2. 同時測試我們的 server 和 upstream server（推薦）

```bash
# 啟動我們的 server（端口 8080）
./scripts/run_server.sh

# 啟動 upstream 原版 server（端口 8081）
# 使用相同的模型，但不帶任何 CGC 修改
./build/bin/llama-server -m models/gguf/your-model.gguf --port 8081

# 運行 e2e 測試（同時對比兩個 server）
python3 scripts/check/e2e/e2e_quality_test.py \
  --server http://127.0.0.1:8080 \
  --upstream-server http://127.0.0.1:8081 \
  --output /tmp/cgc_e2e_results.json
```

### 3. 只測特定 profiles

```bash
python3 scripts/check/e2e/e2e_quality_test.py \
  --server http://127.0.0.1:8080 \
  --profiles coding,math,reasoning \
  --output /tmp/cgc_e2e_results.json
```

### 4. 列出所有可用 profiles

```bash
python3 scripts/check/e2e/e2e_quality_test.py --list-profiles
```

### 5. 運行 agent 驗收檢查

```bash
python3 scripts/check/e2e/agent_acceptance_checker.py \
  --input /tmp/cgc_e2e_results.json \
  --output /tmp/cgc_e2e_acceptance_report.json \
  --verbose
```

### 6. 查看問題覆蓋清單

```bash
python3 scripts/check/e2e/issue_coverage.py
```

## Profiles 清單（15 個，共 50+ prompts）

| Profile | 描述 | Prompts 數 |
|---------|------|-------------|
| qa-zh | 中文短問答 | 5 |
| qa-en | 英文短問答 | 4 |
| longform-zh | 中文長文本生成 | 3 |
| coding | 代碼生成 | 5 |
| coding-debug | 代碼調試 | 2 |
| coding-explain | 代碼解釋 | 2 |
| math | 數學計算 | 5 |
| math-word-problem | 數學應用題 | 3 |
| reasoning | 邏輯推理 | 4 |
| writing | 寫作 | 3 |
| creative-writing | 創意寫作 | 3 |
| translation | 翻譯 | 3 |
| summarization | 摘要總結 | 2 |
| extraction | 信息提取 | 2 |
| role-play | 角色扮演 | 2 |

## 問題覆蓋清單（這十天遇到的 24 個問題）

### P0: 品質崩潰級問題（4 個）

| ID | 問題 | 檢測方法 |
|----|------|----------|
| P0-001 | ZERO-slot 污染導致 logits 崩潰 | upstream 對比 + echo/循環/空輸出檢測 |
| P0-002 | 「前幾次對、之後崩」順序依賴模式 | 同一 prompt 連發 3 次，檢查結果一致性 |
| P0-003 | v1/v2 測試誤判（只測 L3，沒測 L4） | 端到端裸請求測試取代 v1/v2 的有限覆蓋 |
| P0-004 | 甜蜜點品質假象（replay benchmark 帶拐杖） | 完全裸請求測試（不帶 assistant_prefill/stop tokens） |

### P1: 品質閘門缺陷（3 個）

| ID | 問題 | 檢測方法 |
|----|------|----------|
| P1-001 | quality gate「無 reference 直接給 1.0」bypass | 不依賴 reference rules，直接用 upstream 對比 + agent 驗收 |
| P1-002 | loop 偵測只在「有 reference rules」時才跑 | e2e 測試內建 loop 偵測（不依賴 reference） |
| P1-003 | MTP 迴圈時速度很好看，遮住品質崩潰 | 同時記錄速度和品質，品質優先於速度 |

### P2: Windows/客戶端品質退化（6 個）

| ID | 問題 | 檢測方法 |
|----|------|----------|
| P2-001 | Windows 端回顯 prompt 循環 | echo 檢測（檢查輸出是否包含 prompt 的大部分） |
| P2-002 | Content 為空 | 空輸出檢測（長度 < 5 視為空） |
| P2-003 | 代碼 fence 循環 | 代碼 fence 檢測（``` 出現 > 6 次視為異常） |
| P2-004 | <think> 標籤 literal 輸出 | thinking 標籤檢測（<think> 或 </think> 出現） |
| P2-005 | CGC_FORCE_TEMP0=1 強制 greedy 導致 echo 循環 | 所有測試用 temperature=0.7（非 greedy） |
| P2-006 | auto_anchor 對 code-like prompts 未有效注入 | coding profile 不帶 assistant_prefill 測試 |

### P3: 架構/配置問題（5 個）

| ID | 問題 | 檢測方法 |
|----|------|----------|
| P3-001 | MTP+OA_ASYNC+背景 pool fill 崩潰 | 長輸出測試（max_tokens=512）+ 穩定性檢查 |
| P3-002 | 8GB+RN（renorm）品質 0.3 deterministic | math profile 的計算題（檢查 counting loop） |
| P3-003 | ngl=30 速度慢且品質更差 | 所有測試在 ngl=99 下進行（生產配置） |
| P3-004 | 記憶體壓力導致品質退化 | 測試前檢查記憶體狀態，記錄 free memory 百分比 |
| P3-005 | loop guard 是事後補丁，不是根本解決 | e2e 測試不依賴 loop guard，直接檢查輸出品質 |

### P4: 測試/流程問題（4 個）

| ID | 問題 | 檢測方法 |
|----|------|----------|
| P4-001 | 品質閘門從未真的掛在 commit 上 | e2e 測試可作為 pre-commit hook 的一部分 |
| P4-002 | Replay 是「考自己出的題」，不是真實使用 | 用真實 prompt（不帶 prefill），agent 驗收完整輸出 |
| P4-003 | 並行 agent 干擾（頻繁重啟 server、pkill） | e2e 測試有超時處理和錯誤恢復 |
| P4-004 | Colab 環境問題（GPU 未啟用、CUDA 未安裝） | server health 檢查，報告 server 狀態 |

## Agent 驗收檢查維度（10 個）

1. **正確性**：答案是否正確（通過 upstream 對比）
2. **完整性**：輸出是否完整，沒有被截斷
3. **邏輯性**：推理是否合理
4. **循環檢測**：是否有重複循環（詞級、短語級、代碼 fence）
5. echo 檢測**：是否回顯 prompt
6. **thinking 標籤檢測**：是否有 <think> 標籤泄漏
7. **代碼質量**：代碼是否正確、可運行
8. **語言正確性**：是否用正確的語言回答
9. **upstream 對比**：與 upstream 原版 server 的輸出對比
10. **歸因分析**：問題是我們的修改導致的，還是模型/量化本身的問題

## 評分標準

| 分數 | 等級 | 說明 |
|------|------|------|
| ≥ 0.8 | pass | 品質合格 |
| 0.5 - 0.8 | partial | 部分合格，有改進空間 |
| < 0.5 | fail | 品質不合格 |

**扣分規則**：
- critical 問題：每個扣 0.5 分
- high 問題：每個扣 0.2 分
- medium 問題：每個扣 0.1 分

## 與舊 replay benchmark 的對比

| 維度 | 舊 replay benchmark | 新 e2e 測試框架 |
|------|---------------------|------------------|
| 請求類型 | 帶拐杖（assistant_prefill/stop tokens） | 裸請求（模擬真實客戶端） |
| 品質檢查 | anchor substring 匹配 | agent 逐條驗收（10 個維度） |
| upstream 對比 | 無 | 有（區分我們的修改 vs 模型問題） |
| 問題覆蓋 | 3-7 個 profile | 15 個 profile，50+ prompts |
| 歸因分析 | 無 | 有（our_modification / model_quantization / both） |
| 循環偵測 | 只在有 reference 時 | 內建，不依賴 reference |
| 速度 vs 品質 | 只看 t/s，容易被循環欺騙 | 品質優先，同時記錄速度 |

## 為什麼需要這個框架？（這十天的教訓）

1. **ZERO-slot 污染被掩蓋了很久**：舊 replay benchmark 帶了各種拐杖，掩蓋了 ZERO-slot 污染。直到用裸請求（Windows/Claude Code）測試才發現 0/10 崩潰。

2. **v1/v2 測試有設計缺陷**：只測了 L3 Soft Pool 的 fill 後 bytes 一致性，沒有測 L4 fast path 的端到端 logits。2433/2433 100% 一致只能證明 V1/V2 工具不影響 logits，不能證明 expert cache bit-correct。

3. **「歸咎於模型量化」是錯誤的**：upstream 原版 server 用同樣的模型、同樣的量化，10/10 全部正確。這證明問題是我們的修改導致的，不是模型/量化本身的問題。

4. **速度指標會欺騙**：MTP 循環時 draft accept 94.49%、decode 16-23 t/s——比健康生成還快。只看 t/s 的 regression 閘門在循環時永遠不會擋。

5. **品質閘門有漏洞**：「無 reference 直接給 1.0」bypass、loop 偵測只在有 reference 時才跑、品質閘門從未真的掛在 commit 上。

## 後續計劃

1. **集成到 pre-commit hook**：每次 commit 前自動運行 e2e 測試（可選，因為耗時較長）
2. **增加更多 profiles**：覆蓋更多真實使用場景（如 Claude Code CLI 的實際任務）
3. **增加壓力測試**：長時間運行、高並發、記憶體壓力下的穩定性測試
4. **增加回歸測試數據庫**：記錄每個 commit 的 e2e 測試結果，便於回歸分析
5. **與 CI/CD 集成**：在 GitHub Actions 中自動運行 e2e 測試

## 聯繫與反饋

如有問題或建議，請在 GitHub 上提交 issue 或 PR。
