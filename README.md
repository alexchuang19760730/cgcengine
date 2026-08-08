# CGC — ComputeGraphCompiler

端側推理引擎 + 端雲協同的統一編譯/推理框架。

包含：CGC 引擎（Python + C++/Metal kernel）、gate 白皮書（M1-M7.6 / UPKG）、edge-cloud 架構設計、
`app/`（edge_engine / servers / shared / training）、`rswaengine/`（TrueOrthoKDA + RSWA C++ 實作）、
以及 TurboFieldfare（端側 Swift + Metal MoE streaming 引擎，獨立 repo，見下方）。

## 目錄結構

| 路徑 | 內容 |
|---|---|
| `ComputeGraphCompiler-main/` | CGC 引擎主體（cgc_engine / docs / tools / tests）|
| `docs/` | 架構白皮書、protocol、report contracts |
| `app/` | edge-first 應用層（edge_engine / servers / cli / shared / training）|
| `rswaengine/` | C++ TrueOrthoKDA/RSWA kernel 與 Python 綁定 |
| `CGC_Phase2/` | MTP/DeltaNet 探針、實驗腳本 |
| `external_refs/` | host2 RSWA 整合參考 |
| `*.md`（根目錄）| 架構總覽與效能報告 |

## 已排除內容（本 repo 不含）

本 repo 是**公開**的，以下內容刻意**不入庫**（留在本地工作目錄）：

- **含明文憑證的作業腳本（296 個）**——`restart_*.py`、`get_traj*.py`、`run_smoke*.py` 等，
  內含伺服器 IP 與 root 密碼。如需納入：先改成讀環境變數（如 `CGC_HOST1_PWD`）再 commit。
- **模型權重**：`models/`、`*.gguf`、`*.safetensors`、`*.bin`
- **vendored 後端**：`ComputeGraphCompiler-main/Backend/`（1.3GB，Llama.cpp/oMLX/vLLM/Ascend）
- **runtime 資料**：`var/`（benchmark logs）、`ComputeGraphCompiler-main/Output/`、`debug/`
- **大型資料集**：`CGC_TrainingData/`、`CGC_Release/`、`ds4/`、`prime-agent-worktrees/`
- **工具狀態目錄**：`.freebuff/`、`.workbuddy/`、`.dbg/`、`.vscode/`
- **嵌入式 git repo**：`colibri/`、`prime-agent/`、`realtime-vla-v2/`、`turbo-fieldfare*/`

要檢查一個檔案是否安全入庫，先確認無 `password`/`pwd1`/`hf_` token/private key 字樣：
```bash
grep -nE "password=|pwd1?=|hf_[A-Za-z0-9]{20,}" <檔案>
```

## 關聯專案

- **TurboFieldfare**（端側 Swift + Metal MoE streaming 引擎）→ 獨立 repo：
  [alexchuang19760730/edgeexpert](https://github.com/alexchuang19760730/edgeexpert)
  - 端側 Gemma4 26B-A4B 推理：r3 **22.6 tok/s** / r4 **21.5 tok/s** @256tok（乾淨窗）
  - MoE expert streaming + hot pool + MTP draft 等優化，詳見該 repo README

## 快速開始

```bash
# 下載/Clone
git clone git@github.com:alexchuang19760730/cgcengine.git

# 引擎測試（ComputeGraphCompiler-main 內）
cd ComputeGraphCompiler-main
python3 -c "import cgc_engine; print(cgc_engine.__file__)"
```

> 註：本 repo 的作業腳本多數需連線至私有伺服器（未含在 repo 中），
> 端側推理請見 [edgeexpert](https://github.com/alexchuang19760730/edgeexpert) 的 TurboFieldfare。
