# agent_harness/qwen36/ — Qwen3.6-35B-A3B 的推論整合層

**這不是 agent harness 的迴路，是它的一個依賴。** 只有一個檔案（`__init__.py`，10,281 bytes，
3 個 def/class）—— 但「只有一個檔」不等於「不重要」：它是 harness 叫得動模型的其中一條路。

## 它自己說它是什麼

`__init__.py` 的 docstring（原文）：

> Qwen3.6-35B-A3B inference integration layer.
>
> Provides unified interface to Qwen3.6 inference via:
> 1. CGC edge_server (OpenAI-compatible API, with expert-cache + MTP + DOPD)
> 2. llama.cpp llama-server (local GGUF inference)
> 3. Direct PyTorch (for Loop MoE model after grafting)
>
> Used by:
> - loopmoe_agent_adapter.py (Terminal-Bench agent)
> - finetune/finetune_loopmoe.sh (training data generation)
> - SFT data generation pipeline

## ★ 它宣告的三個消費者，現在都在 `tb_loop/` 底下

| 它寫的名字 | E1 之後的實際位置 |
|---|---|
| `loopmoe_agent_adapter.py` | `tb_loop/agents/loopmoe_agent_adapter.py` |
| `finetune/finetune_loopmoe.sh` | `tb_loop/finetune/finetune_loopmoe.sh` |
| SFT data generation pipeline | `tb_loop/` 的 SFT 資料流程（`sft_data*/`） |

所以 `qwen36/` 是 **tb_loop 的依賴**，只是它自己沒跟著搬 —— 與 `../loopmoe/` 同一情況
（見那份 README 對「錨點為什麼要拆兩層」的說明）。

## 引用廣度與一個容易誤導的地方

- 追蹤檔：1（`git ls-files agent_harness/qwen36 | wc -l`）
- 進來的**路徑**引用（`agent_harness/qwen36/`）：**0** 處
- ⚠️ 但全 repo 搜裸字串 `qwen36` 有 **83 處** —— 那些絕大多數指的是**模型**
  （`Qwen3.6-35B-A3B` 的別名，出現在 `CGC-main/`、`scripts/`、白皮書裡），不是這個套件。
  **這是這個目錄唯一容易誤導的地方**：看到 83 處引用會以為它被大量使用，
  而實際上作為**路徑**它一次都沒被引用過。
  判準：用完整路徑前綴 `agent_harness/qwen36/`，不要用裸名。
