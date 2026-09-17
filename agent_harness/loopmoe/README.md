# agent_harness/loopmoe/ — Loop MoE 的模型與訓練套件

**它不是 agent harness 的迴路，但它被 harness 的迴路依賴。** 這個區別很重要，而且
2026-09-17 之前没有人把它寫下來 —— 而那一天正好有一支依賴它的腳本壞掉，卻沒有人發現。

## 它是什麼

22 個追蹤檔，是一個 Python 套件（有 `__init__.py`）：

```
config.py                        LoopMoEConfig
models/                          loop_moe_model.py 等
training/lora.py                 LoRA
training/weight_graft.py         graft_qwen36_weights()（把 Qwen3.6 的權重接上去）
training/train_loopmoe.py        訓練主迴圈
tests/                           4 個測試（雙狀態、遞迴整合、config、models）
lti/                             LTI 相關
configs/                         loopmoe_35b.json（訓練設定）
```

## 誰用它（實測，不是推論）

| 消費者 | 用法 |
|---|---|
| `tb_loop/finetune/finetune_loopmoe.sh` | `from loopmoe.config import LoopMoEConfig`、`from loopmoe.models.loop_moe_model import LoopMoEModel`、`from loopmoe.training.weight_graft import graft_qwen36_weights` |
| `tb_loop/agents/loopmoe_agent_adapter.py` | Terminal-Bench 的 agent 走 Loop MoE |

## ★ 為什麼它住在**上一層**，以及那造成的一個坑

`finetune_loopmoe.sh` 在 `tb_loop/finetune/`，而它需要的東西**不在同一層**：

```
tb_loop/config.env、tb_loop/sft_data_merged/     ← 跟腳本一起搬進 tb_loop（2026-09-16 E1）
agent_harness/loopmoe/、agent_harness/loopmoe_output/   ← 沒搬（就是這個目錄）
```

E1 的搬遷讓那支腳本的 `AGENT_HARNESS_DIR="$(dirname "$SCRIPT_DIR")"` 從 `agent_harness/`
變成 `tb_loop/`。於是 `config.env` 仍然找得到（它也在 `tb_loop/`）而 `loopmoe/` 找不到 ⇒
**失敗是靜默的，因為同一個變數底下有一半的派生仍然解析得到。** 修法是把錨點拆成兩個名字
並互相對照，2026-09-17 完成（見該腳本第 17 行附近的說明）。

⇒ **這個目錄的存在本身就是一份文件**：它記錄了「哪些東西沒跟 tb_loop 一起搬」。

## 引用廣度

- 追蹤檔：`git ls-files agent_harness/loopmoe | wc -l` → 22
- 進來的**路徑**引用（`agent_harness/loopmoe/`）：1 處（`docs/AGENT_HARNESS_E1_RESTRUCTURE_20260916.html`）
- 注意：全 repo 搜裸字串 `loopmoe` 有 14 處，其中多數指的是**這個套件**
  （因為 `config.env` 的 `LOOPMOE_*` 變數名就是它），但那是變數名不是路徑。
  判準要用完整路徑前綴。
