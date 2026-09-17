# agent_harness/pd/ — PD（prefill/decode 分離）推論子系統

**這不是 agent harness 的兩條迴路之一。** 它住在 `agent_harness/` 底下是歷史原因
（2026-09-16 整併 0907 fusionroutemot 時一起進來的），不是設計。

## 它是什麼（實測）

35 個追蹤檔，內容指向一個**分散式推論**子系統：

| 檔案 | 角色 |
|---|---|
| `edge_server.py` | 邊緣端的 OpenAI-compatible 服務 |
| `master_slave_sync.py`、`coordinator.py`、`discovery.py`、`protocol.py` | 節點協調與探索 |
| `kv_async_prefetch.py`、`kv_quantizer.py`、`collect_batch.py` | KV 的非同步預取／量化／批次收集 |
| `train_mot_h.py` | 訓練側 |
| `pd_e2e_test.py`、`router_selftest.py` | 自測 |

## 誰用它

`agent_harness/qwen36/__init__.py` 的 docstring 把 **"CGC edge_server (OpenAI-compatible API,
with expert-cache + MTP + DOPD)"** 列為 Qwen3.6 推論的後端之一 ⇒ 它是**引擎側**的元件。

## 歸屬與引用（可重跑的證據）

- 追蹤檔：`git ls-files agent_harness/pd | wc -l` → 35
- 進來的**路徑**引用：1 處（`docs/AGENT_HARNESS_E1_RESTRUCTURE_20260916.html`）。
  全 repo 搜 `pd/` 這種裸字串會大量誤中，所以判準用完整路徑前綴。

## 白皮書的家在這裡

`CGC_PD_Whitepaper.md` 與 `DOPD_UPGRADE_PLAN.md` 的**權威副本在這個目錄**。
`../docs/` 底下同名的那兩份是**指標**（2026-09-17 之前它們是逐位元組相同的副本 ——
同一份文件兩份副本就是兩個真相，而分岔之後兩邊都還是能開、都看起來正常）。

## 為什麼不搬

進來的路徑引用只有 1 處，所以搬的成本幾乎是零 —— 但**沒有比這裡更好的家**：
repo 根已經有 17 個一級目錄（`CGC-main/`、`CGC_Phase2/`、`moeexpert/`…），
再開一個給 35 個檔不會讓任何人更容易找到它。
所以處置是「**標明它不是什麼**」而不是「搬家」：這一頁存在的目的就是讓下一個讀到
`agent_harness/pd/` 的人不會以為它是 harness 的一部分。
真要搬家時，該由它的 owner 決定去處，不是由 harness 這條線代為發明一個家。
