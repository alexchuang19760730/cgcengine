# CGC Agent 產品介紹與使用手冊

## 一、產品簡介

`CGC Agent` 是 `CGC` 面向 `Agent` 產品鏈的統一命令入口，用來把以下能力收口成一條正式可操作、可驗證、可審計、可回放、可回溯的工作流：

- `workflow DAG` 匯入
- `GUI agent` 示教採集與客戶實戰錄製
- 雲側 `cloud_supervised_plus_q2rl` 訓練
- 端側 `pure_llm_six_element_inference` 推理
- 三方比較可視化
- 審計、回放、回溯、追蹤

它對應 `UPKG 3.8` 的正式產品能力，核心目標是把：

```text
workflow DAG
  ->
GUI teaching
  ->
cloud_supervised_plus_q2rl
  ->
edge pure_llm_six_element_inference
  ->
compare / audit / replay / trace
```

變成一套標準 CLI 入口，而不是只停留在 gate artifact 或內部驗證腳本層。

## 二、產品定位

`CGC Agent` 主要解決四類問題：

1. 如何把業務工作流匯入為標準 DAG，並插入到計算圖裡
2. 如何把 GUI 示教記錄變成正式訓練資料與訓練證據
3. 如何把訓練結果下推到 edge 側執行推理
4. 如何對示教、訓練、推理結果進行比較、審計、回放與回溯

這套能力同時支援兩條 CLI：

- release CLI：

```bash
python3 app/cli/cgc.py agent ...
```

- engine CLI：

```bash
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent ...
```

兩條 CLI 共用同一套 artifact contract，輸出結構一致。

## 三、適用場景

`CGC Agent` 適用於以下場景：

- 已經有業務流程 JSON / DAG，希望匯入 `CGC`
- 希望透過 GUI 示教採集操作軌跡
- 希望把示教結果轉成雲側訓練與 `Q2RL` 優化
- 希望把訓練後模型下推到 edge 側執行推理
- 希望比較示教結果、優化前結果、優化後結果
- 希望對整條鏈路做審計、回放、回溯

## 四、標準使用流程

建議按以下順序使用：

1. `import-dag`
2. `teach`
3. `train`
4. `infer`
5. `visualize`
6. `compare`
7. `audit`
8. `replay`
9. `trace`

如果你只關心快速閉環，最小流程可以是：

1. `import-dag`
2. `teach`
3. `train`
4. `visualize`

## 五、模式說明

`CGC Agent` 的示教與訓練分為兩種模式：

- `development`
  - 面向開發驗證、功能聯調、gate rerun
  - 允許直接使用既有 `gui_runtime_evidence.json`
  - 允許使用 `--gui-duration-s` 做短時 GUI 採樣
  - 適合本地快速驗證 `teach -> train -> visualize`
- `customer`
  - 面向真實客戶場景
  - 必須提供真實錄屏檔
  - 必須提供鍵盤/滑鼠事件檔
  - 後續訓練、比較、審計、回放、回溯都以這份真實證據鏈為基礎

客戶實戰模式至少要求這 4 類輸入形成同一個 `session_id` 閉環：

- `screen_recording.mp4`
- `keyboard_mouse_events.jsonl`
- `screenshot_manifest.json`
- `gui_agent_runtime_evidence.json`

其中：

- 錄屏用於完整回放
- 鍵盤/滑鼠事件用於動作重建
- screenshot manifest 用於 step 對位與錯誤定位
- GUI runtime evidence 用於彙總會話中繼資料與路徑索引

## 六、快速開始

### 6.1 Release CLI

開發模式：

```bash
python3 app/cli/cgc.py agent import-dag --dag-file /path/to/workflow.json
python3 app/cli/cgc.py agent teach --teaching-mode development --dag-file /path/to/workflow.json --gui-duration-s 5
python3 app/cli/cgc.py agent train --teach-session /path/to/agent_teach_session.json --teaching-mode development
python3 app/cli/cgc.py agent infer --train-session /path/to/agent_train_session.json
python3 app/cli/cgc.py agent visualize --train-session /path/to/agent_train_session.json
python3 app/cli/cgc.py agent compare --train-session /path/to/agent_train_session.json
python3 app/cli/cgc.py agent audit --train-session /path/to/agent_train_session.json
python3 app/cli/cgc.py agent replay --train-session /path/to/agent_train_session.json
python3 app/cli/cgc.py agent trace --train-session /path/to/agent_train_session.json
```

客戶實戰模式：

```bash
python3 app/cli/cgc.py agent import-dag --dag-file /path/to/workflow.json
python3 app/cli/cgc.py agent teach --teaching-mode customer --dag-file /path/to/workflow.json --screen-recording-path /path/to/screen_recording.mp4 --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl --gui-evidence-path /path/to/gui_agent_runtime_evidence.json
python3 app/cli/cgc.py agent train --teach-session /path/to/agent_teach_session.json --teaching-mode customer --screen-recording-path /path/to/screen_recording.mp4 --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl
python3 app/cli/cgc.py agent visualize --train-session /path/to/agent_train_session.json
python3 app/cli/cgc.py agent audit --train-session /path/to/agent_train_session.json
python3 app/cli/cgc.py agent replay --train-session /path/to/agent_train_session.json
python3 app/cli/cgc.py agent trace --train-session /path/to/agent_train_session.json
```

### 6.2 Engine CLI

開發模式：

```bash
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent import-dag --dag-file /path/to/workflow.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent teach --teaching-mode development --dag-file /path/to/workflow.json --gui-duration-s 5
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent train --teach-session /path/to/agent_teach_session.json --teaching-mode development
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent infer --train-session /path/to/agent_train_session.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent visualize --train-session /path/to/agent_train_session.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent compare --train-session /path/to/agent_train_session.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent audit --train-session /path/to/agent_train_session.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent replay --train-session /path/to/agent_train_session.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent trace --train-session /path/to/agent_train_session.json
```

客戶實戰模式：

```bash
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent import-dag --dag-file /path/to/workflow.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent teach --teaching-mode customer --dag-file /path/to/workflow.json --screen-recording-path /path/to/screen_recording.mp4 --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl --gui-evidence-path /path/to/gui_agent_runtime_evidence.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent train --teach-session /path/to/agent_teach_session.json --teaching-mode customer --screen-recording-path /path/to/screen_recording.mp4 --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent visualize --train-session /path/to/agent_train_session.json
```

## 七、命令詳解

### 7.1 `cgc agent import-dag`

**作用**

將業務工作流 JSON 或 DAG 匯入為 `CGC Agent` 可辨識的標準結構，並生成計算圖插入契約。

**典型用途**

- 把業務流程從文件/JSON 提升為正式工作流資產
- 讓工作流後續可進入示教、訓練、推理鏈

**常用參數**

- `--dag-file`：工作流 JSON 檔案路徑，必填
- `--dag-name`：可選，覆寫 DAG 邏輯名稱
- `--output-dir`：可選，指定輸出目錄
- `--json`：以 JSON 印出結果

**主要輸出**

- `agent_dag_workflow_manifest.json`
- `agent_graph_insertion_contract.json`
- `agent_subterranean_compile_plan.json`

### 7.2 `cgc agent teach`

**作用**

採集 GUI 示教證據，形成正式 teaching session。

**模式說明**

- `development`
  - 允許使用 `--gui-duration-s`
  - 允許使用 `--gui-evidence-path`
  - 適合開發除錯、短時驗證、gate smoke/rerun
- `customer`
  - 必須提供 `--screen-recording-path`
  - 必須提供 `--keyboard-mouse-events-path`
  - 建議同時提供 `--gui-evidence-path`
  - 目標是形成真實客戶會話的閉環證據鏈

**常用參數**

- `--dag-file`
- `--dag-manifest`
- `--dag-name`
- `--teaching-mode`
- `--gui-duration-s`
- `--gui-evidence-path`
- `--screen-recording-path`
- `--keyboard-mouse-events-path`
- `--output-dir`
- `--json`

**主要輸出**

- `agent_teach_session.json`
- `agent_teach_trace.json`
- `agent_teach_replay_bundle.json`

### 7.3 `cgc agent train`

**作用**

啟動 `UPKG 3.8` 的正式訓練鏈，把示教資料轉成雲側訓練、`Q2RL` 優化結果與 edge 可部署產物。

**模式說明**

- `development`
  - 可直接沿用開發期 `teach` 產物
  - 適合 rerun、流程聯調與演算法驗證
- `customer`
  - 要求訓練輸入對應真實客戶錄屏與鍵盤/滑鼠事件
  - 訓練 session 中會保留 `capture_contract`
  - 後續 `audit / replay / trace` 也沿用這套證據契約

**常用參數**

- `--teach-session`
- `--dag-file`
- `--dag-manifest`
- `--dag-name`
- `--teaching-mode`
- `--gui-duration-s`
- `--gui-evidence-path`
- `--screen-recording-path`
- `--keyboard-mouse-events-path`
- `--output-dir`
- `--json`

**主要輸出**

- `agent_train_session.json`
- `agent_subterranean_bundle.json`
- `agent_model_graph_insertion_contract.json`
- `upkg38/...` 全套正式產物

**2026-06-20 本地里程碑覆蓋補充**

- 已用 release CLI 對 `upkg30 ~ upkg38` 全部正式里程碑做過一輪本地重跑。
- 本地重跑輸出目錄：`temp/test/upkg3x_rerun_20260620/`
- 這輪結果中 `upkg30 / upkg31 / upkg32 / upkg33 / upkg34 / upkg35 / upkg36 / upkg37 / upkg38` 全部為 `PASS`。
- 若要回看每個 gate 的原始證據，可直接讀對應的 `upkg3x_rerun_20260620/*.stdout.txt`。

### 7.4 `cgc agent infer`

**作用**

生成端側推理所需的 infer session 與部署索引。

### 7.5 `cgc agent visualize`

**作用**

生成統一可視化索引，收口三方比較圖、錯誤圖、HTML 可視化頁。

### 7.6 `cgc agent compare`

**作用**

對 `Teaching / Pre-Q2RL / Post-Q2RL` 進行三方比較，並輸出關鍵指標摘要。

### 7.7 `cgc agent audit`

**作用**

查看整條鏈路的正式審計摘要。

### 7.8 `cgc agent replay`

**作用**

查看 replay 錨點與重播相關中繼資料。

### 7.9 `cgc agent trace`

**作用**

查看 stage trace 與 GUI event trace，用於最細粒度的鏈路追蹤與問題排查。

## 八、輸入輸出關係

命令之間的銜接關係如下：

- `import-dag` 的輸出可供 `teach` 與 `train` 使用
- `teach` 的輸出可直接供 `train` 使用
- `train` 的輸出可直接供 `infer / visualize / compare / audit / replay / trace` 使用

## 九、標準產物說明

在 `CGC Agent` 正式產品鏈中，最常見的關鍵產物包括：

- DAG 類：
  - `agent_dag_workflow_manifest.json`
  - `agent_graph_insertion_contract.json`
- Teaching 類：
  - `agent_teach_session.json`
  - `agent_teach_trace.json`
  - `agent_teach_replay_bundle.json`
  - `capture_contract.screen_recording_path`
  - `capture_contract.keyboard_mouse_events_path`
- Training 類：
  - `agent_train_session.json`
  - `agent_subterranean_bundle.json`
  - `agent_model_graph_insertion_contract.json`
  - `q2rl_training_report.json`
- Inference 類：
  - `agent_infer_session.json`
  - `llm_six_element_inference_mode.json`
  - `edge_inference_push_contract.json`
- Visualization / Compare 類：
  - `agent_visualization_index.json`
  - `agent_visualization_index.html`
  - `agent_compare_session.json`
  - `triplet_comparison.html`
  - `before_vs_after_vs_teaching_chart.json`
- Audit / Replay / Trace 類：
  - `agent_audit_session.json`
  - `agent_replay_session.json`
  - `agent_trace_session.json`

## 十、最佳實務建議

1. 先用 `import-dag` 標準化工作流，再進入示教與訓練
2. 開發驗證優先使用 `development` 模式，真實客戶場景統一使用 `customer` 模式
3. 客戶實戰模式下必須保留原始錄屏與鍵盤/滑鼠事件，避免只有聚合報告
4. `train` 結束後優先執行 `visualize` 與 `compare`
5. 在正式驗收前，至少執行一次 `audit / replay / trace`
6. 對外展示時優先使用 `agent_visualization_index.html`

## 十一、常見問題

### 11.1 我只有 workflow.json，沒有 GUI 證據，可以直接訓練嗎？

可以，但通常仍建議先經過 `teach` 或提供 `--gui-evidence-path`，這樣訓練鏈的示教來源更完整。

### 11.2 我已經有一份舊的 GUI evidence，還需要重錄嗎？

不一定。你可以直接透過 `--gui-evidence-path` 注入既有 evidence。

### 11.3 release CLI 和 engine CLI 有什麼差異？

兩者 artifact contract 一致。差異主要在定位：

- release CLI 更偏產品使用入口
- engine CLI 更偏引擎側工作流入口

### 11.4 `compare`、`audit`、`replay`、`trace` 都要跑嗎？

不是必須全部都跑，但正式產品驗收時建議至少保留：

- `compare`
- `audit`
- `replay`
- `trace`

### 11.5 客戶場景為什麼不能只用 `--gui-duration-s`？

因為 `--gui-duration-s` 主要是開發期短時採樣入口，不能保證形成完整客戶證據鏈。真實客戶場景至少需要：

- 完整錄屏
- 鍵盤/滑鼠事件
- GUI evidence 彙總
- step / screenshot 對位

## 十二、最小實戰模板

```bash
python3 app/cli/cgc.py agent import-dag --dag-file /path/to/workflow.json --output-dir /tmp/agent_import --json
python3 app/cli/cgc.py agent teach --dag-file /path/to/workflow.json --teaching-mode customer --screen-recording-path /path/to/screen_recording.mp4 --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl --gui-evidence-path /path/to/gui_agent_runtime_evidence.json --output-dir /tmp/agent_teach --json
python3 app/cli/cgc.py agent train --teach-session /tmp/agent_teach/agent_teach_session.json --teaching-mode customer --screen-recording-path /path/to/screen_recording.mp4 --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl --output-dir /tmp/agent_train --json
python3 app/cli/cgc.py agent infer --train-session /tmp/agent_train/agent_train_session.json --output-dir /tmp/agent_infer --json
python3 app/cli/cgc.py agent visualize --train-session /tmp/agent_train/agent_train_session.json --output-dir /tmp/agent_visualize --json
python3 app/cli/cgc.py agent compare --train-session /tmp/agent_train/agent_train_session.json --output-dir /tmp/agent_compare --json
python3 app/cli/cgc.py agent audit --train-session /tmp/agent_train/agent_train_session.json --output-dir /tmp/agent_audit --json
python3 app/cli/cgc.py agent replay --train-session /tmp/agent_train/agent_train_session.json --output-dir /tmp/agent_replay --json
python3 app/cli/cgc.py agent trace --train-session /tmp/agent_train/agent_train_session.json --output-dir /tmp/agent_trace --json
```

## 十三、總結

`CGC Agent` 並不是單一命令，而是一條完整產品鏈的統一 CLI 收口。

它把：

- 工作流匯入
- GUI 示教
- 雲側訓練
- 端側推理
- 可視化比較
- 審計、回放、回溯

統一成一套正式、穩定、可追蹤的產品使用流程。

對於 `UPKG 3.8` 而言，`cgc agent` 已經不只是驗證入口，而是正式的產品級使用者操作入口。
