# CGC Agent 产品介绍与使用手册

## 一、产品简介

`CGC Agent` 是 `CGC` 面向 `Agent` 产品链的统一命令入口，用于把以下能力收口成一条正式可操作、可验证、可审计、可回放、可回溯的工作流：

- `workflow DAG` 导入
- `GUI agent` 示教采集与客户实战录制
- 云侧 `cloud_supervised_plus_q2rl` 训练
- 端侧 `pure_llm_six_element_inference` 推理
- 三方比较可视化
- 审计、回放、回溯、追踪

它对应 `UPKG 3.8` 的正式产品能力，核心目标是把：

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

变成一套标准 CLI 入口，而不是只停留在 gate artifact 或内部验证脚本层。

## 二、产品定位

`CGC Agent` 解决的是四类问题：

1. 如何把业务工作流导入为标准 DAG，并插入到计算图里
2. 如何把 GUI 示教记录变成正式训练数据与训练证据
3. 如何把训练结果推到 edge 侧执行推理
4. 如何对示教、训练、推理结果进行比较、审计、回放和回溯

这套能力同时支持两条 CLI：

- release CLI：

```bash
python3 app/cli/cgc.py agent ...
```

- engine CLI：

```bash
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent ...
```

两条 CLI 共享同一套 artifact contract，输出结构一致。

## 三、适用场景

`CGC Agent` 适用于以下场景：

- 已经有业务流程 JSON / DAG，希望导入 `CGC`
- 希望通过 GUI 示教采集操作轨迹
- 希望把示教结果转成云侧训练和 `Q2RL` 优化
- 希望把训练后模型下推到 edge 侧执行推理
- 希望比较示教结果、优化前结果、优化后结果
- 希望对整条链路做审计、回放、回溯

## 四、标准使用流程

推荐按以下顺序使用：

1. `import-dag`
2. `teach`
3. `train`
4. `infer`
5. `visualize`
6. `compare`
7. `audit`
8. `replay`
9. `trace`

如果你只关心快速闭环，最小流程可以是：

1. `import-dag`
2. `teach`
3. `train`
4. `visualize`

## 五、模式说明

`CGC Agent` 的示教与训练分为两种模式：

- `development`：
  - 面向开发验证、功能联调、gate rerun
  - 允许直接使用已有 `gui_runtime_evidence.json`
  - 允许使用 `--gui-duration-s` 做短时 GUI 采样
  - 适合本地快速验证 `teach -> train -> visualize`
- `customer`：
  - 面向真实客户场景
  - 必须提供真实录屏文件
  - 必须提供键盘/鼠标事件文件
  - 后续训练、比较、审计、回放、回溯都以这份真实证据链为基础

客户实战模式至少要求这 4 类输入形成同一 `session_id` 闭环：

- `screen_recording.mp4`
- `keyboard_mouse_events.jsonl`
- `screenshot_manifest.json`
- `gui_agent_runtime_evidence.json`

其中：

- 录屏用于完整回放
- 键盘/鼠标事件用于动作重建
- screenshot manifest 用于 step 对位与错误定位
- GUI runtime evidence 用于汇总会话元数据与路径索引

## 六、快速开始

### 6.1 Release CLI

开发模式：

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

客户实战模式：

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

开发模式：

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

客户实战模式：

```bash
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent import-dag --dag-file /path/to/workflow.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent teach --teaching-mode customer --dag-file /path/to/workflow.json --screen-recording-path /path/to/screen_recording.mp4 --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl --gui-evidence-path /path/to/gui_agent_runtime_evidence.json
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent train --teach-session /path/to/agent_teach_session.json --teaching-mode customer --screen-recording-path /path/to/screen_recording.mp4 --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl
python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent visualize --train-session /path/to/agent_train_session.json
```

## 七、命令详解

### 7.1 `cgc agent import-dag`

**作用**

将业务工作流 JSON 或 DAG 导入为 `CGC Agent` 可识别的标准结构，并生成计算图插入契约。

**典型用途**

- 把业务流程从文档/JSON 提升为正式工作流资产
- 让工作流后续可进入示教、训练、推理链

**常用参数**

- `--dag-file`：工作流 JSON 文件路径，必填
- `--dag-name`：可选，覆盖 DAG 逻辑名称
- `--output-dir`：可选，指定输出目录
- `--json`：以 JSON 打印结果

**主要输出**

- `agent_dag_workflow_manifest.json`
- `agent_graph_insertion_contract.json`
- `agent_subterranean_compile_plan.json`

**示例**

```bash
python3 app/cli/cgc.py agent import-dag \
  --dag-file /path/to/workflow.json \
  --output-dir /tmp/agent_import \
  --json
```

### 7.2 `cgc agent teach`

**作用**

采集 GUI 示教证据，形成正式 teaching session。

**典型用途**

- 记录 GUI agent 的实际操作轨迹
- 将 GUI runtime evidence 变为可训练、可回放的正式输入

**模式说明**

- `development`：
  - 允许使用 `--gui-duration-s`
  - 允许使用 `--gui-evidence-path`
  - 适合开发调试、短时验证、gate smoke/rerun
- `customer`：
  - 必须提供 `--screen-recording-path`
  - 必须提供 `--keyboard-mouse-events-path`
  - 建议同时提供 `--gui-evidence-path`
  - 目标是形成真实客户会话的闭环证据链

**常用参数**

- `--dag-file`：可选，先导入 DAG 再开始示教
- `--dag-manifest`：可选，使用已有 DAG manifest
- `--dag-name`：可选，覆盖 DAG 名称
- `--teaching-mode`：`development` 或 `customer`
- `--gui-duration-s`：开发模式下的示教录制时长；客户模式下只作为补充采样
- `--gui-evidence-path`：使用已有 GUI evidence，而不是重新录制
- `--screen-recording-path`：客户模式下的完整录屏文件
- `--keyboard-mouse-events-path`：客户模式下的键盘/鼠标事件文件
- `--output-dir`
- `--json`

**主要输出**

- `agent_teach_session.json`
- `agent_teach_trace.json`
- `agent_teach_replay_bundle.json`

**示例**

```bash
python3 app/cli/cgc.py agent teach \
  --dag-file /path/to/workflow.json \
  --teaching-mode development \
  --gui-duration-s 5 \
  --output-dir /tmp/agent_teach \
  --json
```

```bash
python3 app/cli/cgc.py agent teach \
  --dag-file /path/to/workflow.json \
  --teaching-mode customer \
  --screen-recording-path /path/to/screen_recording.mp4 \
  --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl \
  --gui-evidence-path /path/to/gui_agent_runtime_evidence.json \
  --output-dir /tmp/agent_teach_customer \
  --json
```

### 7.3 `cgc agent train`

**作用**

启动 `UPKG 3.8` 的正式训练链，把示教数据转成云侧训练、`Q2RL` 优化结果与 edge 可部署产物。

**典型用途**

- 用示教数据训练目标模型
- 生成 `UI-TARS` 的训练后模型 manifest
- 输出 `Q2RL` 指标、推理契约和比较 artifact

**模式说明**

- `development`：
  - 可以直接沿用开发期 `teach` 产物
  - 适合 rerun、流程联调和算法验证
- `customer`：
  - 要求训练输入对应真实客户录屏与键盘/鼠标事件
  - 训练 session 中会保留 `capture_contract`
  - 后续 `audit/replay/trace` 也沿用这套证据契约

**常用参数**

- `--teach-session`：已有 `teach` 会话路径
- `--dag-file`：可选，训练前直接导入 DAG
- `--dag-manifest`：可选，使用已有 DAG manifest
- `--teaching-mode`：`development` 或 `customer`
- `--gui-duration-s`：无 teaching session 时直接采集 GUI evidence，仅建议开发模式使用
- `--gui-evidence-path`：使用已有 GUI evidence
- `--screen-recording-path`：客户模式下的完整录屏文件
- `--keyboard-mouse-events-path`：客户模式下的键盘/鼠标事件文件
- `--output-dir`
- `--json`

**主要输出**

- `agent_train_session.json`
- `agent_subterranean_bundle.json`
- `agent_model_graph_insertion_contract.json`
- `upkg38/...` 全套正式产物

**2026-06-20 本地里程碑覆盖补充**

- 已用 release CLI 对 `upkg30 ~ upkg38` 全部正式里程碑做过一轮本地重跑。
- 本地重跑输出目录：`temp/test/upkg3x_rerun_20260620/`
- 这轮结果中 `upkg30 / upkg31 / upkg32 / upkg33 / upkg34 / upkg35 / upkg36 / upkg37 / upkg38` 全部为 `PASS`。
- 若要回看每个 gate 的原始证据，可直接读取对应的 `upkg3x_rerun_20260620/*.stdout.txt`。

**关键训练产物**

- `teaching_dataset_manifest.json`
- `teaching_trained_model_manifest.json`
- `q2rl_training_report.json`
- `edge_inference_push_contract.json`
- `llm_six_element_inference_mode.json`

**示例**

```bash
python3 app/cli/cgc.py agent train \
  --teach-session /path/to/agent_teach_session.json \
  --teaching-mode development \
  --output-dir /tmp/agent_train \
  --json
```

```bash
python3 app/cli/cgc.py agent train \
  --teach-session /path/to/agent_teach_session.json \
  --teaching-mode customer \
  --screen-recording-path /path/to/screen_recording.mp4 \
  --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl \
  --output-dir /tmp/agent_train_customer \
  --json
```

### 7.4 `cgc agent infer`

**作用**

生成端侧推理所需的 infer session 与部署索引。

**典型用途**

- 将训练结果转换为 edge 侧可用推理入口
- 索引 `pure_llm_six_element_inference` 与 edge push contract

**常用参数**

- `--train-session`
- `--artifact-root`
- `--output-dir`
- `--json`

**主要输出**

- `agent_infer_session.json`

**关联产物**

- `llm_six_element_inference_mode.json`
- `edge_inference_push_contract.json`
- `cloud_summary.json`

**示例**

```bash
python3 app/cli/cgc.py agent infer \
  --train-session /path/to/agent_train_session.json \
  --output-dir /tmp/agent_infer \
  --json
```

### 7.5 `cgc agent visualize`

**作用**

生成统一可视化索引，收口三方比较图、错误图、HTML 可视化页。

**典型用途**

- 快速打开比较结果
- 快速查看错误可视化结果

**常用参数**

- `--train-session`
- `--artifact-root`
- `--output-dir`
- `--json`

**主要输出**

- `agent_visualization_index.json`
- `agent_visualization_index.html`

**关联产物**

- `teaching_optimization_triplet_comparison.json`
- `before_vs_after_vs_teaching_chart.json`
- `triplet_comparison.mmd`
- `triplet_comparison.html`
- `graph_error_visualization.json`
- `graph_error_visualization.mmd`

**示例**

```bash
python3 app/cli/cgc.py agent visualize \
  --train-session /path/to/agent_train_session.json \
  --output-dir /tmp/agent_visualize \
  --json
```

### 7.6 `cgc agent compare`

**作用**

对 `Teaching / Pre-Q2RL / Post-Q2RL` 进行三方比较，并输出关键指标摘要。

**典型用途**

- 验证优化前后差异
- 查看 `reward`、`alignment`、`distance_to_teaching`

**常用参数**

- `--train-session`
- `--artifact-root`
- `--output-dir`
- `--json`

**主要输出**

- `agent_compare_session.json`

**核心指标**

- `reward_gain`
- `alignment_gain`
- `distance_to_teaching_after_q2rl`
- `overlay_status`

**示例**

```bash
python3 app/cli/cgc.py agent compare \
  --train-session /path/to/agent_train_session.json \
  --output-dir /tmp/agent_compare \
  --json
```

### 7.7 `cgc agent audit`

**作用**

查看整条链路的正式审计摘要。

**典型用途**

- 检查训练、推理、比较是否具备正式审计证据
- 查看可比性、可追溯性、可回放性

**常用参数**

- `--train-session`
- `--artifact-root`
- `--output-dir`
- `--json`

**主要输出**

- `agent_audit_session.json`

**关联维度**

- `auditability`
- `comparability`
- `replayability`
- `traceability`

**示例**

```bash
python3 app/cli/cgc.py agent audit \
  --train-session /path/to/agent_train_session.json \
  --output-dir /tmp/agent_audit \
  --json
```

### 7.8 `cgc agent replay`

**作用**

查看 replay 锚点与重播相关元数据。

**典型用途**

- 确认 GUI teaching 和 edge inference 是否具备重播入口
- 查看 `replay_anchor` 与关键 evidence 路径

**常用参数**

- `--train-session`
- `--artifact-root`
- `--output-dir`
- `--json`

**主要输出**

- `agent_replay_session.json`

**关联产物**

- `replay_anchor.json`
- `gui_agent_runtime_evidence.json`
- `stage_trace.jsonl`

**示例**

```bash
python3 app/cli/cgc.py agent replay \
  --train-session /path/to/agent_train_session.json \
  --output-dir /tmp/agent_replay \
  --json
```

### 7.9 `cgc agent trace`

**作用**

查看 stage trace 与 GUI event trace，用于最细粒度的链路追踪和问题排查。

**典型用途**

- 排查某个 stage 失败原因
- 对位 GUI event 与训练/推理阶段
- 进行回溯分析

**常用参数**

- `--train-session`
- `--artifact-root`
- `--output-dir`
- `--json`

**主要输出**

- `agent_trace_session.json`

**常见字段**

- `stage_trace_count`
- `gui_event_count`
- `stage_trace_preview`
- `gui_event_preview`

**示例**

```bash
python3 app/cli/cgc.py agent trace \
  --train-session /path/to/agent_train_session.json \
  --output-dir /tmp/agent_trace \
  --json
```

## 八、输入输出关系

命令之间的衔接关系如下：

- `import-dag` 的输出可供 `teach` 和 `train` 使用
- `teach` 的输出可直接供 `train` 使用
- `train` 的输出可直接供 `infer / visualize / compare / audit / replay / trace` 使用

推荐的最常见关系是：

```text
import-dag
  ->
teach
  ->
train
  ->
infer / visualize / compare / audit / replay / trace
```

## 九、标准产物说明

在 `CGC Agent` 正式产品链中，最常见的关键产物包括：

- DAG 类：
  - `agent_dag_workflow_manifest.json`
  - `agent_graph_insertion_contract.json`
- Teaching 类：
  - `agent_teach_session.json`
  - `agent_teach_trace.json`
  - `agent_teach_replay_bundle.json`
  - `capture_contract.screen_recording_path`
  - `capture_contract.keyboard_mouse_events_path`
- Training 类：
  - `agent_train_session.json`
  - `agent_subterranean_bundle.json`
  - `agent_model_graph_insertion_contract.json`
  - `q2rl_training_report.json`
- Inference 类：
  - `agent_infer_session.json`
  - `llm_six_element_inference_mode.json`
  - `edge_inference_push_contract.json`
- Visualization / Compare 类：
  - `agent_visualization_index.json`
  - `agent_visualization_index.html`
  - `agent_compare_session.json`
  - `triplet_comparison.html`
  - `before_vs_after_vs_teaching_chart.json`
- Audit / Replay / Trace 类：
  - `agent_audit_session.json`
  - `agent_replay_session.json`
  - `agent_trace_session.json`

## 十、最佳实践建议

1. 先用 `import-dag` 标准化工作流，再进入示教和训练
2. 开发验证优先使用 `development` 模式，真实客户场景统一使用 `customer` 模式
3. 客户实战模式下必须保留原始录屏与键盘/鼠标事件，避免只有聚合报告
4. `train` 结束后优先执行 `visualize` 与 `compare`，先看结果质量
5. 在正式验收前，至少执行一次 `audit / replay / trace`
6. 对外展示时优先使用 `agent_visualization_index.html`

## 十一、常见问题

### 11.1 我只有 workflow.json，没有 GUI 证据，可以直接训练吗？

可以，但通常仍建议先经过 `teach` 或提供 `--gui-evidence-path`，这样训练链的示教来源更完整。

### 11.2 我已经有一份旧的 GUI evidence，还需要重录吗？

不一定。你可以直接通过 `--gui-evidence-path` 注入已有 evidence。

### 11.3 release CLI 和 engine CLI 有什么差异？

两者 artifact contract 一致。差异主要在定位：

- release CLI 更偏产品使用入口
- engine CLI 更偏引擎侧工作流入口

### 11.4 `compare`、`audit`、`replay`、`trace` 都要跑吗？

不是必须全部都跑，但正式产品验收时建议至少保留：

- `compare`
- `audit`
- `replay`
- `trace`

### 11.5 客户场景为什么不能只用 `--gui-duration-s`？

因为 `--gui-duration-s` 主要是开发期短时采样入口，不能保证形成完整客户证据链。真实客户场景至少需要：

- 完整录屏
- 键盘/鼠标事件
- GUI evidence 汇总
- step/screenshot 对位

否则后续训练、比较、回放、审计都会缺少闭环证据。

## 十二、最小实战模板

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

## 十三、总结

`CGC Agent` 并不是单一命令，而是一条完整产品链的统一 CLI 收口。

它把：

- 工作流导入
- GUI 示教
- 云侧训练
- 端侧推理
- 可视化比较
- 审计回放回溯

统一成一套正式、稳定、可追踪的产品使用流程。

对于 `UPKG 3.8` 而言，`cgc agent` 已经不仅是验证入口，而是正式的产品级用户操作入口。
