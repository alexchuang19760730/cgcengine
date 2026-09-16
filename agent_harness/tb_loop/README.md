# tb_loop — Terminal-Bench × gemma4 × prime-agent 评估-学习循环

在 M4 端侧用 **gemma4**（26B-A4B，TurboFieldfare/llama.cpp Metal）驱动
**prime-agent**（Prime Intellect 自改进 RLM agent）跑 **Terminal-Bench**，
并通过 prime-agent 的 **Continual Harness（/refine）** 做跨轮学习：
每轮失败轨迹 → 提炼成 skills/memories → 注入下一轮 → 分数对比。

这是"agent 级 TTT"：学习发生在 harness 状态（文本）而不是模型权重，M4 完全可跑。

```
round N                round N+1
   │                       ▲
   ▼                       │
┌──────────────────┐   ┌────────────────────┐
│ tb run           │   │ 更新后的 harness    │
│  容器: prime-agent│──▶│ (skills/memories/  │
│  模型: gemma4@M4 │   │  extensions)       │
└────────┬─────────┘   └────────────────────┘
         │ 失败轨迹
         ▼
   extract_failures.py ──▶ refine_harness.sh (/refine)
```

## 前置（M4）

| 组件 | 说明 |
|---|---|
| Docker Desktop | Terminal-Bench 任务沙箱必需 |
| Python 3.13 | venv + terminal-bench |
| gemma4 OpenAI 兼容 server | llama.cpp server / TurboFieldfare，监听 `0.0.0.0`（容器经 `host.docker.internal` 访问） |
| prime-agent | host 侧一份（/refine 用），容器内由 setup 脚本自动装 |

## 安装

```bash
cd tb_loop
bash scripts/setup_env.sh     # venv + terminal-bench + host prime-agent + 检查 Docker
# 修改 config.env：
#   TB_GEMMA4_BASE_URL / API_KEY / MODEL —— 指向你的 gemma4 server
#   TB_N_TASKS / TB_TASK_IDS —— 每轮任务数（学习循环先小批量）
#   TB_ROUNDS —— 轮数
```

## 运行

```bash
./run_round.sh            # 从 round 1 跑 TB_ROUNDS 轮
./run_round.sh 2          # 从 round 2 续跑
```

每轮自动执行：评估 → 提取失败 → host 侧 /refine 更新 harness → 与上一轮对比。
结果在 `results/round_N/`（含 `failures.json`），对比表直接打印在终端。

## 目录

```
tb_loop/
├── config.env                     # 所有参数
├── run_round.sh                   # 主入口（评估→学习→对比）
├── agents/
│   ├── prime_agent_adapter.py     # tb 自定义 installed agent（tb_loop.agents.prime_agent_adapter:PrimeAgentAgent）
│   └── prime-agent-setup.sh       # 容器内安装 prime-agent
├── learning/
│   ├── extract_failures.py        # 解析 tb 结果 → failures.json
│   ├── refine_harness.sh          # host 侧 /refine（失败→skill/记忆）
│   └── compare_rounds.py          # 两轮分数对比表
├── harness/                       # ★ 跨轮学习状态（会被注入容器）
│   └── extensions/gemma4-provider.ts  # gemma4 的 OpenAI 兼容 provider
├── results/                       # 每轮输出
└── scripts/setup_env.sh           # M4 一次性环境准备
```

## 生成 SFT/LoRA 训练数据（freebuff2api × codebuff agent）

用 codebuff 云端 agent（经 freebuff2api）在 Terminal-Bench 任务上迭代执行，
只保留**成功轨迹**，整理成 MLX-LoRA 可用的 SFT 数据集：

```bash
./gen_sft.sh                 # 1) tb run（CodebuffApiAgent）→ 2) build_sft_dataset.py
```

产物：`sft_data/train.jsonl` / `valid.jsonl`，每条是 messages 对话格式：
`system`（终端 agent 提示词）→ `user`（任务指令 + 逐步终端观察）→ `assistant`（模型的原生 DSML 工具调用回复）。

```bash
# 用 MLX-LoRA 微调 gemma4：
mlx_lm.lora --train \
  --model <gemma4 基础模型> \
  --data sft_data \
  --iters 200 --batch-size 1 --lora-layers 8
```

说明：
- `SFT_API_BASE_URL` 指向跑 freebuff2api 的机器（M4 上跑时填那台机器的局域网 IP）；
  `SFT_API_KEY` 必须等于 freebuff2api `.env` 的 `FREEBUFF_API_KEY`。
- `CodebuffApiAgent` 直接解析模型的原生 DSML 工具调用（`terminal_command`/`bash`/`exec` 等），
  不需要容器内装任何东西——轨迹是模型自然行为，训练分布对齐。
- 每次调用消耗 freebuff 免费会话额度（与之前验证一致），注意配额。

## 手动跑单轮（不用 run_round.sh）

```bash
PYTHONPATH=tb_loop .venv/bin/python -m terminal_bench.cli.tb run \
  -d terminal-bench-core==0.1.1 \
  --agent-import-path tb_loop.agents.prime_agent_adapter:PrimeAgentAgent \
  -m openai/gemma-4-26b-a4b-it \
  -k model_name=gemma-4-26b-a4b-it -k api_key=sk-local \
  -k base_url=http://host.docker.internal:1234/v1 \
  -k harness_dir=tb_loop/harness \
  -k max_turns=12 -k max_tokens=30000 -k timeout_ms=600000 \
  --n-tasks 10 --output-path results/round_1 --run-id round_1
```

## 工作原理

1. **适配器**：`PrimeAgentAgent` 继承 tb 的 `AbstractInstalledAgent`。tb 把
   `prime-agent-setup.sh` 复制进任务容器安装 prime-agent，然后在容器 tmux 里以
   headless 模式跑：
   `prime-agent -p --offline --model local-gemma4/<model> --autonomous ... "<指令>"`
2. **harness 注入**：`perform_task` 先把 `tb_loop/harness/` 打包打进容器
   `$PRIME_AGENT_CODING_AGENT_DIR`（含 provider extension 与历轮 skill）。
3. **模型接入**：`harness/extensions/gemma4-provider.ts` 注册 `local-gemma4`
   provider，端点从 `TB_GEMMA4_BASE_URL` 等环境变量读取，指向宿主 M4。
4. **学习**：失败轨迹在 host 上喂给 `prime-agent /refine`，产出证据驱动的
   skill/记忆，写回 `tb_loop/harness/`，下一轮自动生效。

## 常见问题

- **容器连不上 gemma4**：确认 server 监听 `0.0.0.0`；Docker Desktop for Mac 默认支持
  `host.docker.internal`。改 `TB_GEMMA4_BASE_URL` 为局域网 IP 也可。
- **每任务安装较慢**：容器内装 prime-agent 需要 1-2 分钟/任务（Node + release 包）。
  建议先 `--n-tasks 5` 小批量验证链路，再放大。
- **`--model` 参数**：CLI 的 `-m openai/<model>` 只是给 tb 记录用；真实模型由
  adapter 的 `-k model_name` / provider extension 决定。
- **agent 没跑到预期轮数**：调大 `TB_MAX_TURNS/TOKENS/TIMEOUT_MS`；
  tb 侧 `--global-agent-timeout-sec` 可兜底。
- **/refine 无效果**：确认 `PRIME_AGENT_CODING_AGENT_DIR` 指向 `tb_loop/harness`，
  且 refine 后 `harness/` 里出现新的 skill 文件（`find harness -type f`）。

## Windows 本地彩排（无 Docker/WSL 时生成真实轨迹）

M4 正式跑之前，可在 Windows 上用 Git Bash 当真实终端做一次**忠实彩排**：
真实 freebuff2api 模型调用 + 真实命令执行 + 官方 task 测试判分。
输出布局与 `tb run` 一致，直接进 `build_sft_dataset.py`。

```bash
# 1) 一次性：建 venv 装依赖（terminal-bench / pytest / numpy / pyyaml）
py -3.13 -m venv .tb-loop-venv
.tb-loop-venv/Scripts/python.exe -m pip install terminal-bench pytest numpy pyyaml

# 2) 下载数据集（registry 走 git clone；github 被墙时用 api.github.com tarball 解包，
#    已放一份在 datasets/terminal-bench-core-0.1.1/，可直接用）

# 3) 跑彩排（key 自动从 ../app/cloud/freebuff2api/.env 读取）
.tb-loop-venv/Scripts/python.exe scripts/local_rehearsal.py \
  --data-dir datasets/terminal-bench-core-0.1.1 \
  --tasks hello-world,heterogeneous-dates,grid-pattern-transform,new-encrypt-command \
  --run-id rehearsal_1 --max-steps 12

# 4) 构数据集
.tb-loop-venv/Scripts/python.exe learning/build_sft_dataset.py results/rehearsal_1 --out-dir sft_data
```

彩排与 M4 正式版的差异（诚实声明）：
- 沙箱是 Git Bash（Windows）而非 Docker 容器；`/app` 绝对路径仅在**测试侧**映射到工作区
  （任务 prompt 本身这 4 个都是相对路径，agent 行为不受影响）。
- 预置 `.bin` 垫片：`python`/`python3` 指向 venv python（Windows 的 `python3`
  是坏的 WindowsApps 桩），并按 Dockerfile 复刻个别任务的输入布局
  （如 new-encrypt-command 的 `data/` 目录与 `rencrypt` 命令）。
- 判定成功率只对选定的 Git Bash 兼容任务子集有意义；M4 上全量跑才是权威分数。

已知彩排数据集的注意点：
- 输入 token 随步骤累积（每次喂全量终端 transcript），8 步任务 input 可达 45K。
  正式批量建议给 CodebuffApiAgent 的 obs 加滑窗截断，或调小 OBS_LIMIT。

## 用 sft_data 对 gemma4 跑 MLX LoRA + 微调前后对比（M4）

MLX 只能在 Apple Silicon 上跑（这台 Windows 机器不行）。流水线已就绪，在 M4 上执行：

```bash
# 0) 一次性环境（M4）
python3 -m venv ~/.venvs/mlx && source ~/.venvs/mlx/bin/activate
pip install -U mlx-lm mlx
huggingface-cli download <MLX_MODEL>     # 或改 config.env 的 MLX_MODEL 为本地 MLX 权重目录

# 1) 准备 finetune 数据集（轨迹截断到最后 4 步，控制序列长度）
tb_loop/.tb-loop-venv/... python learning/build_sft_dataset.py \
    results/rehearsal_1 --out-dir sft_data_ft --max-steps 4

# 2) 起两个服务：base（未微调）与 ft（挂适配器）
tb_loop/finetune/serve.sh start all      # base=:1234, ft=:1235

# 3) 微调 + 前后对比（同一任务集）
cd tb_loop && bash finetune/compare_finetune.sh
```

对比脚本做的事：base 轮（原版 gemma4）→ `mlx_lm.lora` 微调（sft_data_ft，
`--mask-prompt`，LoRA 8 层，100 iters，lr 1e-5）→ ft 轮（gemma4+适配器）→
`compare_rounds.py` 逐任务 pass/fail 对比表。

**公平 A/B 的关键**：`TB_TASK_IDS` 必须固定同一个任务集（例如就是生成轨迹的
4 个任务），否则两轮随机抽的任务不同，对比无意义：

```bash
# config.env
TB_TASK_IDS="hello-world,heterogeneous-dates,grid-pattern-transform,new-encrypt-command"
```

已知限制（诚实声明）：
- **数据只有 4 条轨迹（train 3 / valid 1）**——这轮微调是"链路验证"级别的冒烟，
  期望分数提升的方差很大；要真实效果先放大彩排批量攒数据。
- gemma4 是 26B A4B MoE：mlx-lm 对 MoE 的 LoRA 会作用到 attention + expert 权重
  （SwitchLinear 每 expert 一个 LoRA），`--grad-checkpoint` 已开以省显存。
- `sft_data_ft` 里 system 角色走 gemma4 tokenizer 的 chat template；
  若模板不支持 system 会报错，届时把 build 脚本的 DEFAULT_SYSTEM 并进 TASK 即可。
- 微调前建议先 `mlx_lm.lora --help` 确认 `--num-layers`（0.31.x）还是
  `--lora-layers`（旧版）——finetune_gemma4.sh 已自动探测两种。

## 负样本开关（失败轨迹作为反例）

`build_sft_dataset.py` 默认只保留成功轨迹；加 `--include-failed` 会把失败轨迹
（含 no_command、timeout、test_failed 等）作为反例混入数据集。每条样例带
`label`（positive/negative）、`failure_mode`、`trajectory_notes`（如 no_command）
字段，正反例混合打散后一起切 train/valid。

```bash
# gen_sft.sh 一键（config.env: SFT_INCLUDE_FAILED=1）
SFT_INCLUDE_FAILED=1 ./gen_sft.sh

# 或手动
python learning/build_sft_dataset.py results/rehearsal_1 \
    --out-dir sft_data_mixed --include-failed --max-steps 4 \
    --max-failed-examples 50        # 反例上限，防止正反例失衡
```

⚠️ **诚实警告**：普通 SFT 训反例会教模型**模仿失败行为**（它只是拟合
(prompt → 行为) 的分布）。真正让模型"避开坏行为"的正确姿势：
1. **偏好训练（DPO/ORPO）**——用这里的 label 把 (成功轨迹, 失败轨迹) 配成对；
   mlx-lm 目前不直接支持 DPO 训练（可用 TRL 等库在 M4 上做）。
2. 或者在反例的回复里显式加一句"这是错误示范，不要这样做"再训。
纯 `--include-failed` 的 SFT 只适合让模型见更多失败形态（比如"模型会在这里
卡住/超时"），不要期望它自动学会规避。

注：当前 `results/rehearsal_1` 是 4/4 全过，没有失败轨迹可收——负样本要等
跑出失败任务后（放大批量/换更难任务）才真正生效。

## 上一步命令回显（减少重复命令）

`CodebuffApiAgent` 在每步的 user 终端消息末尾显式附上上一步执行的命令：

```
--- terminal (step N) ---
<obs>

[Commands you just ran: ls -la; echo hi]
```

原因：obs 是 `capture_pane()[-OBS_LIMIT:]`，命令输出过长时 `$ cmd` 回显行会被
截掉，模型看不到自己上一步跑了什么，容易重复执行"确认一下"。显式回显不可截断，
且**训练数据（build_sft_dataset.to_messages）与运行时格式逐字节一致**，
保证分布对齐。

实测（grid-pattern-transform，同任务同模型）：
- 旧版（无回显）：8 步 / 45K input token，同样的 python3 -c 测试命令跑了 4 次
- 新版（带回显）：3 步 / 5.4K input token，零重复命令，一次通过

⚠️ n=1，不能下因果结论——需要批量 A/B 才能确认是回显的功劳而非模型随机性。
`results/rehearsal_2/` 是新版跑出的轨迹（含 prev_commands 字段），
可与 `rehearsal_1/` 合并进数据集。

## refine_harness.sh 的三大改进

host 侧 /refine 蒸馏（每轮评估后提炼失败教训进 harness）有三处关键设计：

1. **喂完整轨迹**：不只是失败摘要——把 `commands.txt`、`panes/pre-agent.txt`、
   `panes/post-agent.txt`、`agent-logs/trajectory.jsonl` 尾部一并喂给 /refine
   （每个源截断到 `TB_REFINE_MAX_CHARS=6000`，轨迹摘要最多 `TB_REFINE_MAX_TRAJ=6` 条）。
   缺文件的源自动跳过（rehearsal 布局只有 trajectory 也能跑）。
2. **codebuff 云端蒸馏**：默认 `--model freebuff-codebuff/<SFT_MODEL>`，走
   `harness/extensions/freebuff-provider.ts`（freebuff2api 的 deepseek-v4-flash），
   蒸馏质量高于端侧 gemma4 自蒸馏。端点不可达时自动回退 `TB_MODEL_PATTERN`。
   M4 上跑时把 `SFT_API_BASE_URL` 改成跑 freebuff2api 那台机器的局域网 IP。
3. **显式 global + 限定类型**：prompt 以 `/refine --global` 开头（跨会话作用域），
   并硬性要求只提炼 memory/prompt（create/update），不创建 skill/subagent、
   不改写系统提示词——避免对 Terminal-Bench 场景过度设计。

`extract_failures.py` 现在给每条失败输出 `trial_dir`（指向完整轨迹所在目录），
refine 靠它定位轨迹文件；无逐任务目录时回退到汇总 results.json。

已本地验证（假 prime-agent 捕获 argv + 假 subprocess）：prompt 首行
`/refine --global`，包含全部轨迹源与限定指令；freebuff2api 不可达时
WARN 并回退 `local-gemma4/...`。

## 教训级归因验证（round N+1 自动回滚有害 refinement）

`run_round.sh` 每轮结束后自动运行（从第 2 轮起，要求两轮任务集一致）：

```
refine_harness.sh 快照       → round_N/harness_state.{before,after}.json + refinements.{before,after}.jsonl
attribution.py               → round_N/attribution.json（条目级 + refinement 级归因）
rollback_harmful.sh          → （RUN_ROLLBACK=1 时）自动 /refine rollback <id> --global
```

- **归因逻辑**（`learning/attribution.py`）：统计每个 changed entry 的 improve（round1 败→round2 过）与 regress（round1 过→round2 败）任务数，按 `--min-observations`（默认 2，config 里 `TB_MIN_OBSERVATIONS`）分类 useful / harmful / mixed / useless；harmful 要求 `regress >= min_obs 且 regress > improve`。
- **回滚单位是 refinement id**（不是单条目）：prime-agent 的 `/refine rollback <id> --global` 反向重放该 refinement 的所有 edits（refinement.ts rollbackProposal）。
- **默认 dry-run**：`rollback_harmful.sh report.json` 只打印；加 `--execute` 才真正调用 prime-agent（run_round.sh 里 `RUN_ROLLBACK=1` 才会自动执行）。
- **诚实声明**：这是**相关性启发不是因果**——单轮内条目变化与成败同时发生，无法证明某条目导致某成败；要采信需固定任务集、多轮累积、`--min-observations` 调大。harness 状态缺失/失败任务不足时脚本自动跳过，不阻塞循环。

## /refine 证据增强：asciinema 录制（sessions/agent.cast）

refine_harness.sh 现在喂给 /refine 5 个证据源（前 4 个是文本，第 5 个是时序）：

1. `commands.txt` 尾部（命令历史）
2. `panes/pre-agent.txt`（任务开始前终端状态）
3. `panes/post-agent.txt`（agent 结束后的终端状态）
4. `agent-logs/trajectory.jsonl` 摘要（逐步轨迹）
5. **`sessions/agent.cast`（asciinema v2 录制）**——解析出：
   - **输入时间线**：agent 何时敲了什么命令（按 `\r` 提交符切分，逐字符输入的帧会合并成命令级；ANSI 转义清理）
   - **停顿检测**：相邻输入间隔 ≥ `TB_REFINE_STALL_SEC`（默认 30s）标为"疑似卡住/等输出"，定位失败卡点

配置（config.env）：
- `TB_REFINE_MAX_CAST_INPUTS=40`：输入时间线最多显示条数
- `TB_REFINE_STALL_SEC=30`：停顿阈值（秒）

验证方式：合成 .cast（v2 格式 + ANSI 转义 + 57s/50s 停顿）实测，断言命令切分正确、停顿检出、无 ANSI 泄漏。

## M4 预演修复记录（Windows 侧审计发现的必炸 bug）

| Bug | 现象 | 修复 |
|---|---|---|
| config.env 用 `grep -oP`（GNU 专属） | macOS BSD grep 报 `illegal option -- P`，`source config.env` 崩溃 | 改 POSIX 写法（`grep '^KEY=' \| cut -d=`），BSD 兼容 |
| `_FREEBUFF_ENV` 引用 `TB_LOOP_DIR` 早于其定义 | config.env 独立 source 时 key 为空 | `TB_LOOP_DIR` 定义提前到文件顶部 |
| config.env 依赖 Windows 项目路径 `../app/cloud/freebuff2api/.env` | M4 上该路径不存在 | 文件不存在时回退环境变量 `SFT_API_KEY`（跑前 export 即可） |

Windows 侧已全部验证：10 个 bash 脚本 `bash -n`、8 个 py 编译、adapter 通过 AgentFactory 加载、tb 0.2.18 CLI 参数全部存在、config.env 独立 source 正常。

## 批量放大彩排（rehearsal_b1，2026-08-16）

目标 20-30 任务，实际新增 8 个候选任务。**成功 0 新增**——两个硬阻：

1. **freebuff 账户限流**（429 spend_limited，flagged for VPN/proxy usage，07:00 UTC 重置）：中途模型调用全被拒，无法继续批量生成
2. **模型能力边界**：deepseek-v4-flash 在 easy 任务上表现好（4/4），medium+ 纯文件任务上要么长生成退化（csv-to-parquet / raman-fitting / organization-json 都是 8192 token 截断后无限重复 `<DSML>invoke`）、要么理解偏差（git-workflow-hack 把"修复 token 泄漏"理解成"搭建网站"）、要么探索超时（sqlite 8 步探查截断文件）

本轮对 runner 的 4 处实质修复（后续批量可复用）：
- `--rewrite-app`：prompt 里 `/app/xxx` → `./xxx`（Git Bash 无 /app 挂载）
- **execute 格式解析器**：模型输出新增 `<DSML>execute>ls -la</DSML>/execute>` 块（之前只认 invoke+parameter）
- `TEST_DIR` 环境变量：tb 测试依赖它定位 src/，彩排 runner 现在正确注入
- chmod 类任务标记平台不适用（NTFS 无真实执行位，Windows 上 os.chmod 加 x 无效）

数据集现状：`sft_data_ft` = **17 条轨迹（6 正 11 负）**，train 14 / valid 3，含负样本 label + failure_mode。比之前 4 条正例有量级提升，但正例仍少——微调结论要等 freebuff 恢复后补 easy 任务批量。

## 终端观察滑窗截断（CodebuffApiAgent）

控制多步任务 input token 消耗（8 步任务曾达 45K）：

1. **OBS_WINDOW=5**（默认）：对话 history 只保留最近 5 步完整消息，更早的折叠成一行命令摘要
   `[Earlier steps: cmd1 | cmd2 | ...]`——模型始终能看到最新终端状态 + 命令历史
2. **OBS_INCREMENTAL=1**（默认关）：只发上一步之后新增的终端输出，实测累计降幅 61%、峰值 -74%；行为风险（模型看不到早期输出）需真实 API 验证后再开

**关键修复（本轮顺带发现）**：DSML 工具调用解析器对闭合标签有 bug——invoke/parameter 的闭合是 `</\uff5c\uff5cDSML...invoke>`（`<` 后是 `/`），execute 的闭合是 `<\uff5c\uff5cDSML.../execute>`（`<` 后是竖线），两者格式不同。修复后重扫全部 78 条真实轨迹，命令提取 78/78 与运行时记录一致。

模拟实测（grid-pattern 8 步真实轨迹重放）：
| 策略 | 累计发送 | 峰值 |
|---|---|---|
| 旧（全量累积） | 28,020 | 9,503 |
| 窗口折叠 | 26,942 | 8,690 |
| 窗口 + 增量 | **10,997** | **2,490** |
