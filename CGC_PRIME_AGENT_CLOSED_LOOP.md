# CGC ↔ Prime Agent 闭环设计：轨迹学习驱动 MoE Streaming 优化

> 日期：2026-08-08
> 目标：让 prime-agent 从 CGC 云端 SWE agent 的运行轨迹中提炼经验，
> 反向指导下一轮优化——形成"执行 → 审计 → 学习 → 再执行"的闭环。
> 全部基于真实接口（CGC 源码 + prime-agent 源码已验证）。

---

## 1. 架构总览

```
┌─ CGC 体系（云端主导）──────────────────────────────┐
│  cgc.py ──SSH──▶ Host1/Host2 SWE agent 跑优化实验    │
│  (paramiko, 远端 python)                            │
│  轨迹落盘: <swe_agent_root>/trajectories/<任务>/     │
│    ├─ run_batch_exit_statuses.yaml   ← 退出状态     │
│    ├─ *.log                          ← 运行日志     │
│    └─ score_preview / result_files   ← 评分结果     │
└───────────────────┬────────────────────────────────┘
                    │ ① 同步（scp/rsync，见 §3）
┌───────────────────▼────────────────────────────────┐
│  Prime Agent（本地 M4，模型 = 本地 Gemma 4 26B）      │
│  ├─ /goal       承接"MoE streaming 优化"长目标        │
│  ├─ /refine     读轨迹 → 提炼经验 → 更新 harness 态   │
│  └─ skill       沉淀可复用方法（~/.prime/agent/skills/）│
└───────────────────┬────────────────────────────────┘
                    │ ② 反向指导（生成的建议/补丁）
┌───────────────────▼────────────────────────────────┐
│  下一轮 CGC 实验（改进后的配置/代码）                  │
└─────────────────────────────────────────────────────┘
```

---

## 2. 为什么这个闭环成立（能力对照）

| 环节 | CGC 提供 | prime-agent 提供 |
|---|---|---|
| 执行 | SWE agent 远程跑（paramiko） | — |
| 审计 | trajectories/ 目录 + exit_statuses.yaml（**已实现**，cgc.py:5085-5187） | — |
| 学习 | — | `/refine`：轨迹 review → 增量 create/update/delete + 快照回滚 |
| 记忆 | — | `goal` skill：全局 `~/.prime/agent/harness/` + 会话 `harness_state.json` |
| 复用 | — | skill 包：`~/.prime/agent/skills/`（Python 可执行包） |

**关键**：CGC 已有"轨迹收集"，缺"轨迹学习"；prime-agent 的 /refine 恰好补上。二者互补，无重叠。

---

## 3. 接入步骤（三步）

### 3.1 同步 CGC 轨迹到本地

**同步什么**（最小集）：
```bash
# 远端路径（cgc.py:118,4867）
REMOTE_ROOT="<host>:/root/flashkv0516/cgc_engine/SWE-agent/trajectories"
# 本地路径（prime-agent 能读即可，建议项目内）
LOCAL_ROOT=/Users/alexchuang/Documents/flashkv0516/.cgc-trajectories

# 同步（Host1 密码见 SSH config；或走已有 ControlMaster）
rsync -avz -e ssh "$REMOTE_ROOT/" "$LOCAL_ROOT/" \
  --include='*/' --include='run_batch_exit_statuses.yaml' --include='*.log' \
  --exclude='*'
```

**每个任务目录应含**：
- `run_batch_exit_statuses.yaml`：`instances_by_exit_status` → 每个 issue 的 PASS/FAIL
- 运行日志（`*.log`）：实际执行的命令、报错

### 3.2 给 prime-agent 一个"轨迹阅读 skill"

**位置**：`~/.prime/agent/skills/cgc_trajectory_reader/`（prime-agent 官方 skill 位置，见 skills.md "Locations"）

**SKILL.md 内容**（frontmatter + 用法）：
```markdown
---
name: cgc-trajectory-reader
description: 解析 CGC SWE agent 轨迹目录（run_batch_exit_statuses.yaml + 日志），
             汇总失败模式、反复错误、耗时分布。供 /refine 分析用。
---

# CGC Trajectory Reader

## 输入
- 轨迹根目录（默认 $LOCAL_ROOT，可用参数覆盖）

## 输出（JSON 摘要）
{
  "total_issues": 12,
  "pass": 9, "fail": 3,
  "fail_reasons": [{"pattern": "kernel crash", "count": 2}],
  "turns_per_issue": {"min": 4, "max": 31},
  "top_syntax_errors": ["unbalanced brace", "missing import"]
}

## 用法
```python
from cgc_trajectory_reader import summarize
summary = summarize("/path/to/trajectories")
# → 返回上述 JSON，供模型 review
```
```

**配套 Python**（`cgc_trajectory_reader.py`）：解析 yaml + 日志，聚合出失败模式。

### 3.3 /refine 的 prompt 模板（每次同步后执行）

```text
/refine 分析 CGC 最近一轮 MoE streaming 优化轨迹：

1. 用 cgc-trajectory-reader 读取 $LOCAL_ROOT 最新任务目录
2. 找出：
   - 反复出现的失败模式（按次数排序）
   - 耗时异常的步骤（turns 特别多的 issue）
   - 与已知优化点的关联（hot pool / slots / r2量化）
3. 沉淀为 harness 状态：
   - 若是**通用教训**（如"slots 不足导致 LRU 颠簸"）→ create/update 一条经验
   - 若是**可复用方法**（如"trace 收集流程"）→ 用 skill-creator 打包成 skill
   - 若是**过时结论** → delete 旧条目（有快照可回滚）
4. 输出：本轮提炼了什么、建议下一轮 CGC 实验改什么
```

---

## 4. 沉淀成什么 skill（按发现频率预分类）

| 发现模式 | 沉淀形式 | 示例 |
|---|---|---|
| 反复出现的**技术错误** | `harness 经验条目`（refine 文本态） | "expert stride 未页对齐 → pread 慢" |
| 可复用的**实验方法** | `skill 包`（可执行） | `bench-ab-validator`（交错 A/B + 防漂移） |
| **数据洞察** | 写入 `harness_state.json` | "r2 vs r3 质量差 RMSE 0.013-0.019" |
| **过时/已推翻结论** | delete（快照回滚） | "mmap 优于 pread"（已被 §7 推翻） |

---

## 5. 闭环节奏（建议）

```
每轮 CGC 实验结束：
  1. rsync 轨迹 → 本地（3.1）
  2. prime-agent 会话内执行 /refine 模板（3.3）
  3. 得到：N 条经验更新 + 1 份"下一轮建议"
  4. 按建议改 CGC 配置/代码 → 下一轮
```

**首轮目标**（贴合 MOE_STREAMING_TASK.md）：让 prime-agent 先跑通 1 个完整闭环
（同步 → refine → 建议），验证链路，再放开让它自主循环。

---

## 6. 边界与注意事项

1. **轨迹只读**：prime-agent 只读同步来的轨迹副本，**不写回远端**（远端是 CGC 的审计源）
2. **安全**：skill 是可执行 Python（skills.md 明示安全警告）——`cgc_trajectory_reader` 只做解析，不执行轨迹里的命令
3. **模型慢**：本地 26B prefill 慢（4617 token 系统提示 ~10-25 分钟）——建议 /refine 用小上下文（`--prompt-template` 精简），或换 r3 模型
4. **路径约定**：同步目录固定 `~/.prime/agent/harness/` 之外的 `.cgc-trajectories/`，避免污染 harness 状态
5. **回滚保障**：/refine 自带 before/after 快照，误删可回滚

---

## 7. 待验证项（接入前）

- [ ] Host1/Host2 上 trajectories/ 实际文件结构（当前只确认了代码里的引用，未亲验远端）
- [ ] `run_batch_exit_statuses.yaml` 的 yaml schema（instances_by_exit_status 的具体键）
- [ ] rsync 走现有 SSH ControlMaster 的连通性
- [ ] prime-agent 读本地轨迹的上下文预算（4617 token 系统提示下 /refine 的可用性）

---

## 8. 后续可选扩展

- **自动触发**：CGC 实验完成 → 脚本自动 rsync + 自动拉起 prime-agent /refine（cron 或 daemon 模式）
- **双向**：prime-agent 的建议自动生成 CGC 配置 diff → 提交下一轮（需人工 review 闸门）
- **多项目**：同一个 reader skill 复用于其他 CGC 任务类型（repo_debug 之外的）
