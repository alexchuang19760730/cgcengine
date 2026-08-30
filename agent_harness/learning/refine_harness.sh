#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# host 侧学习：把失败轨迹 refine 进 prime-agent 的 Continual Harness。
# 每轮评估后调用，产出更新的 skills/memories 存到 tb_loop/harness/，
# 下一轮 tb run 会把它们注入任务容器。
#
# 三大改进（相对旧版只喂失败摘要）：
#   1. 喂完整轨迹：commands.txt + panes/pre-agent.txt + post-agent.txt +
#      agent-logs/trajectory.jsonl（尾部截断），而不是只有失败原因一行。
#   2. 蒸馏模型用 freebuff2api 的 codebuff（deepseek-v4-flash），质量高于
#      端侧 gemma4 自蒸馏；端点不可达时自动回退本地模型。
#   3. /refine --global 显式全局作用域，且指令限定只提炼 memory/prompt，
#      不创建 skill/subagent、不改写系统提示词。
#
# 用法: refine_harness.sh <failures.json> [max_failures]
#   failures.json 由 extract_failures.py 生成（含 trial_dir 指向完整轨迹）。
# ============================================================
FAILURES="${1:-failures.json}"
MAX="${2:-5}"

TB_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=config.env
source "$TB_LOOP_DIR/config.env"

mkdir -p "$TB_HARNESS_DIR"

if [[ ! -f "$FAILURES" ]]; then
    echo "error: $FAILURES 不存在（先跑 extract_failures.py）" >&2
    exit 2
fi

# 教训级归因快照：记录 refine 前后的 harness_state，供 attribution.py 做条目级归因
SNAPSHOT_DIR="$(cd "$(dirname "$FAILURES")" 2>/dev/null && pwd)"
if [[ -n "$SNAPSHOT_DIR" && -f "$TB_HARNESS_DIR/harness_state.json" ]]; then
    cp "$TB_HARNESS_DIR/harness_state.json" "$SNAPSHOT_DIR/harness_state.before.json"
    # refinements.jsonl 决定条目→refinement 的溯源（回滚单位），必须一起快照
    if [[ -f "$TB_HARNESS_DIR/refinements.jsonl" ]]; then
        cp "$TB_HARNESS_DIR/refinements.jsonl" "$SNAPSHOT_DIR/refinements.before.jsonl"
    fi
fi

# 蒸馏模型：默认 freebuff2api 的 codebuff；不可达时回退本地 gemma4
REFINE_MODEL="${TB_REFINE_MODEL_PATTERN:-freebuff-codebuff/${SFT_MODEL}}"
if [[ "$REFINE_MODEL" == freebuff-codebuff/* ]]; then
    probe="${SFT_API_BASE_URL//host.docker.internal/127.0.0.1}"
    if ! curl -sf -m 8 -H "Authorization: Bearer $SFT_API_KEY" "$probe/models" >/dev/null 2>&1; then
        echo "WARN: freebuff2api ($probe) 不可达 -> 蒸馏回退本地模型 $TB_MODEL_PATTERN"
        REFINE_MODEL="$TB_MODEL_PATTERN"
    fi
fi
echo "== refine 蒸馏模型: $REFINE_MODEL"

export PRIME_AGENT_CODING_AGENT_DIR="$TB_HARNESS_DIR"

# 找一个能用的 python（Windows 上 python3 可能是坏的 WindowsApps 桩）
PY=""
if command -v python3 >/dev/null 2>&1 && python3 -c 'pass' >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1 && python -c 'pass' >/dev/null 2>&1; then
    PY=python
elif [[ -x "$TB_LOOP_DIR/../.tb-loop-venv/Scripts/python" ]]; then
    PY="$TB_LOOP_DIR/../.tb-loop-venv/Scripts/python"
else
    echo "error: 找不到可用的 python（尝试 python3 / python / .tb-loop-venv）" >&2
    exit 2
fi

"$PY" - "$FAILURES" "$MAX" "$REFINE_MODEL" <<'PY'
import json
import os
import re
import subprocess
import sys

failures_path, max_fail, refine_model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
failures = json.load(open(failures_path, encoding="utf-8"))[:max_fail]
if not failures:
    print("no failures to refine")
    sys.exit(0)

max_turns = os.environ.get("TB_MAX_TURNS", "12")
max_tokens = os.environ.get("TB_MAX_TOKENS", "30000")
timeout_ms = os.environ.get("TB_TIMEOUT_MS", "600000")
max_cont = os.environ.get("TB_MAX_CONTINUATIONS", "3")

MAX_CHARS = int(os.environ.get("TB_REFINE_MAX_CHARS", "6000"))
MAX_TRAJ = int(os.environ.get("TB_REFINE_MAX_TRAJ", "6"))
STALL_SEC = int(os.environ.get("TB_REFINE_STALL_SEC", "30"))
MAX_CAST_INPUTS = int(os.environ.get("TB_REFINE_MAX_CAST_INPUTS", "40"))

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][0-9A-B]")


def clean_ansi(s):
    """去掉 ANSI 转义序列（颜色/光标控制），保留可读文本。"""
    s = ANSI_RE.sub("", s)
    s = s.replace("\r", "")
    return s


def asciinema_evidence(trial_dir, max_inputs=MAX_CAST_INPUTS, stall_sec=STALL_SEC):
    """解析 tb 的 asciinema 录制（<trial>/sessions/agent.cast，v2 格式），
    产出：输入时间线（agent 何时敲了什么）+ 停顿检测（在哪一步卡住）。
    .cast 结构：首行 header JSON，随后每行 [time, duration, type, data]。
    type "i" = 键盘输入（还原为命令），type "o" = 终端输出，type "m" = marker。"""
    p = os.path.join(trial_dir, "sessions", "agent.cast")
    if not os.path.isfile(p):
        return None
    inputs = []  # (time, text) —— 注意保留 \r 提交符，切分后再清理
    stall_warn = []
    prev_t = None
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or not line.startswith("["):
                    continue  # 跳过 header
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if not isinstance(ev, list) or len(ev) < 4:
                    continue
                t, dur, typ, data = ev[0], ev[1], ev[2], ev[3]
                if typ == "i" and data:
                    txt = str(data)
                    if txt:  # 注意：\r 提交符单帧也要保留，不能 strip 掉
                        if prev_t is not None and (t - prev_t) >= stall_sec:
                            stall_warn.append(f"t=+{t:.0f}s 上一条输入后停顿 {t - prev_t:.0f}s（疑似卡住/等输出）")
                        inputs.append((t, txt))
                        prev_t = t
    except Exception:
        return None
    if not inputs:
        return None
    # 合并输入为命令级：终端按字符/小块发送输入，命令以 \r（回车提交）结束。
    # 把累积文本按第一个 \r 或 \n 切分，避免把两条命令拼在一起。
    merged = []
    buf, buf_start = "", None
    for t, txt in inputs:
        if buf_start is None:
            buf_start = t
        buf += txt
        while True:
            r, n = buf.find("\r"), buf.find("\n")
            cuts = [i for i in (r, n) if i >= 0]
            if not cuts:
                break
            idx = min(cuts)
            cmd, buf = buf[:idx], buf[idx + 1:]
            cmd = clean_ansi(cmd).strip()
            if cmd:
                merged.append([buf_start, cmd])
            buf_start = t
    tail = clean_ansi(buf).strip()
    if tail:
        merged.append([buf_start, tail])
    lines = []
    for t, txt in merged[:max_inputs]:
        txt = " ".join(txt.split())[:160]
        lines.append(f"t=+{t:.0f}s  输入: {txt}")
    head = f"总输入事件 {len(merged)} 条（显示前 {min(len(merged), max_inputs)} 条）"
    if stall_warn:
        head += f"；检测到 {len(stall_warn)} 处停顿>= {stall_sec}s"
    out = [head, *lines]
    if stall_warn:
        out.append("停顿点:")
        out.extend(stall_warn[:6])
    text = "\n".join(out)
    return text[: MAX_CHARS * 2]


def tail_file(path, max_chars=MAX_CHARS, max_lines=200):
    """读文件尾部（按行截断 + 按字符截断），文件不存在返回 None。"""
    p = os.path.join(path)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except Exception:
        return None
    lines = lines[-max_lines:]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "…[截断]…\n" + text[-max_chars:]
    return text


def trajectory_summary(trial_dir, max_records=MAX_TRAJ):
    """trajectory.jsonl 尾部若干条，压成 紧凑 摘要（step + commands + 结尾片段）。"""
    p = os.path.join(trial_dir, "agent-logs", "trajectory.jsonl")
    if not os.path.isfile(p):
        return None
    recs = []
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    except Exception:
        return None
    out = []
    for r in recs[-max_records:]:
        cmds = r.get("commands") or []
        cmd_txt = "; ".join(str(c)[:120] for c in cmds) if cmds else "(无命令)"
        note = r.get("note") or ""
        obs_tail = (r.get("obs") or "").strip().splitlines()
        obs_tail = obs_tail[-1][:120] if obs_tail else ""
        out.append(f"step {r.get('step')} | cmds: {cmd_txt}{f' | note: {note}' if note else ''}\n"
                   f"   obs尾: {obs_tail}")
    text = "\n".join(out)
    return text[: MAX_CHARS * 2]


for f in failures:
    trial_dir = f.get("trial_dir") or ""
    sections = []
    for label, path in [
        ("命令历史 commands.txt 尾部", os.path.join(trial_dir, "commands.txt")),
        ("任务开始前终端状态 panes/pre-agent.txt 尾部", os.path.join(trial_dir, "panes", "pre-agent.txt")),
        ("agent 结束后的终端状态 panes/post-agent.txt 尾部", os.path.join(trial_dir, "panes", "post-agent.txt")),
    ]:
        txt = tail_file(path)
        if txt:
            sections.append(f"===== {label} =====\n{txt}")
    traj = trajectory_summary(trial_dir)
    if traj:
        sections.append(f"===== agent 轨迹摘要 (agent-logs/trajectory.jsonl) =====\n{traj}")
    cast = asciinema_evidence(trial_dir)
    if cast:
        sections.append(f"===== asciinema 终端录制 (sessions/agent.cast, 输入时间线+停顿) =====\n{cast}")

    evidence = "\n\n".join(sections) if sections else "(无轨迹文件，只有失败摘要)"

    prompt = (
        "/refine --global\n\n"
        "下面是一条 Terminal-Bench 任务的失败轨迹。请提炼 1-2 条可复用的经验，"
        "帮助同类终端任务下次成功。\n\n"
        "硬性要求：\n"
        "- 只提炼 memory 和 prompt 两种类型的条目（create/update 均可）；"
        "不要创建或修改 skill、subagent。\n"
        "- 只做有证据支持的小改动：拒绝一次性噪音、未经验证的假设、瞬时工具输出。\n"
        "- 教训必须能跨会话复用（global scope）：例如通用的命令知识（memory），"
        "或行为策略（prompt，如\"每条命令后检查退出码\"、\"安装失败先试 --break-system-packages\"）。\n"
        "- 不要改写系统提示词，不要修改 base system prompt。\n\n"
        f"任务: {f.get('task_id')}\n"
        f"指令: {f.get('instruction')}\n"
        f"失败原因: {f.get('failure_mode')}\n\n"
        f"{evidence}\n"
    )

    cmd = [
        "prime-agent", "-p", "--offline",
        "--model", refine_model,
        "--autonomous",
        "--autonomous-max-turns", max_turns,
        "--autonomous-max-tokens", max_tokens,
        "--autonomous-timeout-ms", timeout_ms,
        "--autonomous-max-continuations", max_cont,
        prompt,
    ]
    print(f"[refine] {f.get('task_id')} (model={refine_model}, evidence={len(evidence)} chars) ...")
    # stdin=DEVNULL：无 TTY/管道环境（如 Windows）下 prime-agent 会等 stdin 挂起
    r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-500:]
        print(f"  WARN rc={r.returncode}: {tail}")
PY

echo "harness updated at $TB_HARNESS_DIR:"
find "$TB_HARNESS_DIR" -type f -not -name ".gitkeep" | head -20

# 快照 refine 后的状态（归因的 after 侧）
if [[ -n "$SNAPSHOT_DIR" && -f "$TB_HARNESS_DIR/harness_state.json" ]]; then
    cp "$TB_HARNESS_DIR/harness_state.json" "$SNAPSHOT_DIR/harness_state.after.json"
    if [[ -f "$TB_HARNESS_DIR/refinements.jsonl" ]]; then
        cp "$TB_HARNESS_DIR/refinements.jsonl" "$SNAPSHOT_DIR/refinements.after.jsonl"
    fi
    echo "harness 快照: $SNAPSHOT_DIR/harness_state.before.json / .after.json"
fi
