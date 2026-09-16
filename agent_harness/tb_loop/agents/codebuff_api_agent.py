"""
Codebuff API 迭代式 terminal agent（tb 自定义 BaseAgent，host 侧运行）。

用 freebuff2api（codebuff 上游的 OpenAI 兼容代理）驱动模型在 Terminal-Bench
任务容器里迭代执行命令，并把完整轨迹写入 tb 的 agent-logs 目录，供
learning/build_sft_dataset.py 整理成 SFT/LoRA 数据集。

与 prime_agent_adapter 的区别：模型是云端 codebuff（经 freebuff2api），
不需要在容器里装任何东西——本 agent 在 host 侧跑循环，通过 tmux 读写容器终端。

轨迹格式（logging_dir/trajectory.jsonl，每行一步）：
    {"step": 0, "obs": "<终端片段>", "reply": "<模型原始回复>",
     "commands": ["..."], "explanation": "...", "input_tokens": n, "output_tokens": m}

用法：
    tb run --agent-import-path tb_loop.agents.codebuff_api_agent:CodebuffApiAgent \
      -m openai/deepseek/deepseek-v4-flash \
      -k base_url=http://127.0.0.1:8000/v1 -k api_key=sk-local -k max_steps=20 \
      --n-tasks 10 --output-path results/sft_gen --run-id gen_1
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

SYSTEM_PROMPT = (
    "You are an expert terminal agent inside a Linux shell. "
    "Complete the given task by running shell commands.\n"
    "Rules:\n"
    "- Use your terminal tool to run commands; read the terminal output between steps and adapt.\n"
    "- Prefer small, verifiable steps over one giant script.\n"
    "- Never modify files outside the task workspace.\n"
    "- When the task is complete, reply with a short summary and stop."
)

# codebuff agent 原生工具调用格式（DSML，标记是全角竖线）：
#   <||DSML||>tool_calls\n<||DSML||>invoke name="terminal_command">\n
#   <||DSML||>parameter name="command" string="true">ls -la<||DSML||>/parameter>\n
#   <||DSML||>/invoke>\n<||DSML||>/tool_calls>
# 解析用结构匹配，不依赖具体标记字符；只要某 invoke 的参数里有 cmd/command 就算命令。
TOOL_INVOKE_RE = re.compile(r'invoke\s+name="([^"]+)"')
# invoke/parameter 的闭合是 </\uff5c\uff5cDSML\uff5c\uff5cinvoke>（`<` 后是 `/`）；
# execute 的闭合才是 <\uff5c\uff5cDSML\uff5c\uff5c/execute>（`<` 后是竖线）。两种都要兼容。
TOOL_CLOSE_RE = re.compile(r"</[^>]*?invoke>|<[^>]*?/invoke>")
TOOL_PARAM_RE = re.compile(
    r'<[^>]*?parameter\s+name="([^"]+)"[^>]*>(.*?)(?:</[^>]*?parameter>|<[^>]*?/parameter>)', re.S
)
TOOL_PARAMS = {"cmd", "command"}

OBS_LIMIT = 4000  # 每次喂给模型的终端片段最大字符数
OBS_WINDOW = 5    # 发给模型的对话窗口：保留最近 N 步完整消息，更早的压缩成摘要
# 增量模式（默认关）：只发上一步之后新增的终端输出，省 60%+ input token。
# 行为风险：模型看不到早期输出（靠窗口摘要补命令线索），需真实 API 验证后再开。
OBS_INCREMENTAL = os.environ.get("OBS_INCREMENTAL", "0") == "1"


# 另一种常见格式：<||DSML||>execute>\nls -la\n<||DSML||>/execute> （命令文本直接包在块里）
TOOL_EXEC_RE = re.compile(r"<[^>]*?execute>(.*?)<[^>]*?/execute>", re.S)


def _extract_tool_commands(reply: str) -> list[str]:
    """解析 codebuff 原生 DSML 工具调用，返回要执行的命令列表。
    支持两种格式：
      1) tool_calls + invoke name="..." + parameter name="cmd|command"（旧）
      2) <DSML>execute>\n<命令>\n</DSML>/execute>（新，命令文本直接包块）
    """
    commands: list[str] = []
    for m in TOOL_INVOKE_RE.finditer(reply):
        close = TOOL_CLOSE_RE.search(reply, m.end())
        block = reply[m.end() : close.start() if close else len(reply)]
        for pm in TOOL_PARAM_RE.finditer(block):
            if pm.group(1) in TOOL_PARAMS:
                val = pm.group(2).strip()
                if val:
                    commands.append(val)
    if commands:
        return commands
    # 兜底：execute 块（命令文本直接包在 <DSML>execute> ... </DSML>/execute> 里）
    for m in TOOL_EXEC_RE.finditer(reply):
        val = m.group(1).strip()
        if val:
            commands.append(val)
    return commands



def _extract_plaintext_command(content: str) -> list[str]:
    """从纯文本 content 提取命令（Qwen3.6 等非 DSML 模型的兜底）。"""
    if not content or not content.strip():
        return []
    text = content.strip()
    fence = re.search(r"```(?:bash|sh)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    if len(lines) == 1:
        line = lines[0]
        if re.match(r'^(I |The |Here |Sure|Okay|Let me|Now )', line, re.I):
            return []
        return [line]
    cmds = []
    for line in lines:
        if not re.match(r'^(I |The |Here |Sure|Okay|Let me|Now )', line, re.I):
            cmds.append(line)
    return cmds


def _extract_json(reply: str) -> dict[str, Any] | None:
    """从模型回复里稳健地抠出 JSON 对象（兜底格式）。"""
    text = reply.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 兜底：找第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


class CodebuffApiAgent(BaseAgent):
    """host 侧迭代 agent：调 freebuff2api，写轨迹日志。"""

    @staticmethod
    def name() -> str:
        return "codebuff-api"

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_steps: int = 20,
        max_tokens: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name or os.environ.get("SFT_MODEL", "deepseek/deepseek-v4-flash")
        self._api_key = api_key or os.environ.get("SFT_API_KEY", "sk-local")
        self._base_url = (base_url or os.environ.get("SFT_API_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self._max_steps = int(max_steps)
        self._max_tokens = int(max_tokens)

    # ------------------------------------------------------------------
    # 模型调用（stdlib only，零额外依赖）
    # ------------------------------------------------------------------
    def _chat(self, messages: list[dict[str, str]]) -> tuple[str, int, int, dict]:
        """返回 (reply_text, input_tokens, output_tokens, raw_response)。"""
        body = json.dumps(
            {
                "model": self._model_name,
                "messages": messages,
                "stream": False,
                "max_tokens": self._max_tokens,
                "temperature": 0.0,
            }
        ).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"URL error: {e}") from e

        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return (
            content,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            data,
        )

    # ------------------------------------------------------------------
    # tb BaseAgent 接口
    # ------------------------------------------------------------------
    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        total_in = 0
        total_out = 0
        trajectory: list[dict[str, Any]] = []

        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
            (logging_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT)
            (logging_dir / "task.txt").write_text(instruction)

        # 会话历史：user 终端片段 ↔ assistant 回复
        # 只保留最近 OBS_WINDOW 步的完整消息；更早的压缩成命令摘要，
        # 控制长任务 input token 不随步数线性增长（8 步任务曾达 45K）。
        history: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"TASK: {instruction}"},
        ]
        history_n_steps = 0  # 已加入的 terminal step 数（用于窗口压缩）

        # 上一步命令回显：显式附在终端观察后面（obs 被 OBS_LIMIT 截断时
        # "$ cmd" 回显行会丢，模型看不到自己跑了什么 → 重复执行确认）。
        # 与 learning/build_sft_dataset.py 的 to_messages 保持同一格式。
        prev_commands: list[str] = []
        last_pane_len = 0
        for step in range(self._max_steps):
            pane = session.capture_pane()
            if step == 0 or not OBS_INCREMENTAL:
                obs = pane[-OBS_LIMIT:]
            else:
                # 增量模式：只发上一步之后新增的终端输出，避免重复发送历史 transcript。
                delta = pane[last_pane_len:]
                obs = delta[-OBS_LIMIT:] if delta.strip() else pane[-OBS_LIMIT:]
            last_pane_len = len(pane)
            history, history_n_steps = self._append_terminal_message(
                history, history_n_steps, step, obs, prev_commands
            )

            reply, inp, out, _raw = self._chat(history)
            total_in += inp
            total_out += out
            reasoning = ((_raw.get("choices") or [{}])[0].get("message") or {}).get(
                "reasoning_content"
            )

            commands, explanation = self._parse_reply(reply)
            if not commands and not explanation:
                # 无命令（可能是纯文本计划/结束语）：一次修复机会
                history.append({"role": "assistant", "content": reply})
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "No terminal command was produced. "
                            "Use your terminal tool to run a shell command, or reply '"
                            "DONE' if the task is finished."
                        ),
                    }
                )
                reply2, inp2, out2, _ = self._chat(history)
                total_in += inp2
                total_out += out2
                commands2, explanation2 = self._parse_reply(reply2)
                if commands2 or explanation2:
                    reply = reply2
                    commands, explanation = commands2, explanation2
                else:
                    # 仍然无命令：把文本当结束步记录，交给 tb 判分，不视为致命错误
                    trajectory.append(
                        {
                            "step": step,
                            "obs": obs,
                            "prev_commands": list(prev_commands),
                            "reply": reply + "\n" + reply2,
                            "commands": [],
                            "explanation": explanation2 or reply2.strip()[:200],
                            "input_tokens": inp + inp2,
                            "output_tokens": out + out2,
                            "note": "no_command",
                        }
                    )
                    self._write_trajectory(logging_dir, trajectory)
                    break

            step_rec = {
                "step": step,
                "obs": obs,
                "prev_commands": list(prev_commands),
                "reply": reply,
                "commands": commands,
                "explanation": explanation,
                "reasoning": reasoning,
                "input_tokens": inp,
                "output_tokens": out,
            }
            trajectory.append(step_rec)
            self._write_trajectory(logging_dir, trajectory)

            for cmd in commands:
                if cmd.strip():
                    session.send_keys([cmd, "Enter"], block=True)

            if not commands:
                # 没有命令 = 模型认为完成（或结束语），交给 tb 判分
                break

            prev_commands = commands

        return AgentResult(
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            failure_mode=FailureMode.NONE,
        )

    @staticmethod
    def _terminal_message(step: int, obs: str, prev_commands: list[str]) -> str:
        """终端观察 + 上一步命令回显（与 build_sft_dataset.to_messages 同格式）。"""
        msg = f"--- terminal (step {step}) ---\n{obs}"
        if prev_commands:
            msg += "\n\n[Commands you just ran: " + "; ".join(prev_commands) + "]"
        return msg

    @staticmethod
    def _append_terminal_message(
        history: list[dict[str, str]],
        n_steps: int,
        step: int,
        obs: str,
        prev_commands: list[str],
    ) -> tuple[list[dict[str, str]], int]:
        """把新一步的终端消息加进 history，并做滑窗压缩。

        history 结构（正常路径无 assistant 回放）：
            [system, TASK, user_step0, user_step1, ...]
        超过 OBS_WINDOW 步后，把窗口外最老的 user 观察折叠成一行命令摘要，
        history 长度保持恒定（≈ OBS_WINDOW 条），input token 不再随步数线性增长。
        模型始终能看到最新一步的全量 obs（终端现状），早期步骤用命令历史补足。
        """
        history.append(
            {
                "role": "user",
                "content": CodebuffApiAgent._terminal_message(step, obs, prev_commands),
            }
        )
        n_steps += 1
        if n_steps <= OBS_WINDOW:
            return history, n_steps
        # 窗口超限：折叠窗口外最老的 user 观察。history 结构：
        # [system, TASK, (summary? | u_i), u_{i+1}, ...] —— 若已有摘要它在 index 2
        has_summary = history[2]["content"].startswith("[Earlier steps:")
        fold_idx = 3 if has_summary else 2
        oldest_user = history[fold_idx]
        # 只从被折叠消息自带的命令回显提取（上一步命令，语义正确、无时序残留）
        folded_cmds = []
        m = re.search(r"\[Commands you just ran: ([^\]]+)\]", oldest_user.get("content", ""))
        if m:
            folded_cmds.append(m.group(1))
        if not folded_cmds:
            folded_cmds.append("(无命令)")
        new_summary = "[Earlier steps: " + "; ".join(folded_cmds)[:400] + "]"
        if has_summary:
            # 合并进已有摘要：旧摘要 + 追加本次折叠命令
            combined = history[2]["content"][:400] + " | " + "; ".join(folded_cmds)[:200]
            history = history[:2] + [{"role": "user", "content": combined[:600]}] + history[3:fold_idx] + history[fold_idx + 1:]
        else:
            history = history[:2] + [{"role": "user", "content": new_summary}] + history[3:]
        return history, n_steps

    @staticmethod
    def _parse_reply(reply: str, content: str = "") -> tuple[list[str], str]:
        """先解析 DSML，再兜底 JSON，最后纯文本命令。"""
        commands = _extract_tool_commands(reply)
        if commands:
            return commands, ""
        parsed = _extract_json(reply)
        if parsed:
            return (
                [str(c) for c in (parsed.get("commands") or [])],
                str(parsed.get("explanation") or ""),
            )
        # 兜底：Qwen3.6 markdown code block / 纯文本命令
        text = content if content else reply
        cmds = _extract_plaintext_command(text)
        if cmds:
            return cmds, ""
        return [], ""

    @staticmethod
    def _write_trajectory(logging_dir: Path | None, trajectory: list[dict[str, Any]]) -> None:
        if logging_dir is None:
            return
        with (logging_dir / "trajectory.jsonl").open("w", encoding="utf-8") as fh:
            for rec in trajectory:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
