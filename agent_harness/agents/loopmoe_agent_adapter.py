"""
Loop MoE × Terminal-Bench 适配器（tb 的 installed-agent 扩展点）。

与 prime_agent_adapter.py 的区别：
- prime_agent_adapter: 在容器内跑 prime-agent CLI（外部 agent 框架）
- loopmoe_agent_adapter: 直接调用 Loop MoE / Qwen3.6 模型生成 bash 命令，
  不依赖 prime-agent 框架，模型可以是：
    1. Qwen3.6-35B via CGC edge_server (OpenAI API)
    2. Loop MoE grafted model via PyTorch
    3. llama.cpp GGUF server

工作原理：
- 继承 AbstractInstalledAgent
- 每轮观察终端输出 → 构造 prompt → 模型生成 bash 命令 → 执行
- 支持 system prompt 注入（agent 行为规范）
- 支持循环思考（模型内部 RDT 循环迭代）
- 轨迹记录为 trajectory.jsonl，可用于 SFT 数据生成

用法：
  tb run -d terminal-bench-core==0.1.1 \
    --agent-import-path agent_harness.agents.loopmoe_agent_adapter:LoopMoEAgent \
    -m openai/qwen3.6-35b-a3b \
    -k base_url=http://127.0.0.1:1234/v1 -k api_key=sk-local \
    -k max_turns=12 -k max_tokens=4096 \
    --n-tasks 10 --output-path results/loopmoe_round1 --run-id loopmoe_round1
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand


# Default system prompt for terminal agent
DEFAULT_SYSTEM_PROMPT = """You are an expert terminal agent. Your task is to solve the given task by executing bash commands.

Rules:
1. Output ONLY the bash command to execute, no explanation, no markdown code blocks.
2. Each command should be a single line.
3. Check command exit codes and output before proceeding.
4. If a command fails, analyze the error and try a different approach.
5. Use `echo` to create files, `cat` to view them, `ls` to explore.
6. When the task is complete, output `echo "TASK_COMPLETE"`.
7. Do not use interactive commands (vim, nano, less, top).
8. Use `timeout` for long-running commands.
"""

# Regex to extract bash command from model output (strips markdown, explanations)
COMMAND_PATTERNS = [
    re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"^\s*\$\s*(.+)$", re.MULTILINE),
]


class LoopMoEAgent(AbstractInstalledAgent):
    """
    Terminal-Bench agent that uses Loop MoE / Qwen3.6 directly.

    No external agent framework (prime-agent) required.
    Model is called via OpenAI-compatible API (CGC edge_server / llama.cpp).
    """

    @staticmethod
    def name() -> str:
        return "loopmoe-agent"

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        max_turns: int = 12,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout_ms: int = 600000,
        trajectory_dir: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name or os.environ.get("LOOPMOE_MODEL", "qwen3.6-35b-a3b")
        self._api_key = api_key or os.environ.get("LOOPMOE_API_KEY", "sk-local")
        self._base_url = base_url or os.environ.get("LOOPMOE_BASE_URL", "http://127.0.0.1:1234/v1")
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_turns = int(max_turns)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._timeout_ms = int(timeout_ms)
        self._trajectory_dir = trajectory_dir
        self._turn_count = 0
        self._trajectory: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # AbstractInstalledAgent interface
    # ------------------------------------------------------------------
    @property
    def _env(self) -> dict[str, str]:
        return {
            "LOOPMOE_MODEL": self._model_name,
            "LOOPMOE_BASE_URL": self._base_url,
            "LOOPMOE_API_KEY": self._api_key,
        }

    @property
    def _install_agent_script_path(self) -> Path:
        # LoopMoEAgent doesn't need container installation (calls API from host)
        # Return a minimal no-op script
        script = Path(__file__).parent / "loopmoe-agent-setup.sh"
        if not script.exists():
            script.write_text("#!/usr/bin/env bash\necho 'loopmoe-agent: no installation needed (API-based)'\n")
        return script

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        """
        Main agent loop: observe → think (model) → act (bash command).

        Returns list of TerminalCommand to execute.
        """
        self._turn_count += 1
        turn_start = time.time()

        # Build prompt from instruction + conversation history
        prompt = self._build_prompt(instruction)

        # Call model
        model_output = self._call_model(prompt)

        # Extract bash command
        command = self._extract_command(model_output)

        # Record trajectory
        self._trajectory.append({
            "turn": self._turn_count,
            "instruction": instruction,
            "model_output": model_output,
            "command": command,
            "timestamp": time.time(),
            "latency_ms": int((time.time() - turn_start) * 1000),
        })

        # Check for task complete signal
        if "TASK_COMPLETE" in command.upper():
            return [
                TerminalCommand(
                    command='echo "TASK_COMPLETE"',
                    min_timeout_sec=0.0,
                    max_timeout_sec=10.0,
                    block=True,
                    append_enter=True,
                )
            ]

        return [
            TerminalCommand(
                command=command,
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=False,
                append_enter=True,
            )
        ]

    def perform_task(self, instruction, session, logging_dir=None):
        """Override to save trajectory after task completion."""
        result = super().perform_task(instruction, session, logging_dir)
        self._save_trajectory(logging_dir)
        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _build_prompt(self, instruction: str) -> List[Dict[str, str]]:
        """Build chat messages from instruction and trajectory."""
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add recent trajectory as context (last 4 turns)
        for step in self._trajectory[-4:]:
            messages.append({"role": "user", "content": f"Task: {step['instruction']}"})
            messages.append({"role": "assistant", "content": step["command"]})

        # Current instruction
        messages.append({"role": "user", "content": instruction})

        return messages

    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        """Call model via OpenAI-compatible API."""
        import requests

        payload = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": 0.9,
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LoopMoEAgent] Model call error: {e}")
            return f'echo "Model error: {e}"'

    def _extract_command(self, model_output: str) -> str:
        """Extract bash command from model output."""
        text = model_output.strip()

        # Try markdown code block
        for pattern in COMMAND_PATTERNS:
            match = pattern.search(text)
            if match:
                cmd = match.group(1).strip()
                if cmd:
                    return cmd.split("\n")[0].strip()  # First line only

        # If output is multiple lines, take first non-empty line
        for line in text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("```"):
                # Remove leading $ if present
                if line.startswith("$"):
                    line = line[1:].strip()
                return line

        # Fallback
        return text[:500] if text else 'echo "empty model output"'

    def _save_trajectory(self, logging_dir: Optional[str]):
        """Save trajectory to JSONL file."""
        if not logging_dir and not self._trajectory_dir:
            return
        save_dir = Path(logging_dir or self._trajectory_dir or ".")
        save_dir.mkdir(parents=True, exist_ok=True)
        traj_path = save_dir / "trajectory.jsonl"
        with open(traj_path, "w", encoding="utf-8") as f:
            for step in self._trajectory:
                f.write(json.dumps(step, ensure_ascii=False) + "\n")
        print(f"[LoopMoEAgent] Trajectory saved: {traj_path} ({len(self._trajectory)} turns)")
