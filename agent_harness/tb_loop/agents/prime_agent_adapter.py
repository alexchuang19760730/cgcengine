"""
Prime Agent × Terminal-Bench 适配器（tb 的 installed-agent 扩展点）。

工作原理：
- 通过 `tb run --agent-import-path tb_loop.agents.prime_agent_adapter:PrimeAgentAgent`
  注册进 tb 的 AgentFactory；
- 本类继承 AbstractInstalledAgent：安装脚本把 prime-agent 装进任务容器，然后在
  容器 tmux 里以 headless 模式跑 prime-agent（-p --autonomous），模型指向 M4
  宿主机上 gemma4 的 OpenAI 兼容端点；
- 学习循环：perform_task 先把 host 侧 harness 状态（tb_loop/harness/，含
  extensions/gemma4-provider.ts 以及前几轮 /refine 产生的 skill/记忆）打包打进
  容器，任务跑完再回传，实现跨轮持续学习（agent 级 TTT）。

用法（由 run_round.sh 调用，也可手动）：
  tb run -d terminal-bench-core==0.1.1 \
    --agent-import-path tb_loop.agents.prime_agent_adapter:PrimeAgentAgent \
    -m openai/<model> \
    -k model_name=<model> -k api_key=<key> -k base_url=<url> \
    -k harness_dir=<tb_loop>/harness -k max_turns=12 -k max_tokens=30000 \
    --n-tasks 10 --output-path results/round_1 --run-id round_1
"""

from __future__ import annotations

import os
import shlex
import tarfile
import tempfile
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand

# 容器内 prime-agent 的工作目录（harness 状态 + provider extension 都在这里）
CONTAINER_HARNESS_DIR = "/prime-agent-harness"
CONTAINER_ENV_SCRIPT = "/prime-agent-harness-env.sh"
CONTAINER_HARNESS_TAR = "/installed-agent/harness.tar.gz"


class PrimeAgentAgent(AbstractInstalledAgent):
    """在 Terminal-Bench 任务容器里跑 prime-agent，模型走宿主 M4 的 gemma4。"""

    @staticmethod
    def name() -> str:
        return "prime-agent"

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_prefix: str = "local-gemma4",
        harness_dir: str | None = None,
        max_turns: int = 12,
        max_tokens: int = 30000,
        timeout_ms: int = 600000,
        max_continuations: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name or os.environ.get("TB_GEMMA4_MODEL", "gemma-4-26b-a4b-it")
        self._api_key = api_key or os.environ.get("TB_GEMMA4_API_KEY", "sk-local")
        self._base_url = base_url or os.environ.get(
            "TB_GEMMA4_BASE_URL", "http://host.docker.internal:1234/v1"
        )
        self._model_prefix = model_prefix or os.environ.get("TB_MODEL_PREFIX", "local-gemma4")
        self._harness_dir = Path(harness_dir or os.environ.get("TB_HARNESS_DIR", "harness"))
        self._max_turns = int(max_turns)
        self._max_tokens = int(max_tokens)
        self._timeout_ms = int(timeout_ms)
        self._max_continuations = int(max_continuations)

    # ------------------------------------------------------------------
    # AbstractInstalledAgent 接口
    # ------------------------------------------------------------------
    @property
    def _env(self) -> dict[str, str]:
        return {
            # 把 harness 目录指到容器内注入的位置（provider extension 自动被发现）
            "PRIME_AGENT_CODING_AGENT_DIR": CONTAINER_HARNESS_DIR,
            # 模型端点（provider extension 里读取这些变量）
            "TB_GEMMA4_BASE_URL": self._base_url,
            "TB_GEMMA4_API_KEY": self._api_key,
            "TB_GEMMA4_MODEL": self._model_name,
            "TB_MODEL_PREFIX": self._model_prefix,
            # 兜底：部分 OpenAI 兼容 server 也认标准 env
            "OPENAI_API_KEY": self._api_key,
            "OPENAI_BASE_URL": self._base_url,
        }

    @property
    def _install_agent_script_path(self) -> Path:
        return Path(__file__).parent / "prime-agent-setup.sh"

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        model = f"{self._model_prefix}/{self._model_name}"
        cmd = (
            f"source {CONTAINER_ENV_SCRIPT} 2>/dev/null || true; "
            f"export PATH=\"$HOME/.local/bin:$PATH\"; "
            f"prime-agent -p --offline --model {shlex.quote(model)} "
            f"--autonomous "
            f"--autonomous-max-turns {self._max_turns} "
            f"--autonomous-max-tokens {self._max_tokens} "
            f"--autonomous-timeout-ms {self._timeout_ms} "
            f"--autonomous-max-continuations {self._max_continuations} "
            f"{shlex.quote(instruction)}"
        )
        return [
            TerminalCommand(
                command=cmd,
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=False,
                append_enter=True,
            )
        ]

    # ------------------------------------------------------------------
    # 学习循环：harness 状态注入
    # ------------------------------------------------------------------
    def perform_task(self, instruction, session, logging_dir=None):
        self._ship_harness(session)
        return super().perform_task(instruction, session, logging_dir)

    def _ship_harness(self, session) -> None:
        """把 host 侧 harness（skills/memories/provider extension）打包进容器。"""
        src = self._harness_dir
        if not src.is_dir():
            return

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as fh:
            tar_path = Path(fh.name)
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for p in sorted(src.rglob("*")):
                    if p.is_file() and p.name != ".gitkeep":
                        tar.add(p, arcname=p.relative_to(src))

            session.copy_to_container(
                tar_path,
                container_dir="/installed-agent",
                container_filename="harness.tar.gz",
            )
            session.container.exec_run(
                [
                    "sh",
                    "-c",
                    (
                        "mkdir -p /prime-agent-harness && "
                        "tar -xzf /installed-agent/harness.tar.gz -C /prime-agent-harness 2>/dev/null || true; "
                        "echo 'export PRIME_AGENT_CODING_AGENT_DIR=/prime-agent-harness' > "
                        "/prime-agent-harness-env.sh"
                    ),
                ]
            )
        finally:
            tar_path.unlink(missing_ok=True)
