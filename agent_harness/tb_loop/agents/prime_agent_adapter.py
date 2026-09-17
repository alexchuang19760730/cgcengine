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
# host 上快取的 prime-agent release 鏡像（由 scripts/fetch-prime-agent-mirror.sh 產生）。
# 為什麼要送它進容器：同一個 URL，host 是 ~12.5 MB/s、容器內只有 ~164 KB/s（差 80 倍，
# 瓶頸是 colima NAT），而官方安裝器把 `--max-time 300` 寫死 ⇒ 59.6 MB 的 release 永遠
# 抓不完。實測後果：整段安裝 421.89s 逾時、`INSTALL_FAIL_STATUS`、agent 從未被啟動、
# `total_input_tokens = 0`，而 tb 把這個結果報成 `agent_timeout`（與模型無關）。
CONTAINER_MIRROR_TAR = "/installed-agent/prime-agent-mirror.tar.gz"
DEFAULT_MIRROR_TAR = Path.home() / ".cache" / "prime-agent-mirror.tar.gz"


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
        # ★ tb 會把 `-m` 的**值**當成 model_name 傳進來，而那個值傳統寫法帶 provider 前綴
        #   （`openai/<model>`）。那個前綴不該變成送給端點的 model id —— 2026-09-17 單變數實測：
        #   `local-gemma4/openai/deepseek/deepseek-flash` ⇒ 400「model or service ID … does
        #   not exist」；拿掉 `openai/` 之後同一顆容器 rc=0、正常回覆。所以在這裡剝掉它，
        #   讓本 adapter 對 tb 的兩種寫法（有／無前綴）都成立。
        if self._model_name.startswith("openai/"):
            self._model_name = self._model_name[len("openai/"):]
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

    @property
    def _mirror_tar(self) -> Path:
        return Path(os.environ.get("TB_PA_MIRROR_TAR", str(DEFAULT_MIRROR_TAR)))

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
        self._ship_mirror(session)
        self._ship_harness(session)
        return super().perform_task(instruction, session, logging_dir)

    def _ship_mirror(self, session) -> None:
        """把 host 上快取的 prime-agent release 鏡像送進容器。

        送檔走 `copy_to_container`（Docker API），**不經過容器那個 164 KB/s 的網路**；
        送進去之後由 prime-agent-setup.sh 在容器內起一個 loopback http server 供檔，
        讓官方安裝器從 `127.0.0.1` 取 —— 那是上游明文支援的 feed（見該腳本的說明）。

        沒有鏡像時只印一行就繼續：安裝會退回「直接從外網下載」的老路徑，
        容器網路慢時那條路徑會逾時失敗（可接受，比讓 agent 靜默 0 token 好讀）。
        """
        tar_path = self._mirror_tar
        if not tar_path.is_file():
            print(
                f"[prime-agent] 沒有離線鏡像 {tar_path} —— 安裝會退回直接下載。\n"
                "              產生它: bash agent_harness/tb_loop/scripts/"
                "fetch-prime-agent-mirror.sh"
            )
            return
        session.copy_to_container(
            tar_path,
            container_dir="/installed-agent",
            container_filename="prime-agent-mirror.tar.gz",
        )
        print(
            f"[prime-agent] 已注入離線鏡像 {tar_path.name}"
            f"（{tar_path.stat().st_size} B）"
        )

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
