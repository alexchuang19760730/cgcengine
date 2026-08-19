# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from pathlib import Path
from typing import Iterator


def _env_to_bool(env_name: str, default: bool) -> bool:
    env_value = str(os.environ.get(env_name, default))
    if env_value.lower() in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if env_value.lower() in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return default


@contextlib.contextmanager
def set_env_var(key: str, value: str) -> Iterator[None]:
    """Temporarily set an environment variable."""
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


# Enable FX graph visualization, which actually controls `enable_fx_graph_viz` in the Inductor config.
MAGI_ENABLE_FX_GRAPH_VIZ: bool = _env_to_bool("MAGI_ENABLE_FX_GRAPH_VIZ", default=False)

# FX graph visualization node description mode: simple or detailed.
MAGI_FX_GRAPH_VIZ_NODE_DESC: str = os.getenv("MAGI_FX_GRAPH_VIZ_NODE_DESC", "simple")

# Equal to TORCHINDUCTOR_PATTERN_MATCH_DEBUG environment.
MAGI_PATTERN_MATCH_DEBUG: str | None = os.getenv("MAGI_PATTERN_MATCH_DEBUG")

# Default path for shared memory binaries.
MAGI_SHARED_BIN_PATH = "/dev/shm"

# Logging level for MagiCompiler (DEBUG / INFO / WARNING / ERROR). Read once at import time.
MAGI_LOGGING_LEVEL: str = os.getenv("MAGI_LOGGING_LEVEL", "WARNING").upper()


def _cgc_repo_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def cgc_root_dir() -> str:
    return os.environ.get("CGC_DSTMC") or _cgc_repo_root()


def cgc_output_dir() -> str:
    out_dir = os.environ.get("CGC_OUTPUT_DIR") or os.path.join(cgc_root_dir(), "Output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def cgc_temp_dir() -> str:
    temp_dir = os.environ.get("CGC_TEMP_DIR") or os.path.join(cgc_root_dir(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def cgc_system_output_dir(system_name: str) -> str:
    normalized = str(system_name).strip().lower()

    if normalized in {"psi0_system", "pi0_system"}:
        system_dir = os.path.join(cgc_output_dir(), "Models", "psi0_system")
    elif normalized in {"ds4", "ds4_system"}:
        system_dir = os.path.join(cgc_output_dir(), "Models", "DS4")
    elif normalized == "mac_system":
        system_dir = os.path.join(cgc_output_dir(), "Hardware", "Mac_system")
    elif normalized == "nvidia_system":
        system_dir = os.path.join(cgc_output_dir(), "Hardware", "Nvidia_system")
    elif normalized == "ascend_system":
        system_dir = os.path.join(cgc_output_dir(), "Hardware", "Ascend_system")
    else:
        system_dir = os.path.join(cgc_output_dir(), system_name)
    os.makedirs(system_dir, exist_ok=True)
    return system_dir


def cgc_report_path(filename: str, system_name: str | None = None) -> str:
    name = os.path.basename(str(filename))
    if system_name:
        base_dir = cgc_system_output_dir(system_name)
    else:
        lowered = name.lower()
        fallback_system = os.environ.get("CGC_REPORT_FALLBACK_SYSTEM") or "nvidia_system"
        if "ds4" in lowered:
            base_dir = cgc_system_output_dir("ds4")
        elif "psi0" in lowered or "vla" in lowered or "holomotion" in lowered:
            base_dir = cgc_system_output_dir("psi0_system")
        elif "ascend" in lowered:
            base_dir = cgc_system_output_dir("ascend_system")
        elif "mlx" in lowered or "metal" in lowered or "mps" in lowered:
            base_dir = cgc_system_output_dir("mac_system")
        elif (
            "cuda" in lowered
            or "vllm" in lowered
            or "rtx" in lowered
            or "a100" in lowered
            or "h100" in lowered
            or "deepseek" in lowered
            or "qwen" in lowered
            or "ablation" in lowered
            or lowered.startswith("pd_")
            or "speculative" in lowered
            or "mtp" in lowered
        ):
            base_dir = cgc_system_output_dir("nvidia_system")
        else:
            base_dir = cgc_system_output_dir(fallback_system)

    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    return os.path.join(reports_dir, name)


def cgc_detect_task_domain_and_model_family(*, model: str = "", gguf_path: str | None = None) -> dict[str, str]:
    signals = " ".join([str(model or ""), str(gguf_path or "")]).strip().lower()
    if any(k in signals for k in ["agent", "harness", "moe"]):
        return {"task_domain": "agent", "model_family": "agent", "model_tag": "moe_harness"}
    if any(k in signals for k in ["psi0", "pi0", "vla", "openvla", "embodied", "holomotion"]):
        return {"task_domain": "embodied", "model_family": "psi0_vla", "model_tag": "vla_psi0"}
    if any(k in signals for k in ["gemma4", "gemma 4", "gemma"]):
        return {"task_domain": "models", "model_family": "gemma4", "model_tag": "gemma4"}
    if any(k in signals for k in ["deepseek", "ds4"]):
        if any(k in signals for k in ["flash_pro", "flash-pro", "flashpro", "flash pro", "pro"]):
            return {"task_domain": "models", "model_family": "ds4_flash_pro", "model_tag": "deepseek_v4_flash_pro"}
        return {"task_domain": "models", "model_family": "ds4", "model_tag": "deepseek_v4"}
    if any(k in signals for k in ["qwen"]):
        return {"task_domain": "models", "model_family": "qwen", "model_tag": "qwen"}
    return {"task_domain": "models", "model_family": "unknown", "model_tag": "unknown"}


def cgc_detect_hardware_profile(*, device: str = "") -> str:
    d = str(device or "").strip().lower()
    if d in {"cuda", "mps", "cpu", "ascend"}:
        return d
    return "unknown"
