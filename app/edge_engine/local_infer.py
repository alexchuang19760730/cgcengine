from __future__ import annotations

import asyncio
import importlib.machinery
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import types
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from app.edge_engine.kda_state_runtime import inspect_kda_state_bytes, resume_one_token_from_kda_state

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "ComputeGraphCompiler-main"
CGC_STATE_DIR = Path(os.environ.get("CGC_HOME") or (Path.home() / ".cgc")).expanduser().resolve()
CGC_CONFIG_PATH = (CGC_STATE_DIR / "config.json").resolve()
for candidate in (REPO_ROOT, ENGINE_ROOT, ENGINE_ROOT / "Backend" / "oMLX"):
    raw = str(candidate.resolve())
    if raw not in sys.path:
        sys.path.insert(0, raw)


# #region debug-point A:m8-gguf-fallback-helper
def _debug_report_m8_gguf_fallback(*, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    _env_path = REPO_ROOT / ".dbg" / "m8-gguf-fallback.env"
    _url = "http://127.0.0.1:7777/event"
    _session = "m8-gguf-fallback"
    try:
        _raw_env = _env_path.read_text(encoding="utf-8")
        for _line in _raw_env.splitlines():
            if _line.startswith("DEBUG_SERVER_URL="):
                _url = _line.split("=", 1)[1].strip() or _url
            if _line.startswith("DEBUG_SESSION_ID="):
                _session = _line.split("=", 1)[1].strip() or _session
        urllib.request.urlopen(
            urllib.request.Request(
                _url,
                data=json.dumps(
                    {
                        "sessionId": _session,
                        "runId": "pre-fix",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": f"[DEBUG] {message}",
                        "data": data,
                        "ts": int(time.time() * 1000),
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.35,
        ).read()
    except Exception:
        pass


# #endregion


# #region debug-point E:dense-streaming-helper
def _debug_report_dense_streaming_measure(*, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    _env_path = REPO_ROOT / ".dbg" / "dense-streaming-measure.env"
    _url = "http://127.0.0.1:7777/event"
    _session = "dense-streaming-measure"
    try:
        _raw_env = _env_path.read_text(encoding="utf-8")
        for _line in _raw_env.splitlines():
            if _line.startswith("DEBUG_SERVER_URL="):
                _url = _line.split("=", 1)[1].strip() or _url
            if _line.startswith("DEBUG_SESSION_ID="):
                _session = _line.split("=", 1)[1].strip() or _session
        urllib.request.urlopen(
            urllib.request.Request(
                _url,
                data=json.dumps(
                    {
                        "sessionId": _session,
                        "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": f"[DEBUG] {message}",
                        "data": data,
                        "ts": int(time.time() * 1000),
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.35,
        ).read()
    except Exception:
        pass


# #endregion


def _read_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _resolve_writable_dir(*candidates: Path) -> Path:
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            target = candidate.expanduser().resolve()
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return target
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("no_writable_local_infer_dir")


def _resolve_local_infer_evidence_root() -> Path:
    explicit = str(os.environ.get("CGC_LOCAL_INFER_EVIDENCE_ROOT") or "").strip()
    default_root = REPO_ROOT / "ComputeGraphCompiler-main" / "Output" / "edge_runtime" / "local_infer"
    fallback_root = Path("/private/tmp/cgc_local_infer")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([default_root, fallback_root])
    return _resolve_writable_dir(*candidates)


def _resolve_cached_hf_snapshot(model_ref: str) -> str:
    model_str = str(model_ref or "").strip()
    if not model_str or Path(model_str).expanduser().exists():
        return model_str
    if "/" not in model_str:
        return model_str

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return model_str

    try:
        resolved = snapshot_download(model_str, local_files_only=True)
    except Exception:
        return model_str

    resolved_path = Path(str(resolved)).expanduser()
    return str(resolved_path.resolve()) if resolved_path.exists() else model_str


def _should_force_qwen2_tokenizer(model_ref: str) -> bool:
    model_path = Path(str(model_ref or "")).expanduser()
    if model_path.exists() and model_path.is_dir():
        tokenizer_config = _read_json_dict(model_path / "tokenizer_config.json")
        if str(tokenizer_config.get("tokenizer_class") or "") == "Qwen2Tokenizer":
            return True

        config = _read_json_dict(model_path / "config.json")
        if str(config.get("model_type") or "") == "qwen2":
            return True

    lowered = str(model_ref or "").lower()
    return "qwen2" in lowered or "qwen2.5" in lowered


def _install_mlx_lm_qwen2_tokenizer_shim(model_ref: str) -> Callable[[], None]:
    if not _should_force_qwen2_tokenizer(model_ref):
        return lambda: None

    try:
        from app.edge_engine import mlx_tokenizer_shim
    except Exception:
        return lambda: None

    original_modules: Dict[str, Any] = {}
    for module_name in ("mlx_lm.tokenizer_utils",):
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]
    sys.modules["mlx_lm.tokenizer_utils"] = mlx_tokenizer_shim

    original_attrs: List[tuple[Any, str, Any]] = []
    live_attr_overrides = {
        "mlx_lm.utils": {
            "TokenizerWrapper": mlx_tokenizer_shim.TokenizerWrapper,
            "_load_tokenizer": mlx_tokenizer_shim.load,
        },
        "mlx_lm.generate": {
            "TokenizerWrapper": mlx_tokenizer_shim.TokenizerWrapper,
        },
    }
    for module_name, overrides in live_attr_overrides.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attr_name, attr_value in overrides.items():
            if hasattr(module, attr_name):
                original_attrs.append((module, attr_name, getattr(module, attr_name)))
                setattr(module, attr_name, attr_value)

    def _restore() -> None:
        for module, attr_name, attr_value in reversed(original_attrs):
            setattr(module, attr_name, attr_value)
        if "mlx_lm.tokenizer_utils" in original_modules:
            sys.modules["mlx_lm.tokenizer_utils"] = original_modules["mlx_lm.tokenizer_utils"]
        else:
            sys.modules.pop("mlx_lm.tokenizer_utils", None)

    return _restore


def _install_text_only_torch_stub(model_ref: str) -> Callable[[], None]:
    if not _should_force_qwen2_tokenizer(model_ref):
        return lambda: None

    original_modules: Dict[str, Any] = {}
    stubbed_names = [
        "torch",
        "torch.cuda",
        "torch.distributed",
        "torch.nn",
        "torch.utils",
        "torch.utils._pytree",
    ]
    for module_name in stubbed_names:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]

    def _make_module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)
        return module

    torch_stub = _make_module("torch")
    torch_stub.__dict__.update(
        {
            "__version__": "0.0.0",
            "Tensor": object,
            "device": object,
            "dtype": object,
            "float16": "float16",
            "float32": "float32",
        }
    )
    torch_stub.version = types.SimpleNamespace(cuda=None)
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.distributed = _make_module("torch.distributed")
    torch_stub.nn = _make_module("torch.nn")
    torch_stub.utils = _make_module("torch.utils")
    torch_stub.utils._pytree = _make_module("torch.utils._pytree")
    torch_stub.utils._pytree.tree_flatten = lambda x: (x, None)
    torch_stub.utils._pytree.tree_unflatten = lambda values, _spec=None: values
    torch_stub.utils._pytree.register_pytree_node = lambda *args, **kwargs: None

    sys.modules["torch"] = torch_stub
    sys.modules["torch.cuda"] = torch_stub.cuda
    sys.modules["torch.distributed"] = torch_stub.distributed
    sys.modules["torch.nn"] = torch_stub.nn
    sys.modules["torch.utils"] = torch_stub.utils
    sys.modules["torch.utils._pytree"] = torch_stub.utils._pytree

    patched_attrs: List[tuple[Any, str, Any]] = []
    replacement_fns = {
        "is_torch_available": lambda: False,
        "get_torch_version": lambda: "0.0.0",
        "is_torch_accelerator_available": lambda: False,
        "is_torch_cuda_available": lambda: False,
        "is_cuda_platform": lambda: False,
    }
    for module_name in ("transformers", "transformers.utils", "transformers.utils.import_utils"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attr_name, attr_value in replacement_fns.items():
            if hasattr(module, attr_name):
                patched_attrs.append((module, attr_name, getattr(module, attr_name)))
                setattr(module, attr_name, attr_value)

    def _restore() -> None:
        for module, attr_name, attr_value in reversed(patched_attrs):
            setattr(module, attr_name, attr_value)
        for module_name in reversed(stubbed_names):
            if module_name in original_modules:
                sys.modules[module_name] = original_modules[module_name]
            else:
                sys.modules.pop(module_name, None)

    return _restore


def _install_text_only_scipy_stub(model_ref: str) -> Callable[[], None]:
    if not _should_force_qwen2_tokenizer(model_ref):
        return lambda: None

    original_modules: Dict[str, Any] = {}
    stubbed_names = [
        "scipy",
        "scipy.linalg",
        "scipy.ndimage",
    ]
    for module_name in stubbed_names:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]

    def _make_module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)
        return module

    scipy_stub = _make_module("scipy")
    scipy_stub.__dict__["__version__"] = "0.0.0"
    scipy_stub.linalg = _make_module("scipy.linalg")
    scipy_stub.ndimage = _make_module("scipy.ndimage")

    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.linalg"] = scipy_stub.linalg
    sys.modules["scipy.ndimage"] = scipy_stub.ndimage

    patched_attrs: List[tuple[Any, str, Any]] = []
    replacement_fns = {
        "is_scipy_available": lambda: False,
    }
    for module_name in ("transformers", "transformers.utils", "transformers.utils.import_utils"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attr_name, attr_value in replacement_fns.items():
            if hasattr(module, attr_name):
                patched_attrs.append((module, attr_name, getattr(module, attr_name)))
                setattr(module, attr_name, attr_value)

    def _restore() -> None:
        for module, attr_name, attr_value in reversed(patched_attrs):
            setattr(module, attr_name, attr_value)
        for module_name in reversed(stubbed_names):
            if module_name in original_modules:
                sys.modules[module_name] = original_modules[module_name]
            else:
                sys.modules.pop(module_name, None)

    return _restore


@dataclass
class LocalGenerationResult:
    status: str
    executed_locally: bool
    backend: str
    text: str
    chunks: List[str]
    evidence_path: str
    reason: str = ""
    model_ref: str = ""


class EdgeLocalInferenceRuntime:
    def __init__(self) -> None:
        self.evidence_root = _resolve_local_infer_evidence_root()
        self._mlx_lm_cache: Dict[str, tuple[Any, Any]] = {}
        self._mlx_lm_cache_lock = threading.Lock()
        self._omlx_engine: Any = None  # OMLXMLXEngine 引用 (streaming mode)

    def _write_evidence(self, payload: Dict[str, Any]) -> str:
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        path = self.evidence_root / f"local_infer_{stamp}_{int(time.time() * 1000) % 1000:03d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path.resolve())

    def _chunk_text(self, text: str) -> List[str]:
        pieces = re.split(r"(\s+)", str(text or ""))
        return [piece for piece in pieces if piece]

    async def resume_from_kda_state(
        self,
        *,
        state_kind: str,
        state_codec: str,
        state_bytes: bytes | bytearray | memoryview,
        state_meta: Dict[str, Any] | None = None,
        trace_id: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        def _worker() -> Dict[str, Any]:
            merged_state_summary = {
                **(state_meta if isinstance(state_meta, dict) else {}),
            }
            resume_result = resume_one_token_from_kda_state(
                state_kind=state_kind,
                state_codec=state_codec,
                state_bytes=state_bytes,
                trace_id=trace_id,
            )
            runtime_state_summary = resume_result.get("state_summary") if isinstance(resume_result.get("state_summary"), dict) else {}
            merged_state_summary = {
                **runtime_state_summary,
                **merged_state_summary,
            }
            zero_copy_runtime = resume_result.get("zero_copy_runtime") if isinstance(resume_result.get("zero_copy_runtime"), dict) else {}
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "PASS",
                "mode": "m75_trueorthokda_single_step_resume",
                "trace_id": str(trace_id),
                "state_kind": state_kind,
                "state_codec": state_codec,
                "state_source": str(merged_state_summary.get("state_source") or ""),
                "state_meta": merged_state_summary,
                "max_tokens": int(max_tokens),
                "resume_decode_executed": bool(resume_result.get("resume_decode_executed")),
                "reason": "single_step_decode_completed",
                "seed_token_id": int(resume_result.get("seed_token_id") or 0),
                "edge_token_id": int(resume_result.get("edge_token_id") or 0),
                "next_state_shape": list(resume_result.get("next_state_shape") or []),
                "state_payload_bytes": len(state_bytes),
                "raw_state_bytes": int(merged_state_summary.get("raw_state_bytes") or len(state_bytes)),
                "compressed_state_bytes": int(merged_state_summary.get("compressed_state_bytes") or len(state_bytes)),
                "compression_ratio": float(merged_state_summary.get("compression_ratio") or 1.0),
                "selected_payload_bytes": int(merged_state_summary.get("selected_payload_bytes") or len(state_bytes)),
                "compression_codec_selected": str(merged_state_summary.get("compression_codec_selected") or state_codec),
                "cpu_copy_count": int(zero_copy_runtime.get("cpu_copy_count") or 0),
                "uma_buffer_used": bool(zero_copy_runtime.get("uma_buffer_used")),
                "device_resume_consumed": bool(zero_copy_runtime.get("device_resume_consumed")),
                "resume_tensor_device": str(resume_result.get("resume_tensor_device") or "cpu"),
            }
            evidence_path = self._write_evidence(payload)
            return {
                "status": "PASS",
                "executed_locally": True,
                "backend": "kda_state_single_step_resume",
                "text": str(resume_result.get("text") or ""),
                "chunks": self._chunk_text(str(resume_result.get("text") or "")),
                "reason": "single_step_decode_completed",
                "state_summary": merged_state_summary,
                "seed_token_id": int(resume_result.get("seed_token_id") or 0),
                "edge_token_id": int(resume_result.get("edge_token_id") or 0),
                "next_state_shape": list(resume_result.get("next_state_shape") or []),
                "evidence_path": evidence_path,
            }

        return await asyncio.to_thread(_worker)

    def _base_payload(
        self,
        *,
        model: str,
        prompt: str,
        use_omlx: bool,
        use_flashmoe: bool,
        max_tokens: int,
    ) -> Dict[str, Any]:
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "SKIP",
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "requested": {
                "model": str(model),
                "use_omlx": bool(use_omlx),
                "use_flashmoe": bool(use_flashmoe),
                "max_tokens": int(max_tokens),
                "prompt_preview": str(prompt)[:200],
            },
        }

    def _can_attempt_local(self, model: str, use_omlx: bool, use_flashmoe: bool) -> bool:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        model_str = str(model or "").strip()
        if use_omlx or use_flashmoe:
            return True
        lowered = model_str.lower()
        return lowered.endswith(".mlx") or lowered.endswith(".gguf") or "mlx-community/" in lowered

    # --- Gate 2.0 stage 2: by-layer dynamic partition (max_local_layer) ---

    def _get_vram_watermark(self) -> Dict[str, float]:
        """返回端侧显存/内存水位（Apple Silicon UMA 统一内存）。

        优先级：mlx.active_memory > psutil > vm_stat > 未知。
        单位：MB。返回 {total_mb, available_mb, used_mb, source}。
        """
        info: Dict[str, float] = {"total_mb": 0.0, "available_mb": 0.0, "used_mb": 0.0, "source": "unknown"}
        # 1. mlx（Apple Silicon Metal）
        try:
            import mlx.core as mx  # type: ignore
            used_bytes = int(mx.get_active_memory()) if hasattr(mx, "get_active_memory") else 0
            # mlx 不提供总量，用 sysctl 补
            total_bytes = self._sysctl_memsize()
            if total_bytes > 0:
                info.update({"total_mb": total_bytes / 1048576.0, "used_mb": used_bytes / 1048576.0,
                             "available_mb": (total_bytes - used_bytes) / 1048576.0, "source": "mlx"})
                return info
        except Exception:
            pass
        # 2. psutil
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            info.update({"total_mb": vm.total / 1048576.0, "available_mb": vm.available / 1048576.0,
                         "used_mb": vm.used / 1048576.0, "source": "psutil"})
            return info
        except Exception:
            pass
        # 3. vm_stat（macOS 兜底）
        try:
            total_bytes = self._sysctl_memsize()
            if total_bytes > 0:
                # 粗略：无法精确取 available，返回 total，used 留 0
                info.update({"total_mb": total_bytes / 1048576.0, "source": "vm_stat"})
                return info
        except Exception:
            pass
        return info

    @staticmethod
    def _sysctl_memsize() -> int:
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL, timeout=2)
            return int(out.strip())
        except Exception:
            return 0

    def _calc_max_safe_layers(
        self,
        *,
        vram_watermark: Dict[str, float],
        num_total_layers: int,
        layer_mem_mb: float = 0.0,
        safety_threshold: float = 0.80,
    ) -> int:
        """根据显存水位计算端侧可安全执行的最大层数。

        - safety_threshold: 显存使用水位上限（默认 0.80，超过即停止向后执行）
        - layer_mem_mb: 单层估算显存（权重+KV+激活）；0 时按均匀切分启发式
        - 返回 [0, num_total_layers]，0 表示端侧无法执行任何层
        """
        total_mb = float(vram_watermark.get("total_mb") or 0.0)
        used_mb = float(vram_watermark.get("used_mb") or 0.0)
        if total_mb <= 0:
            return 0
        budget_mb = total_mb * float(safety_threshold) - used_mb
        if budget_mb <= 0:
            return 0
        if layer_mem_mb > 0:
            max_layers = int(budget_mb // layer_mem_mb)
        else:
            # 启发式：可用预算占总量的比例 × 总层数
            ratio = budget_mb / total_mb
            max_layers = int(ratio * num_total_layers)
        return max(0, min(max_layers, num_total_layers))

    def _resolve_layer_decision(
        self,
        *,
        model: str,
        use_omlx: bool,
        use_flashmoe: bool,
        num_total_layers: int = 0,
        layer_mem_mb: float = 0.0,
        safety_threshold: float = 0.80,
    ) -> Dict[str, Any]:
        """端侧层粒度决策：返回 max_local_layer + 水位快照。

        若 num_total_layers<=0 或端侧不可执行，max_local_layer=0（全部上云）。
        若预算充足覆盖全部层，max_local_layer=num_total_layers（全部本地）。
        """
        can_local = self._can_attempt_local(model, use_omlx, use_flashmoe)
        watermark = self._get_vram_watermark()
        if not can_local or num_total_layers <= 0:
            return {
                "max_local_layer": 0,
                "can_attempt_local": can_local,
                "vram_watermark": watermark,
                "decision": "all_cloud",
                "reason": "edge cannot execute or num_total_layers<=0",
            }
        max_local = self._calc_max_safe_layers(
            vram_watermark=watermark,
            num_total_layers=num_total_layers,
            layer_mem_mb=layer_mem_mb,
            safety_threshold=safety_threshold,
        )
        if max_local >= num_total_layers:
            decision = "all_local"
            reason = f"vram budget covers all {num_total_layers} layers"
        elif max_local == 0:
            decision = "all_cloud"
            reason = "vram budget insufficient for any layer"
        else:
            decision = "split"
            reason = f"edge runs first {max_local} layers, cloud continues from layer {max_local + 1}"
        return {
            "max_local_layer": max_local,
            "can_attempt_local": True,
            "vram_watermark": watermark,
            "num_total_layers": num_total_layers,
            "decision": decision,
            "reason": reason,
        }

    # --- Gate 2.0 stage 2: edge→cloud layer handoff ---

    def upload_hidden_states_for_cloud_continuation(
        self,
        *,
        finished_layer: int,
        hidden_states: Any,
        partial_kv: Any = None,
        model_id: str = "",
        cloud_endpoint: str = "",
        layer_metadata_list: Any = None,
    ) -> Dict[str, Any]:
        """端侧执行完前 finished_layer 层后，上传 hidden_states 到云侧接续 Prefill。

        对应能力 hidden_states_partial_kv_abi + finished_layer_prefill_continuation。

        Args:
            finished_layer: 端侧已完成的最后一层 index（0-based）
            hidden_states: 端侧最后一层输出，shape=[num_tokens, hidden_dim]，支持 torch.Tensor / numpy.ndarray
            partial_kv: 端侧已生成 KV cache（可选）
            model_id: 目标模型 ID，云侧校验同模型
            cloud_endpoint: 云侧 cgc_api_server 地址；空则读 CGC_CLOUD_ENDPOINT 环境变量
            layer_metadata_list: 每层元数据（layer_id, layer_norm_eps, layer_norm_weight_hash 等）

        Returns:
            云侧响应 {cloud_request_id, accepted_layer, start_layer_on_cloud, ...}
        """
        # 延迟 import：仅在端侧实际触发上传时引入 transport 模块
        try:
            import numpy as _np
        except ImportError:
            _np = None  # type: ignore

        # torch.Tensor → numpy.ndarray
        if hasattr(hidden_states, "detach"):
            hidden_states_np = hidden_states.detach().cpu().numpy()
        elif _np is not None and isinstance(hidden_states, _np.ndarray):
            hidden_states_np = hidden_states
        else:
            raise TypeError(f"hidden_states must be torch.Tensor or numpy.ndarray, got {type(hidden_states)}")

        if _np is None:
            raise RuntimeError("numpy is required for edge→cloud handoff serialization")

        # partial_kv 同样转 numpy
        partial_kv_np = None
        if partial_kv is not None:
            partial_kv_np = {}
            for k, v in partial_kv.items():
                if hasattr(v, "detach"):
                    partial_kv_np[k] = v.detach().cpu().numpy()
                elif isinstance(v, _np.ndarray):
                    partial_kv_np[k] = v
                else:
                    raise TypeError(f"partial_kv[{k}] must be Tensor/ndarray, got {type(v)}")

        # 构造 EdgeCloudLayerHandoff
        try:
            from Backend.CGC.edge_moe_transport import (
                EdgeCloudLayerHandoff as _Handoff,
                serialize_handoff as _serialize,
            )
        except ImportError:
            # 退化路径：直接走 cgc_engine 的 lightweight 实现
            sys.path.insert(0, str(ENGINE_ROOT / "Backend" / "CGC"))
            from edge_moe_transport import (  # type: ignore
                EdgeCloudLayerHandoff as _Handoff,
                serialize_handoff as _serialize,
            )

        handoff = _Handoff(
            finished_layer=int(finished_layer),
            hidden_states=hidden_states_np,
            partial_kv=partial_kv_np,
            layer_metadata_list=layer_metadata_list or [],
            model_id=str(model_id),
        )
        handoff.validate()

        # 通过 CQ4 session 上传
        endpoint = cloud_endpoint or os.environ.get("CGC_CLOUD_ENDPOINT", "http://127.0.0.1:7777")
        try:
            from Backend.CGC.edge_moe_transport.cq4_session import (
                CQ4Session as _CQ4Session,
                CQ4SessionConfig as _CQ4Config,
                CQ4QoSClass as _QoS,
            )
        except ImportError:
            from edge_moe_transport.cq4_session import (  # type: ignore
                CQ4Session as _CQ4Session,
                CQ4SessionConfig as _CQ4Config,
                CQ4QoSClass as _QoS,
            )

        session = _CQ4Session(_CQ4Config(cloud_endpoint=endpoint))
        try:
            resp = session.send_handoff(handoff, qos=_QoS.HIDDEN_STATES)
        finally:
            session.close()
        return resp

    def _resolve_model_ref(self, model: str, *, use_flashmoe: bool) -> str:
        model_str = str(model or "").strip()
        override_key = "CGC_LOCAL_FLASHMOE_MODEL" if use_flashmoe else "CGC_LOCAL_OMLX_MODEL"
        local_override = str(os.environ.get(override_key) or "").strip()
        if local_override:
            return local_override
        if Path(model_str).expanduser().exists():
            return str(Path(model_str).expanduser().resolve())
        return _resolve_cached_hf_snapshot(model_str)

    def _can_attempt_mlx_model_ref(self, model_ref: str) -> bool:
        model_path = Path(str(model_ref or "")).expanduser()
        if model_path.exists() and model_path.is_file():
            return False
        lowered = str(model_ref or "").strip().lower()
        if lowered.endswith(".gguf"):
            return False
        return True

    def _resolve_llama_cli_path(self) -> str:
        explicit = str(os.environ.get("CGC_LOCAL_LLAMA_CPP_BIN") or "").strip()
        if explicit:
            return explicit
        configured = str(_read_json_dict(CGC_CONFIG_PATH).get("active_edge_backend_binary_path") or "").strip()
        if configured:
            return configured
        return str(shutil.which("llama-cli") or "").strip()

    def _can_attempt_llama_cpp_model_ref(self, model_ref: str) -> bool:
        model_path = Path(str(model_ref or "")).expanduser()
        lowered = str(model_ref or "").strip().lower()
        return bool(
            model_path.exists()
            and model_path.is_file()
            and lowered.endswith(".gguf")
            and self._resolve_llama_cli_path()
        )

    async def _run_llama_cpp(self, *, model_ref: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
        def _worker() -> Dict[str, Any]:
            model_path = Path(str(model_ref)).expanduser().resolve()
            if not model_path.exists() or not model_path.is_file():
                return {"status": "SKIP", "reason": "llama_cpp_requires_local_gguf_file"}
            if model_path.suffix.lower() != ".gguf":
                return {"status": "SKIP", "reason": "llama_cpp_requires_gguf_model"}
            cli_path = self._resolve_llama_cli_path()
            if not cli_path:
                return {"status": "SKIP", "reason": "llama_cpp_cli_not_found"}

            command = [
                cli_path,
                "--model",
                str(model_path),
                "--prompt",
                str(prompt),
                "--n-predict",
                str(int(max_tokens)),
                "--temp",
                "0",
                "--top-k",
                "1",
                "--top-p",
                "1.0",
                "--seed",
                "0",
                "--simple-io",
                "--single-turn",
                "--no-display-prompt",
                "--no-warmup",
                "--log-disable",
            ]
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(60, int(max_tokens) * 8),
                env={
                    **os.environ,
                    "LLAMA_LOG_COLORS": "0",
                },
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            stdout_text = str(completed.stdout or "")
            stderr_text = str(completed.stderr or "")
            generated_text = stdout_text.strip()
            if completed.returncode != 0:
                error_excerpt = "\n".join(
                    part
                    for part in [stderr_text.strip(), stdout_text.strip()]
                    if part
                ).strip()
                return {
                    "status": "FAIL",
                    "reason": f"llama_cpp_runtime_error:{error_excerpt or completed.returncode}",
                    "command": command,
                    "returncode": int(completed.returncode),
                }
            if not generated_text:
                return {
                    "status": "FAIL",
                    "reason": "llama_cpp_returned_empty_text",
                    "command": command,
                    "stderr_excerpt": stderr_text[-400:],
                }
            return {
                "status": "PASS",
                "backend": "llama.cpp_cli",
                "text": generated_text,
                "stats": {
                    "elapsed_ms": round(elapsed_ms, 3),
                    "finish_reason": "stop",
                    "command_path": cli_path,
                    "generation_tokens": int(max_tokens),
                },
            }

        return await asyncio.to_thread(_worker)

    async def _run_mlx_lm(self, *, model_ref: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
        def _worker() -> Dict[str, Any]:
            # #region debug-point A:mlx-entry
            _model_path = Path(str(model_ref)).expanduser()
            _debug_report_m8_gguf_fallback(
                hypothesis_id="A",
                location="app/edge_engine/local_infer.py:_run_mlx_lm",
                message="enter mlx loader",
                data={
                    "model_ref": str(model_ref),
                    "exists": _model_path.exists(),
                    "is_file": _model_path.is_file(),
                    "is_dir": _model_path.is_dir(),
                    "suffix": _model_path.suffix.lower(),
                    "max_tokens": int(max_tokens),
                },
            )
            # #endregion
            restore_scipy_stub = _install_text_only_scipy_stub(model_ref)
            restore_torch_stub = _install_text_only_torch_stub(model_ref)
            restore_tokenizer_shim = _install_mlx_lm_qwen2_tokenizer_shim(model_ref)
            import mlx.core as mx
            import mlx_lm
            from mlx_lm.generate import stream_generate

            mx.reset_peak_memory()
            try:
                # oMLX+FlashMoE: 包装 mlx_lm + StreamingSwitchGLU 注入
                use_omlx_streaming = os.environ.get("EDGE_USE_OMLX_MLX") == "1"
                omlx_cache_key = f"omlx_streaming::{model_ref}" if use_omlx_streaming else f"::{model_ref}"

                cache_key = omlx_cache_key
                with self._mlx_lm_cache_lock:
                    cached_pair = self._mlx_lm_cache.get(cache_key)

                if cached_pair is None:
                    if use_omlx_streaming:
                        from app.edge_engine.omlx_mlx_engine import OMLXMLXEngine
                        engine = OMLXMLXEngine(
                            model_path=model_ref,
                            enable_streaming=True,
                            streaming_config={
                                "max_experts_in_memory": int(os.environ.get("EDGE_OMLX_CACHE_SIZE", "2")),
                                "swap_time_per_expert_ms": float(os.environ.get("EDGE_OMLX_SWAP_IO_MS", "0.0")),
                                "enable_io_simulation": os.environ.get("EDGE_OMLX_SIMULATE_IO") == "1",
                            },
                        )
                        engine.load()
                        model_obj = engine.model
                        tokenizer = engine.tokenizer
                        self._omlx_engine = engine  # 保存引用防止 GC
                    else:
                        model_obj, tokenizer = mlx_lm.load(model_ref, lazy=True)
                    with self._mlx_lm_cache_lock:
                        self._mlx_lm_cache[cache_key] = (model_obj, tokenizer)
                else:
                    model_obj, tokenizer = cached_pair
                final_text = ""
                final_stats: Dict[str, Any] = {}
                for resp in stream_generate(model_obj, tokenizer, prompt, max_tokens=int(max_tokens)):
                    final_text += str(getattr(resp, "text", "") or "")
                    final_stats = {
                        "prompt_tokens": int(getattr(resp, "prompt_tokens", 0) or 0),
                        "generation_tokens": int(getattr(resp, "generation_tokens", 0) or 0),
                        "generation_tps": float(getattr(resp, "generation_tps", 0.0) or 0.0),
                        "peak_memory_gb": float(getattr(resp, "peak_memory", 0.0) or 0.0),
                        "finish_reason": str(getattr(resp, "finish_reason", "stop") or "stop"),
                    }
                if final_text == "":
                    raise RuntimeError("mlx_lm_stream_generate_returned_empty_text")

                # 收集 streaming stats (如果 oMLX engine 启用)
                streaming_stats_str = "N/A"
                backend_name = "omlx_mlx_lm"
                if use_omlx_streaming and self._omlx_engine is not None:
                    stats = self._omlx_engine.get_stats()
                    if stats:
                        streaming_stats_str = stats.summary()
                    backend_name = "omlx_flashmoe_streaming"

                return {
                    "status": "PASS",
                    "backend": backend_name,
                    "text": final_text,
                    "stats": final_stats,
                    "streaming_stats": streaming_stats_str,
                }
            finally:
                restore_tokenizer_shim()
                restore_torch_stub()
                restore_scipy_stub()

        return await asyncio.to_thread(_worker)

    async def _run_dflash(self, *, model_ref: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
        from omlx.engine.dflash import DFlashEngine, is_dflash_compatible

        model_path = Path(model_ref).expanduser()
        if not model_path.exists() or not model_path.is_dir():
            return {"status": "SKIP", "reason": "dflash_requires_local_model_directory"}

        ok, reason = is_dflash_compatible(model_path)
        if not ok:
            return {"status": "SKIP", "reason": reason or "dflash_model_not_compatible"}

        cache_dir = self.evidence_root / "omlx_ssd_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        engine = DFlashEngine(
            model_name=str(model_path.resolve()),
            draft_model_path=str(model_path.resolve()),
            omlx_ssd_cache_dir=str(cache_dir.resolve()),
        )
        try:
            await engine.start()
            output = await engine.generate(prompt, max_tokens=int(max_tokens))
            text = str(getattr(output, "text", "") or "")
            if text == "":
                raise RuntimeError("dflash_generate_returned_empty_text")
            return {
                "status": "PASS",
                "backend": "omlx_dflash_flashmoe",
                "text": text,
                "stats": {
                    "prompt_tokens": int(getattr(output, "prompt_tokens", 0) or 0),
                    "generation_tokens": int(getattr(output, "completion_tokens", 0) or 0),
                    "finish_reason": str(getattr(output, "finish_reason", "stop") or "stop"),
                    "ssd_cache_dir": str(cache_dir.resolve()),
                },
            }
        finally:
            try:
                await engine.stop()
            except Exception:
                pass

    async def maybe_generate(
        self,
        *,
        model: str,
        prompt: str,
        use_omlx: bool,
        use_flashmoe: bool,
        max_tokens: int,
    ) -> LocalGenerationResult:
        payload = self._base_payload(
            model=model,
            prompt=prompt,
            use_omlx=use_omlx,
            use_flashmoe=use_flashmoe,
            max_tokens=max_tokens,
        )
        if not self._can_attempt_local(model, use_omlx, use_flashmoe):
            payload["reason"] = "local_omlx_flashmoe_not_requested_or_platform_unsupported"
            evidence_path = self._write_evidence(payload)
            return LocalGenerationResult(
                status="SKIP",
                executed_locally=False,
                backend="",
                text="",
                chunks=[],
                evidence_path=evidence_path,
                reason=str(payload["reason"]),
                model_ref="",
            )

        model_ref = self._resolve_model_ref(model, use_flashmoe=bool(use_flashmoe))
        attempts: List[Dict[str, Any]] = []
        llama_attempt_enabled = self._can_attempt_llama_cpp_model_ref(model_ref)
        mlx_attempt_enabled = self._can_attempt_mlx_model_ref(model_ref)
        # #region debug-point B:local-dispatch
        _resolved_path = Path(str(model_ref)).expanduser()
        _debug_report_m8_gguf_fallback(
            hypothesis_id="B",
            location="app/edge_engine/local_infer.py:maybe_generate",
            message="resolved local model dispatch",
            data={
                "requested_model": str(model),
                "model_ref": str(model_ref),
                "exists": _resolved_path.exists(),
                "is_file": _resolved_path.is_file(),
                "is_dir": _resolved_path.is_dir(),
                "suffix": _resolved_path.suffix.lower(),
                "use_omlx": bool(use_omlx),
                "use_flashmoe": bool(use_flashmoe),
                "llama_attempt_enabled": bool(llama_attempt_enabled),
                "mlx_attempt_enabled": bool(mlx_attempt_enabled),
                "attempt_order": [
                    "dflash" if bool(use_flashmoe) else None,
                    "llama.cpp" if bool(llama_attempt_enabled) else None,
                    "mlx_lm" if bool(mlx_attempt_enabled) else "mlx_lm_skipped",
                ],
            },
        )
        # #endregion
        # #region debug-point E:local-dispatch
        _debug_report_dense_streaming_measure(
            hypothesis_id="E",
            location="app/edge_engine/local_infer.py:maybe_generate",
            message="resolved local runtime backend candidates",
            data={
                "requested_model": str(model),
                "model_ref": str(model_ref),
                "use_omlx": bool(use_omlx),
                "use_flashmoe": bool(use_flashmoe),
                "llama_attempt_enabled": bool(llama_attempt_enabled),
                "mlx_attempt_enabled": bool(mlx_attempt_enabled),
                "attempt_order": [
                    "dflash" if bool(use_flashmoe) else None,
                    "llama.cpp" if bool(llama_attempt_enabled) else None,
                    "mlx_lm" if bool(mlx_attempt_enabled) else None,
                ],
            },
        )
        # #endregion
        if use_flashmoe:
            try:
                dflash = await self._run_dflash(model_ref=model_ref, prompt=prompt, max_tokens=max_tokens)
            except Exception as exc:
                dflash = {
                    "status": "FAIL",
                    "reason": f"dflash_runtime_error:{exc}",
                    "traceback": traceback.format_exc(),
                }
            attempts.append(dflash)
            # #region debug-point E:dflash-attempt
            _debug_report_dense_streaming_measure(
                hypothesis_id="E",
                location="app/edge_engine/local_infer.py:maybe_generate",
                message="completed dflash attempt",
                data={
                    "status": str(dflash.get("status") or ""),
                    "reason": str(dflash.get("reason") or ""),
                    "backend": str(dflash.get("backend") or ""),
                    "use_flashmoe": bool(use_flashmoe),
                },
            )
            # #endregion
            if str(dflash.get("status") or "") == "PASS":
                payload.update(
                    {
                        "status": "PASS",
                        "executed_locally": True,
                        "backend": dflash["backend"],
                        "model_ref": model_ref,
                        "stats": dflash.get("stats", {}),
                        "text_preview": str(dflash.get("text") or "")[:300],
                    }
                )
                evidence_path = self._write_evidence(payload)
                return LocalGenerationResult(
                    status="PASS",
                    executed_locally=True,
                    backend=str(dflash["backend"]),
                    text=str(dflash.get("text") or ""),
                    chunks=self._chunk_text(str(dflash.get("text") or "")),
                    evidence_path=evidence_path,
                    model_ref=model_ref,
                )

        if llama_attempt_enabled:
            try:
                llama_result = await self._run_llama_cpp(model_ref=model_ref, prompt=prompt, max_tokens=max_tokens)
            except Exception as exc:
                llama_result = {
                    "status": "FAIL",
                    "reason": f"llama_cpp_runtime_error:{exc}",
                    "traceback": traceback.format_exc(),
                }
        else:
            llama_result = {
                "status": "SKIP",
                "reason": "llama_cpp_requires_local_gguf_and_cli",
            }
        attempts.append(llama_result)
        # #region debug-point E:llama-attempt
        _debug_report_dense_streaming_measure(
            hypothesis_id="E",
            location="app/edge_engine/local_infer.py:maybe_generate",
            message="completed llama.cpp fallback attempt",
            data={
                "status": str(llama_result.get("status") or ""),
                "reason": str(llama_result.get("reason") or ""),
                "backend": str(llama_result.get("backend") or ""),
            },
        )
        # #endregion
        if str(llama_result.get("status") or "") == "PASS":
            payload.update(
                {
                    "status": "PASS",
                    "executed_locally": True,
                    "backend": llama_result["backend"],
                    "model_ref": model_ref,
                    "stats": llama_result.get("stats", {}),
                    "text_preview": str(llama_result.get("text") or "")[:300],
                }
            )
            evidence_path = self._write_evidence(payload)
            return LocalGenerationResult(
                status="PASS",
                executed_locally=True,
                backend=str(llama_result["backend"]),
                text=str(llama_result.get("text") or ""),
                chunks=self._chunk_text(str(llama_result.get("text") or "")),
                evidence_path=evidence_path,
                model_ref=model_ref,
            )

        if mlx_attempt_enabled:
            try:
                mlx_result = await self._run_mlx_lm(model_ref=model_ref, prompt=prompt, max_tokens=max_tokens)
            except Exception as exc:
                mlx_result = {
                    "status": "FAIL",
                    "reason": f"omlx_runtime_error:{exc}",
                    "traceback": traceback.format_exc(),
                }
        else:
            mlx_result = {
                "status": "SKIP",
                "reason": "omlx_requires_model_directory_or_non_gguf_reference",
            }
        attempts.append(mlx_result)
        # #region debug-point E:mlx-attempt
        _debug_report_dense_streaming_measure(
            hypothesis_id="E",
            location="app/edge_engine/local_infer.py:maybe_generate",
            message="completed mlx runtime attempt",
            data={
                "status": str(mlx_result.get("status") or ""),
                "reason": str(mlx_result.get("reason") or ""),
                "backend": str(mlx_result.get("backend") or ""),
            },
        )
        # #endregion
        if str(mlx_result.get("status") or "") == "PASS":
            payload.update(
                {
                    "status": "PASS",
                    "executed_locally": True,
                    "backend": mlx_result["backend"],
                    "model_ref": model_ref,
                    "stats": mlx_result.get("stats", {}),
                    "text_preview": str(mlx_result.get("text") or "")[:300],
                }
            )
            evidence_path = self._write_evidence(payload)
            return LocalGenerationResult(
                status="PASS",
                executed_locally=True,
                backend=str(mlx_result["backend"]),
                text=str(mlx_result.get("text") or ""),
                chunks=self._chunk_text(str(mlx_result.get("text") or "")),
                evidence_path=evidence_path,
                model_ref=model_ref,
            )

        payload.update(
            {
                "status": "FAIL",
                "executed_locally": False,
                "model_ref": model_ref,
                "attempts": attempts,
                "reason": "all_local_inference_backends_failed",
            }
        )
        # #region debug-point C:local-fail
        _debug_report_m8_gguf_fallback(
            hypothesis_id="C",
            location="app/edge_engine/local_infer.py:maybe_generate",
            message="local inference failed before fallback",
            data={
                "requested_model": str(model),
                "model_ref": str(model_ref),
                "reason": str(payload.get("reason") or ""),
                "attempt_statuses": [str((attempt or {}).get("status") or "") for attempt in attempts],
                "attempt_reasons": [str((attempt or {}).get("reason") or "") for attempt in attempts],
            },
        )
        # #endregion
        evidence_path = self._write_evidence(payload)
        return LocalGenerationResult(
            status="FAIL",
            executed_locally=False,
            backend="",
            text="",
            chunks=[],
            evidence_path=evidence_path,
            reason=str(payload["reason"]),
            model_ref=model_ref,
        )
