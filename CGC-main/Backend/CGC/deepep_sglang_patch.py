import argparse
import importlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

LOGGER = logging.getLogger(__name__)
ENGINE_REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_SGLANG_PYTHON = (ENGINE_REPO_ROOT / "Backend" / "CGC" / "cloud_sglang" / "python").resolve()
torch = None
dist = None
deep_ep = None

_ELASTIC_BUFFER_CACHE: Dict[tuple, Any] = {}


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = str(os.environ.get(name, str(default)) or "").strip()
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    return max(1, value)


def resolve_deepep_parallelism(
    *,
    tp_size: Optional[int] = None,
    ep_size: Optional[int] = None,
    deepep_parallel_profile: Optional[str] = None,
) -> Dict[str, Any]:
    profile_raw = str(
        deepep_parallel_profile
        or os.environ.get("CGC_DEEPEP_PARALLEL_PROFILE", "")
        or ""
    ).strip().lower()
    profile_match = re.fullmatch(r"ep(\d+)_tp(\d+)", profile_raw)
    profile_ep = int(profile_match.group(1)) if profile_match else None
    profile_tp = int(profile_match.group(2)) if profile_match else None

    resolved_tp_size = max(
        1,
        int(
            tp_size
            or os.environ.get("CGC_DEEPEP_TP_SIZE", "")
            or os.environ.get("CGC_SGLANG_TP_SIZE", "")
            or os.environ.get("CGC_MEGATRAIN_PARALLEL_TP_SIZE", "")
            or profile_tp
            or 4
        ),
    )
    resolved_ep_size = max(
        1,
        int(
            ep_size
            or os.environ.get("CGC_DEEPEP_EP_SIZE", "")
            or os.environ.get("CGC_SGLANG_EP_SIZE", "")
            or os.environ.get("CGC_MEGATRAIN_PARALLEL_EP_SIZE", "")
            or profile_ep
            or resolved_tp_size
        ),
    )
    parallel_profile = profile_raw or f"ep{resolved_ep_size}_tp{resolved_tp_size}"
    return {
        "deepep_parallel_profile": parallel_profile,
        "ep_size": resolved_ep_size,
        "tp_size": resolved_tp_size,
    }


def ensure_vendored_sglang_on_path() -> str:
    path = str(VENDORED_SGLANG_PYTHON)
    if path not in sys.path:
        sys.path.insert(0, path)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [entry for entry in current_pythonpath.split(os.pathsep) if entry]
    if path not in pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([path, *pythonpath_entries])
    return path


def _candidate_site_packages() -> list[str]:
    major = sys.version_info.major
    minor = sys.version_info.minor
    version = f"python{major}.{minor}"
    roots = [
        str(os.environ.get("VIRTUAL_ENV", "") or "").strip(),
        str(os.environ.get("CGC_SGLANG_PYTHON_BIN", "") or "").strip(),
        str(ENGINE_REPO_ROOT.parent / ".venv_deepep_ssp"),
        str(ENGINE_REPO_ROOT / ".venv_deepep_ssp"),
    ]
    paths: list[str] = []
    seen: set[str] = set()
    for raw in roots:
        if raw == "":
            continue
        candidate = Path(raw).expanduser()
        if candidate.name == "python" and candidate.parent.name == "bin":
            candidate = candidate.parent.parent
        site_packages = str(candidate / "lib" / version / "site-packages")
        if site_packages in seen:
            continue
        seen.add(site_packages)
        paths.append(site_packages)
    return paths


def _prepend_site_packages() -> None:
    for path in _candidate_site_packages():
        if not Path(path).exists():
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        current = os.environ.get("PYTHONPATH", "")
        entries = [entry for entry in current.split(os.pathsep) if entry]
        if path not in entries:
            os.environ["PYTHONPATH"] = os.pathsep.join([path, *entries])


def _load_torch_modules() -> bool:
    global torch, dist
    _prepend_site_packages()
    try:
        torch = importlib.import_module("torch")
        dist = importlib.import_module("torch.distributed")
        return True
    except ImportError:
        torch = None
        dist = None
        return False


def _load_deepep_module() -> bool:
    global deep_ep
    _prepend_site_packages()
    try:
        deep_ep = importlib.import_module("deep_ep")
        return True
    except ImportError:
        deep_ep = None
        return False


def _require_torch_and_deepep() -> None:
    if not _load_torch_modules():
        raise RuntimeError("PyTorch is required for DeepEP integration.")
    if os.environ.get("CGC_MOE_A2A_BACKEND", "deepep") != "custom":
        if not _load_deepep_module():
            LOGGER.warning("[M7.6 Gate] DeepEP runtime not found. Falling back to native SGLang routing.")
            raise RuntimeError("DeepEP is required for DeepEP integration.")


def select_model_path(candidates: Optional[Iterable[str]] = None) -> str:
    if candidates is None:
        candidates = (
            os.environ.get("CGC_CLOUD_MODEL_PATH", "").strip(),
            "/data/models/DeepSeek-V4-Flash-UD-IQ2",
            "/data/models/DeepSeek-V4-Flash",
            "/data2/models/DeepSeek-V4-Flash-UD-IQ2",
            "/data2/models/DeepSeek-V4-Flash",
            "/root/models/DeepSeek-V4-Flash",
            "/data/models/Qwen2.5-7B-Instruct",
            "/root/models/Qwen2.5-7B-Instruct",
        )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    raise FileNotFoundError(
        "No usable model path found for cloud inference. Set CGC_CLOUD_MODEL_PATH explicitly."
    )


def build_sglang_deepep_engine_kwargs(
    *,
    tp_size: Optional[int] = None,
    ep_size: Optional[int] = None,
    deepep_parallel_profile: Optional[str] = None,
    deepep_mode: Optional[str] = None,
    enable_deepep_waterfill: Optional[bool] = None,
) -> Dict[str, Any]:
    _require_torch_and_deepep()
    ensure_vendored_sglang_on_path()

    # Keep these defaults overridable by the deployment environment.
    os.environ.setdefault("CGC_DEEPEP_ENABLED", "1")
    os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    resolved_parallelism = resolve_deepep_parallelism(
        tp_size=tp_size,
        ep_size=ep_size,
        deepep_parallel_profile=deepep_parallel_profile,
    )
    parallel_profile = str(resolved_parallelism["deepep_parallel_profile"])
    resolved_ep_size = int(resolved_parallelism["ep_size"])
    resolved_tp_size = int(resolved_parallelism["tp_size"])
    os.environ.setdefault("CGC_DEEPEP_PARALLEL_PROFILE", parallel_profile)
    os.environ.setdefault("CGC_DEEPEP_EP_SIZE", str(resolved_ep_size))
    os.environ.setdefault("CGC_DEEPEP_TP_SIZE", str(resolved_tp_size))

    mode = str(deepep_mode or os.environ.get("CGC_DEEPEP_MODE", "normal")).strip().lower()
    if mode in {"1", "true", "yes", "on", "enabled"}:
        mode = "auto"
    if mode not in {"auto", "normal", "low_latency"}:
        raise ValueError(f"Unsupported DeepEP mode: {mode}")
    waterfill = (
        _bool_env("CGC_ENABLE_DEEPEP_WATERFILL", False)
        if enable_deepep_waterfill is None
        else bool(enable_deepep_waterfill)
    )

    # Import from the vendored runtime so we always configure the fork that ships with this repo.
    import sglang
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer

    validation_targets = (sglang.__name__, DeepEPMoE.__name__, DeepEPBuffer.__name__)

    engine_kwargs = {
        "deepep_parallel_profile": parallel_profile,
        "tp_size": resolved_tp_size,
        "ep_size": resolved_ep_size,
        "moe_dense_tp_size": 1,
        "moe_a2a_backend": os.environ.get("CGC_MOE_A2A_BACKEND", "deepep"),
        "deepep_mode": mode,
        "enable_deepep_waterfill": waterfill,
    }
    LOGGER.info(
        "[DeepEP] Configured SGLang MoE runtime with vendored DeepEP backend: %s",
        engine_kwargs,
    )
    LOGGER.debug("[DeepEP] Validated SGLang runtime symbols: %s", validation_targets)
    return engine_kwargs


def patch_sglang_moe(
    *,
    tp_size: Optional[int] = None,
    ep_size: Optional[int] = None,
    deepep_parallel_profile: Optional[str] = None,
    deepep_mode: Optional[str] = None,
    enable_deepep_waterfill: Optional[bool] = None,
) -> Dict[str, Any]:
    engine_kwargs = build_sglang_deepep_engine_kwargs(
        tp_size=tp_size,
        ep_size=ep_size,
        deepep_parallel_profile=deepep_parallel_profile,
        deepep_mode=deepep_mode,
        enable_deepep_waterfill=enable_deepep_waterfill,
    )
    return {
        "patched": True,
        "sglang_python_path": str(VENDORED_SGLANG_PYTHON),
        "engine_kwargs": engine_kwargs,
    }


def _get_elastic_buffer(
    *,
    group: "dist.ProcessGroup",
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
):
    cache_key = (
        id(group),
        hidden,
        num_topk,
        num_experts,
        num_max_tokens_per_rank,
    )
    buffer = _ELASTIC_BUFFER_CACHE.get(cache_key)
    if buffer is not None:
        return buffer

    required_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=False,
    )
    buffer = deep_ep.ElasticBuffer(
        group,
        num_bytes=required_bytes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=False,
        explicitly_destroy=False,
    )
    _ELASTIC_BUFFER_CACHE[cache_key] = buffer
    return buffer


def run_deepep_v2_probe(
    *,
    num_tokens: int = 128,
    hidden: int = 4096,
    num_topk: int = 2,
    num_experts: int = 8,
    num_max_tokens_per_rank: Optional[int] = None,
) -> Dict[str, Any]:
    _require_torch_and_deepep()
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before running the DeepEP V2 probe.")

    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        raise RuntimeError("DeepEP V2 probe requires CUDA.")

    num_max_tokens_per_rank = num_max_tokens_per_rank or num_tokens
    torch.manual_seed(1234 + rank)

    hidden_states = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    router_logits = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(router_logits, num_topk, dim=-1)
    topk_ids = topk_ids.to(deep_ep.topk_idx_t)

    buffer = _get_elastic_buffer(
        group=group,
        hidden=hidden,
        num_topk=num_topk,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
    )

    torch.cuda.synchronize(device)
    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
        hidden_states,
        topk_idx=topk_ids,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        async_with_compute_stream=False,
    )
    torch.cuda.synchronize(device)

    num_recv_tokens = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    combine_input = torch.zeros_like(recv_x, dtype=torch.bfloat16, device=device)
    if num_recv_tokens > 0:
        combine_input[:num_recv_tokens] = recv_x[:num_recv_tokens]

    combined_x, combined_topk_weights, _ = buffer.combine(
        combine_input,
        handle=handle,
        topk_weights=recv_topk_weights,
        async_with_compute_stream=False,
    )
    torch.cuda.synchronize(device)

    result = {
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "num_tokens": num_tokens,
        "hidden": hidden,
        "num_topk": num_topk,
        "num_experts": num_experts,
        "num_recv_tokens": num_recv_tokens,
        "recv_shape": list(recv_x.shape),
        "combined_shape": list(combined_x.shape),
        "recv_topk_idx_shape": list(recv_topk_idx.shape) if recv_topk_idx is not None else None,
        "combined_topk_weights_shape": (
            list(combined_topk_weights.shape) if combined_topk_weights is not None else None
        ),
        "combined_l1": float(combined_x.abs().sum().item()),
    }
    LOGGER.info("[DeepEP] ElasticBuffer probe finished: %s", result)
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description="DeepEP V2 patch/probe utility")
    parser.add_argument("--probe", action="store_true", help="Run a distributed DeepEP V2 ElasticBuffer probe")
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-topk", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--deepep-mode", type=str, default="normal")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.probe:
        _require_torch_and_deepep()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        try:
            result = run_deepep_v2_probe(
                num_tokens=args.num_tokens,
                hidden=args.hidden,
                num_topk=args.num_topk,
                num_experts=args.num_experts,
                num_max_tokens_per_rank=(
                    args.num_max_tokens_per_rank or args.num_tokens
                ),
            )
            print(json.dumps(result, ensure_ascii=False))
            dist.barrier()
        finally:
            dist.destroy_process_group()
        return 0

    result = patch_sglang_moe(tp_size=args.tp_size, deepep_mode=args.deepep_mode)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
