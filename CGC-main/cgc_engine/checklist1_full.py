#!/usr/bin/env python3

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve_device(device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _run_case(
    *,
    name: str,
    fn,
    results: Dict[str, Any],
) -> None:
    t0 = time.perf_counter()
    try:
        payload = fn()
        results["cases"][name] = {
            "ok": True,
            "elapsed_s": time.perf_counter() - t0,
            "payload": payload,
        }
    except Exception:
        results["cases"][name] = {
            "ok": False,
            "elapsed_s": time.perf_counter() - t0,
            "traceback": traceback.format_exc(),
        }
        results["ok"] = False


def _round_trip_expert_weights(
    *,
    seed: int,
    device: "torch.device",
    expert_dir: str,
    expert_dim: int,
    intermediate_dim: int,
) -> Dict[str, Any]:
    import torch
    from cgc_engine.io_unified.unified_io_controller import UnifiedIOController

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    expert_dir_p = Path(expert_dir)
    expert_dir_p.mkdir(parents=True, exist_ok=True)

    base_path = str(expert_dir_p / f"rt_expert_{seed}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    saved = {
        "w1": torch.randn((intermediate_dim, expert_dim), dtype=dtype, device="cpu"),
        "w3": torch.randn((intermediate_dim, expert_dim), dtype=dtype, device="cpu"),
        "w2": torch.randn((expert_dim, intermediate_dim), dtype=dtype, device="cpu"),
    }

    io1 = UnifiedIOController.get_instance()
    ok = bool(io1.save_expert_mlp(expert_id=0, base_path=base_path, weights=saved))
    if not ok:
        raise RuntimeError("save_expert_mlp failed")

    UnifiedIOController.reset_instance()

    io2 = UnifiedIOController.get_instance()
    loaded = io2.load_expert_mlp(
        expert_id=0,
        base_path=base_path,
        expert_dim=expert_dim,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
    )

    mismatches: List[str] = []
    for k in ("w1", "w3", "w2"):
        a = saved[k].detach().to("cpu")
        b = loaded[k].detach().to("cpu")
        if a.shape != b.shape or a.dtype != b.dtype:
            mismatches.append(f"{k}: shape/dtype")
            continue
        if not torch.allclose(a, b, atol=0.0, rtol=0.0):
            mismatches.append(f"{k}: values")

    if mismatches:
        raise RuntimeError("round-trip mismatch: " + ", ".join(mismatches))

    return {
        "seed": seed,
        "dtype": str(dtype),
        "base_path": base_path,
    }


def _topk_determinism(
    *,
    seed: int,
    device: "torch.device",
    batch_size: int,
    seq_len: int,
    expert_dim: int,
    top_k: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    import torch
    from cgc_engine.pipeline import HarnessAgentPipeline

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x = torch.randn(batch_size, seq_len, expert_dim, dtype=dtype, device=device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    p1 = HarnessAgentPipeline(config, device=device).predictor
    e1 = p1.predict(x, top_k=top_k).detach().to("cpu")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    p2 = HarnessAgentPipeline(config, device=device).predictor
    e2 = p2.predict(x, top_k=top_k).detach().to("cpu")

    if not torch.equal(e1, e2):
        raise RuntimeError("top_k gating is not deterministic under fixed seed")

    return {
        "seed": seed,
        "dtype": str(dtype),
        "expert_ids": e1.flatten().tolist(),
    }


def _pipeline_functional(
    *,
    seed: int,
    device: "torch.device",
    batch_size: int,
    seq_len: int,
    expert_dim: int,
    intermediate_dim: int,
    top_k: int,
    num_experts: int,
    max_cached_experts: int,
    expert_dir: str,
) -> Dict[str, Any]:
    import torch
    from cgc_engine.pipeline import HarnessAgentPipeline

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    config = {
        "device": str(device),
        "num_experts": int(num_experts),
        "expert_dim": int(expert_dim),
        "intermediate_dim": int(intermediate_dim),
        "top_k": int(top_k),
        "max_cached_experts": int(max_cached_experts),
        "expert_dir": str(expert_dir),
    }

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = torch.randn(batch_size, seq_len, expert_dim, dtype=dtype, device=device)
    out = HarnessAgentPipeline(config, device=device).run_pipeline(x)
    y = out["result"]
    fb = out["feedback"]

    if tuple(y.shape) != (batch_size, seq_len, expert_dim):
        raise RuntimeError(f"unexpected output shape: {tuple(y.shape)}")

    return {
        "seed": seed,
        "dtype": str(dtype),
        "input_shape": [batch_size, seq_len, expert_dim],
        "output_shape": list(y.shape),
        "stats": fb.get("stats", {}),
        "cache_size": fb.get("cache_size"),
        "hot_experts": fb.get("hot_experts"),
    }


def _moe_entrypoint_functional(
    *,
    seed: int,
    device: "torch.device",
    expert_dir: str,
    num_experts: int,
    expert_dim: int,
    intermediate_dim: int,
    batch_size: int,
    seq_len: int,
    top_k: int,
) -> Dict[str, Any]:
    from cgc_engine.moe_entrypoint import run_moe_entrypoint

    res = run_moe_entrypoint(
        device=device,
        expert_dir=expert_dir,
        num_experts=int(num_experts),
        expert_dim=int(expert_dim),
        intermediate_dim=int(intermediate_dim),
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        top_k=int(top_k),
        seed=int(seed),
    )

    if tuple(res["output_shape"]) != (batch_size, seq_len, expert_dim):
        raise RuntimeError(f"unexpected moe output shape: {res['output_shape']}")

    return {
        "seed": seed,
        "device": res["device"],
        "dtype": res["dtype"],
        "input_shape": res["input_shape"],
        "output_shape": res["output_shape"],
        "expert_ids": res["expert_ids"],
    }


def _env_info() -> Dict[str, Any]:
    import torch

    info: Dict[str, Any] = {
        "hostname": os.uname().nodename,
        "torch": None,
        "cuda_available": False,
    }
    try:
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_0"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_error"] = traceback.format_exc()
    return info


def _cache_hitrate_stress(
    *,
    seed: int,
    device: "torch.device",
    expert_dir: str,
    expert_dim: int,
    intermediate_dim: int,
    num_experts: int,
    rounds: int,
    hitrate_threshold: float,
) -> Dict[str, Any]:
    import torch
    from cgc_engine.io_unified.unified_io_controller import UnifiedIOController

    def _snap(io) -> Dict[str, int]:
        s = io.get_stats()
        return {
            "reads": int(s.reads),
            "writes": int(s.writes),
            "hits": int(s.hits),
            "misses": int(s.misses),
            "bytes_read": int(s.bytes_read),
            "bytes_written": int(s.bytes_written),
        }

    def _delta(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        return {k: int(b[k] - a[k]) for k in a.keys()}

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    expert_dir_p = Path(expert_dir)
    expert_dir_p.mkdir(parents=True, exist_ok=True)

    io0 = UnifiedIOController.get_instance()
    for eid in range(int(num_experts)):
        torch.manual_seed(int(seed) * 10000 + int(eid))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed) * 10000 + int(eid))
        base_path = str(expert_dir_p / f"expert_{eid}.pt")
        weights = {
            "w1": torch.randn((intermediate_dim, expert_dim), dtype=dtype, device="cpu"),
            "w3": torch.randn((intermediate_dim, expert_dim), dtype=dtype, device="cpu"),
            "w2": torch.randn((expert_dim, intermediate_dim), dtype=dtype, device="cpu"),
        }
        ok = bool(io0.save_expert_mlp(expert_id=int(eid), base_path=base_path, weights=weights))
        if not ok:
            raise RuntimeError("save_expert_mlp failed in hitrate precondition")

    UnifiedIOController.reset_instance()
    io = UnifiedIOController.get_instance()

    per_round: List[Dict[str, Any]] = []
    for r in range(int(rounds)):
        st0 = _snap(io)
        t0 = time.perf_counter()
        for eid in range(int(num_experts)):
            base_path = str(expert_dir_p / f"expert_{eid}.pt")
            _ = io.load_expert_mlp(
                expert_id=int(eid),
                base_path=base_path,
                expert_dim=int(expert_dim),
                intermediate_dim=int(intermediate_dim),
                dtype=dtype,
            )
        elapsed = time.perf_counter() - t0
        st1 = _snap(io)
        d = _delta(st0, st1)
        denom = d["hits"] + d["misses"]
        hit_rate = float(d["hits"]) / float(denom) if denom > 0 else 0.0
        per_round.append(
            {
                "round": r + 1,
                "elapsed_s": elapsed,
                "hits": d["hits"],
                "misses": d["misses"],
                "hit_rate": hit_rate,
            }
        )

    for item in per_round[1:]:
        if float(item["hit_rate"]) < float(hitrate_threshold):
            raise RuntimeError(f"hit-rate below threshold: {item['hit_rate']:.4f} < {hitrate_threshold:.4f}")

    return {
        "seed": seed,
        "dtype": str(dtype),
        "expert_dir": str(expert_dir),
        "num_experts": int(num_experts),
        "rounds": int(rounds),
        "hitrate_threshold": float(hitrate_threshold),
        "per_round": per_round,
    }


def _eviction_stress(
    *,
    seed: int,
    max_cached_experts: int,
    num_inserts: int,
) -> Dict[str, Any]:
    import torch
    from cgc_engine.storage_layer.cache_manager import ExpertCacheManager

    cache = ExpertCacheManager(max_size=int(max_cached_experts), enable_kda=False)
    evictions = 0
    inserted: List[int] = []

    for eid in range(int(num_inserts)):
        inserted.append(int(eid))
        pre_full = len(cache) >= int(max_cached_experts) and int(eid) not in cache.cache
        torch.manual_seed(int(seed) * 10000 + int(eid))
        w = torch.randn((4, 4), dtype=torch.float32, device="cpu")
        cache.set(int(eid), {"w1": w, "w3": w, "w2": w})
        if pre_full:
            evictions += 1
        if len(cache) > int(max_cached_experts):
            raise RuntimeError("cache size exceeds max after insertion")

    if evictions < 1:
        raise RuntimeError("eviction did not happen")

    return {
        "seed": seed,
        "max_cached_experts": int(max_cached_experts),
        "num_inserts": int(num_inserts),
        "evictions": int(evictions),
        "final_cache_size": int(len(cache)),
        "final_keys": [int(x) for x in cache.keys()],
        "inserted": inserted,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--expert-dir", default="/tmp/cgc_engine_experts_checklist1_full")
    parser.add_argument("--report-path", default="/tmp/cgc_engine_checklist1_full_report.json")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--hitrate-threshold", type=float, default=0.30)
    parser.add_argument("--hitrate-rounds", type=int, default=3)
    parser.add_argument("--hitrate-num-experts", type=int, default=8)
    parser.add_argument("--eviction-max-size", type=int, default=2)
    parser.add_argument("--eviction-num-inserts", type=int, default=6)
    args = parser.parse_args()

    import torch

    device = _resolve_device(args.device)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    results: Dict[str, Any] = {
        "ok": True,
        "env": _env_info(),
        "device": str(device),
        "expert_dir": str(args.expert_dir),
        "cases": {},
    }

    Path(args.expert_dir).mkdir(parents=True, exist_ok=True)
    expert_root = str(args.expert_dir)

    for seed in seeds:
        _run_case(
            name=f"round_trip_weights_seed_{seed}",
            fn=lambda seed=seed: _round_trip_expert_weights(
                seed=seed,
                device=device,
                expert_dir=str(Path(expert_root) / "roundtrip"),
                expert_dim=32,
                intermediate_dim=64,
            ),
            results=results,
        )

    for seed in seeds:
        _run_case(
            name=f"topk_determinism_seed_{seed}",
            fn=lambda seed=seed: _topk_determinism(
                seed=seed,
                device=device,
                batch_size=2,
                seq_len=8,
                expert_dim=32,
                top_k=2,
                config={
                    "device": str(device),
                    "num_experts": 16,
                    "expert_dim": 32,
                    "intermediate_dim": 64,
                    "top_k": 2,
                    "max_cached_experts": 2,
                    "expert_dir": args.expert_dir,
                },
            ),
            results=results,
        )

    for seed in seeds:
        _run_case(
            name=f"pipeline_small_seed_{seed}",
            fn=lambda seed=seed: _pipeline_functional(
                seed=seed,
                device=device,
                batch_size=2,
                seq_len=8,
                expert_dim=32,
                intermediate_dim=64,
                top_k=2,
                num_experts=16,
                max_cached_experts=2,
                expert_dir=str(Path(expert_root) / "pipeline_small"),
            ),
            results=results,
        )

    for seed in seeds:
        _run_case(
            name=f"pipeline_medium_seed_{seed}",
            fn=lambda seed=seed: _pipeline_functional(
                seed=seed,
                device=device,
                batch_size=4,
                seq_len=64,
                expert_dim=256,
                intermediate_dim=512,
                top_k=2,
                num_experts=32,
                max_cached_experts=4,
                expert_dir=str(Path(expert_root) / "pipeline_medium"),
            ),
            results=results,
        )

    for seed in seeds:
        _run_case(
            name=f"moe_entrypoint_seed_{seed}",
            fn=lambda seed=seed: _moe_entrypoint_functional(
                seed=seed,
                device=device,
                expert_dir=str(Path(expert_root) / "moe_entrypoint"),
                num_experts=16,
                expert_dim=32,
                intermediate_dim=64,
                batch_size=2,
                seq_len=8,
                top_k=2,
            ),
            results=results,
        )

    for seed in seeds:
        _run_case(
            name=f"cache_hitrate_seed_{seed}",
            fn=lambda seed=seed: _cache_hitrate_stress(
                seed=seed,
                device=device,
                expert_dir=str(Path(expert_root) / "cache_hitrate"),
                expert_dim=32,
                intermediate_dim=64,
                num_experts=int(args.hitrate_num_experts),
                rounds=int(args.hitrate_rounds),
                hitrate_threshold=float(args.hitrate_threshold),
            ),
            results=results,
        )

    for seed in seeds:
        _run_case(
            name=f"eviction_stress_seed_{seed}",
            fn=lambda seed=seed: _eviction_stress(
                seed=seed,
                max_cached_experts=int(args.eviction_max_size),
                num_inserts=int(args.eviction_num_inserts),
            ),
            results=results,
        )

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": results["ok"], "report_path": args.report_path}, ensure_ascii=False))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
