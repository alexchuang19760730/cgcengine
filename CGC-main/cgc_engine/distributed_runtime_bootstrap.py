from __future__ import annotations

import inspect
import os
from typing import Any, Mapping

import torch


def _env_snapshot(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    env = environ or os.environ
    return {
        "RANK": str(env.get("RANK") or "").strip(),
        "WORLD_SIZE": str(env.get("WORLD_SIZE") or "").strip(),
        "LOCAL_RANK": str(env.get("LOCAL_RANK") or "").strip(),
    }


def materialize_distributed_runtime_bootstrap(
    *,
    enable_nccl: bool,
    device: torch.device,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = _env_snapshot(environ)
    if not bool(enable_nccl):
        return {
            "schema_version": "distributed_runtime_bootstrap_v1",
            "status": "SKIP",
            "reason": "enable_nccl=false",
            "env": env,
            "backend": "",
            "bootstrap_plan": {
                "target_device_index": None,
                "device_id": "",
                "set_device_applied": False,
                "init_process_group_device_id_enabled": False,
            },
            "bound_device_index": None,
            "barrier_device_ids": [],
        }
    if device.type != "cuda":
        return {
            "schema_version": "distributed_runtime_bootstrap_v1",
            "status": "SKIP",
            "reason": f"device={device.type}",
            "env": env,
            "backend": "",
            "bootstrap_plan": {
                "target_device_index": None,
                "device_id": "",
                "set_device_applied": False,
                "init_process_group_device_id_enabled": False,
            },
            "bound_device_index": None,
            "barrier_device_ids": [],
        }

    world_size = int(env["WORLD_SIZE"]) if env["WORLD_SIZE"].isdigit() else 1
    rank = int(env["RANK"]) if env["RANK"].isdigit() else 0
    local_rank = int(env["LOCAL_RANK"]) if env["LOCAL_RANK"].isdigit() else None
    current_device_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    target_device_index = local_rank if local_rank is not None and local_rank >= 0 else current_device_index
    device_id = f"cuda:{target_device_index}" if target_device_index is not None else ""

    if world_size <= 1:
        return {
            "schema_version": "distributed_runtime_bootstrap_v1",
            "status": "SKIP",
            "reason": "WORLD_SIZE<=1",
            "env": env,
            "rank": rank,
            "world_size": world_size,
            "backend": "nccl",
            "bootstrap_plan": {
                "target_device_index": target_device_index,
                "device_id": device_id,
                "set_device_applied": False,
                "init_process_group_device_id_enabled": False,
            },
            "bound_device_index": current_device_index,
            "barrier_device_ids": [current_device_index] if current_device_index is not None else [],
        }

    return {
        "schema_version": "distributed_runtime_bootstrap_v1",
        "status": "READY",
        "reason": "",
        "env": env,
        "rank": rank,
        "world_size": world_size,
        "backend": "nccl",
        "bootstrap_plan": {
            "target_device_index": target_device_index,
            "device_id": device_id,
            "set_device_applied": False,
            "init_process_group_device_id_enabled": False,
        },
        "bound_device_index": current_device_index,
        "barrier_device_ids": [target_device_index] if target_device_index is not None else [],
    }


def initialize_distributed_runtime_bootstrap(bootstrap: dict[str, Any]) -> dict[str, Any]:
    report = dict(bootstrap)
    try:
        import torch.distributed as dist

        if not dist.is_available():
            report["status"] = "FAIL"
            report["error"] = "torch.distributed not available"
            return report

        if str(report.get("status") or "") != "READY":
            return report

        plan = report.get("bootstrap_plan") if isinstance(report.get("bootstrap_plan"), dict) else {}
        target_device_index = plan.get("target_device_index")
        current_device_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
        if isinstance(target_device_index, int) and target_device_index >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(target_device_index)
            current_device_index = int(torch.cuda.current_device())

        device_id_obj = torch.device(f"cuda:{current_device_index}") if current_device_index is not None else None
        supports_device_id = "device_id" in inspect.signature(dist.init_process_group).parameters

        if dist.is_initialized():
            report["status"] = "PASS"
            report["already_initialized"] = True
            report["rank"] = int(dist.get_rank())
            report["world_size"] = int(dist.get_world_size())
            report["backend"] = str(dist.get_backend())
        else:
            init_kwargs: dict[str, Any] = {"backend": str(report.get("backend") or "nccl")}
            if supports_device_id and device_id_obj is not None:
                init_kwargs["device_id"] = device_id_obj
            dist.init_process_group(**init_kwargs)
            report["status"] = "PASS"
            report["already_initialized"] = False
            report["rank"] = int(dist.get_rank())
            report["world_size"] = int(dist.get_world_size())
            report["backend"] = str(dist.get_backend())

        report["bound_device_index"] = current_device_index
        report["barrier_device_ids"] = [current_device_index] if current_device_index is not None else []
        report["bootstrap_plan"] = {
            "target_device_index": current_device_index,
            "device_id": str(device_id_obj) if device_id_obj is not None else "",
            "set_device_applied": bool(current_device_index is not None),
            "init_process_group_device_id_enabled": bool(supports_device_id and device_id_obj is not None),
        }
        return report
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        return report


def distributed_runtime_barrier(bootstrap: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return {"status": "SKIP", "reason": "distributed_not_initialized"}

        barrier_device_ids = bootstrap.get("barrier_device_ids") if isinstance(bootstrap.get("barrier_device_ids"), list) else []
        backend = str(dist.get_backend() or "")
        if backend == "nccl" and barrier_device_ids:
            dist.barrier(device_ids=[int(barrier_device_ids[0])])
        else:
            dist.barrier()
        return {
            "status": "PASS",
            "backend": backend,
            "barrier_device_ids": [int(barrier_device_ids[0])] if barrier_device_ids else [],
        }
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}
