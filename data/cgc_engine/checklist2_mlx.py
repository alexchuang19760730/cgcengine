#!/usr/bin/env python3

import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Dict


def _run_case(name: str, fn, results: Dict[str, Any]) -> None:
    t0 = time.perf_counter()
    try:
        payload = fn()
        results["cases"][name] = {
            "status": "PASS",
            "elapsed_s": time.perf_counter() - t0,
            "payload": payload,
        }
    except RuntimeError as e:
        results["cases"][name] = {
            "status": "FAIL",
            "elapsed_s": time.perf_counter() - t0,
            "error": str(e),
        }
        results["ok"] = False
    except Exception:
        results["cases"][name] = {
            "status": "FAIL",
            "elapsed_s": time.perf_counter() - t0,
            "traceback": traceback.format_exc(),
        }
        results["ok"] = False


def _skip_case(name: str, reason: str, results: Dict[str, Any]) -> None:
    results["cases"][name] = {"status": "SKIP", "reason": reason}


def _mlx_smoke(device: str) -> Dict[str, Any]:
    import torch
    from cgc_engine.cgc.mlx_tune_integration import get_mlx_tune_info, mlx_lora_fwd, mlx_rope_fwd

    info = get_mlx_tune_info()
    if not bool(info.get("mlx_available", False)):
        raise RuntimeError("MLX is not available (import failed or unsupported platform)")

    dev = torch.device(device)
    dtype = torch.float16 if dev.type in ("mps", "cuda") else torch.float32
    x = torch.randn(2, 16, device=dev, dtype=dtype)
    w = torch.randn(16, 32, device=dev, dtype=dtype)
    a = torch.randn(16, 8, device=dev, dtype=dtype)
    b = torch.randn(8, 32, device=dev, dtype=dtype)
    y = mlx_lora_fwd(x, w, a, b, scale=1.0)

    cos = torch.randn(2, 16, device=dev, dtype=dtype)
    sin = torch.randn(2, 16, device=dev, dtype=dtype)
    z = mlx_rope_fwd(x, cos, sin)

    return {
        "mlx_available": bool(info.get("mlx_available", False)),
        "flashkda_available": bool(info.get("flashkda_available", False)),
        "device": str(dev),
        "dtype": str(dtype),
        "lora_out_shape": list(y.shape),
        "rope_out_shape": list(z.shape),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--report-path", default="/tmp/cgc_engine_checklist2_mlx_report.json")
    args = p.parse_args()

    results: Dict[str, Any] = {
        "ok": True,
        "env": {
            "hostname": os.uname().nodename,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "cases": {},
    }

    if platform.system() != "Darwin":
        _skip_case("mlx_compute_smoke", "MLX is macOS-only; NVIDIA/Linux is not applicable", results)
    else:
        _run_case("mlx_compute_smoke", lambda: _mlx_smoke(args.device), results)

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": results["ok"], "report_path": args.report_path}, ensure_ascii=False))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

