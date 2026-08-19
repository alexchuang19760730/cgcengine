import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(n: int, d: int) -> float:
    return float(n) / float(max(1, d))


def _snapshot_files(base: Path) -> Tuple[int, int]:
    if not base.exists():
        return 0, 0
    files = 0
    size = 0
    for fp in base.rglob("*"):
        if fp.is_file():
            files += 1
            try:
                size += int(fp.stat().st_size)
            except Exception:
                pass
    return int(files), int(size)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_torch_inductor_cache_dir() -> Optional[Path]:
    env = str(os.environ.get("TORCHINDUCTOR_CACHE_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    cands = [
        Path.home() / ".cache" / "torch" / "inductor",
        Path.home() / "Library" / "Caches" / "torch" / "inductor",
    ]
    for p in cands:
        try:
            if p.exists():
                return p.resolve()
        except Exception:
            continue
    return None


def _torch_dtype(dtype: str):
    import torch

    dt = str(dtype).strip().lower()
    if dt in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dt in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def generate_cloud_training_psi0(
    *,
    out_dir: Path,
    seq_lens: List[int],
    train_steps: int,
    ckpt_steps: int,
    num_layers: int,
    batch_size: int,
    device: str,
    dtype: str,
    tiny: bool,
) -> Dict[str, Any]:
    m73_dir = (out_dir / "m73_physical").resolve()

    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        data = {"status": "FAIL", "reason": f"missing_dependency:torch:{repr(e)}", "compile_success_rate": 0.0, "cache_hit_rate": 0.0}
        _write_json(m73_dir / "cloud_training_psi0.json", data)
        return data

    from cgc_engine.pipeline import MegatrainEightStepPipeline, MegatrainPipelineConfig

    tdtype = _torch_dtype(dtype)

    dev = str(device).strip().lower()
    if dev == "":
        dev = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if dev == "cuda" and not bool(torch.cuda.is_available()):
        data = {"status": "FAIL", "reason": "cuda_not_available", "compile_success_rate": 0.0, "cache_hit_rate": 0.0}
        _write_json(m73_dir / "cloud_training_psi0.json", data)
        return data

    base_dir = (m73_dir / "cloud_training_psi0_runs").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, Any]] = []
    compile_ok = 0
    cache_hit = 0

    for sl in seq_lens:
        seq_len = int(sl)
        export_dir = (base_dir / f"seq_{seq_len}").resolve()
        export_dir.mkdir(parents=True, exist_ok=True)

        cache_root = export_dir / "step5_generate" / "torchinductor_cache"
        before1 = _snapshot_files(cache_root)
        cfg1 = MegatrainPipelineConfig(
            task_type="train",
            backend=dev,
            environment="cloud_single",
            task_domain="psi0_system",
            model_name="psi0_system",
            dtype=tdtype,
            tiny=bool(tiny),
            num_layers=int(num_layers),
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            train_steps=max(1, int(train_steps)),
            export_dir=str(export_dir),
            report_filename="m73_cloud_training_report.json",
        )
        pipe1 = MegatrainEightStepPipeline(cfg1)
        rep1 = pipe1.run()
        after1 = _snapshot_files(cache_root)
        r1_path = export_dir / "pass1_report.json"
        _write_json(r1_path, rep1)

        before2 = _snapshot_files(cache_root)
        cfg2 = MegatrainPipelineConfig(
            task_type="train",
            backend=dev,
            environment="cloud_single",
            task_domain="psi0_system",
            model_name="psi0_system",
            dtype=tdtype,
            tiny=bool(tiny),
            num_layers=int(num_layers),
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            train_steps=max(1, int(train_steps)),
            export_dir=str(export_dir),
            report_filename="m73_cloud_training_report.json",
        )
        pipe2 = MegatrainEightStepPipeline(cfg2)
        rep2 = pipe2.run()
        after2 = _snapshot_files(cache_root)
        r2_path = export_dir / "pass2_report.json"
        _write_json(r2_path, rep2)

        compile1 = ((rep1.get("step5_generate") or {}).get("torch_compile") or {}) if isinstance(rep1.get("step5_generate"), dict) else {}
        compile2 = ((rep2.get("step5_generate") or {}).get("torch_compile") or {}) if isinstance(rep2.get("step5_generate"), dict) else {}

        ok1 = str(compile1.get("status") or "") == "PASS"
        ok2 = str(compile2.get("status") or "") == "PASS"
        if ok1 and ok2:
            compile_ok += 1

        a_new = int(after1[0]) - int(before1[0])
        b_new = int(after2[0]) - int(before2[0])
        hit = bool(ok1 and ok2 and (b_new <= max(0, a_new // 10)))
        if hit:
            cache_hit += 1

        variants.append(
            {
                "seq_len": int(seq_len),
                "pass1": {
                    "status": "PASS" if ok1 else "FAIL",
                    "report_path": str(r1_path),
                    "compile": compile1,
                    "cache": {"root": str(cache_root), "files_before": before1[0], "files_after": after1[0], "bytes_before": before1[1], "bytes_after": after1[1]},
                },
                "pass2": {
                    "status": "PASS" if ok2 else "FAIL",
                    "report_path": str(r2_path),
                    "compile": compile2,
                    "cache": {"root": str(cache_root), "files_before": before2[0], "files_after": after2[0], "bytes_before": before2[1], "bytes_after": after2[1]},
                },
                "cache_hit": hit,
            }
        )

    compile_success_rate = _pct(int(compile_ok), len(seq_lens))
    cache_hit_rate = _pct(int(cache_hit), len(seq_lens))

    ckpt: dict[str, Any] = {"status": "FAIL", "reason": "not_generated"}
    try:
        ckpt_dir = (m73_dir / "megatrain_ckpt").resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = (ckpt_dir / "ckpt_epoch_0.pth").resolve()

        train_seq_len = int(max(seq_lens) if seq_lens else 256)
        export_dir = (base_dir / f"seq_{train_seq_len}").resolve()
        export_dir.mkdir(parents=True, exist_ok=True)

        cfg = MegatrainPipelineConfig(
            task_type="train",
            backend=dev,
            environment="cloud_single",
            task_domain="psi0_system",
            model_name="psi0_system",
            dtype=tdtype,
            tiny=bool(tiny),
            num_layers=int(num_layers),
            batch_size=int(batch_size),
            seq_len=int(train_seq_len),
            train_steps=max(1, int(train_steps)),
            export_dir=str(export_dir),
            report_filename="m73_cloud_training_report.json",
        )
        pipe = MegatrainEightStepPipeline(cfg)
        rep = pipe.run()
        model = pipe.model
        if model is None:
            raise RuntimeError("pipeline_model_is_none")
        device_obj = next(model.parameters()).device

        vocab_size = int(getattr(getattr(model, "embed", None), "num_embeddings", 0) or 0)
        if vocab_size <= 0:
            vocab_size = int(cfg.vocab_size or 4096)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses: list[float] = []
        for _ in range(max(1, int(ckpt_steps))):
            input_ids = torch.randint(0, vocab_size, (int(batch_size), int(train_seq_len)), device=device_obj, dtype=torch.long)
            labels = torch.randint(0, vocab_size, (int(batch_size), int(train_seq_len)), device=device_obj, dtype=torch.long)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size).float(), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))

        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload = {
            "model_state": state,
            "config": {
                "task_domain": str(cfg.task_domain),
                "task_type": str(cfg.task_type),
                "backend": str(cfg.backend),
                "model_name": str(cfg.model_name),
                "dtype": str(cfg.dtype),
                "tiny": bool(cfg.tiny),
                "num_layers": int(cfg.num_layers),
                "batch_size": int(cfg.batch_size),
                "seq_len": int(cfg.seq_len),
                "train_steps": int(cfg.train_steps),
                "ckpt_steps": int(ckpt_steps),
            },
            "train_losses": losses,
            "pipeline_report_path": str((export_dir / "ckpt_pipeline_report.json").resolve()),
        }
        _write_json(Path(payload["pipeline_report_path"]), rep)
        torch.save(payload, str(ckpt_path))

        ckpt = {
            "status": "PASS",
            "ckpt_path": str(ckpt_path),
            "sha256": _sha256_file(ckpt_path),
            "size_bytes": int(ckpt_path.stat().st_size),
            "train_losses": losses,
        }
    except Exception as e:
        ckpt = {"status": "FAIL", "reason": repr(e)}

    status = "PASS" if (compile_success_rate >= 1.0 and cache_hit_rate >= 0.6667 and str(ckpt.get("status")) == "PASS") else "FAIL"

    data = {
        "status": status,
        "device": str(dev),
        "dtype": str(dtype),
        "tiny": bool(tiny),
        "seq_lens": [int(x) for x in seq_lens],
        "compile_success_rate": float(compile_success_rate),
        "cache_hit_rate": float(cache_hit_rate),
        "variants": variants,
        "checkpoint": ckpt,
    }
    _write_json(m73_dir / "cloud_training_psi0.json", data)
    return data


def _parse_pipeline_summary(stdout: str) -> Optional[Dict[str, Any]]:
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    for ln in reversed(lines[-50:]):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                d = json.loads(ln)
                if isinstance(d, dict) and "report_path" in d:
                    return d
            except Exception:
                continue
    return None


def _pipeline_latency_ms(report: Dict[str, Any]) -> Optional[float]:
    for root_key in ("optimized", "native"):
        root = report.get(root_key)
        if not isinstance(root, dict):
            continue
        
        # Format 1: measure -> stats -> p99
        measure = root.get("measure")
        if isinstance(measure, dict):
            stats = measure.get("stats")
            if isinstance(stats, dict):
                for k in ("p99_time_ms", "p99", "p99_ms"):
                    if k in stats:
                        try:
                            return float(stats.get(k))
                        except Exception:
                            pass
                            
        # Format 2: contexts -> decode_tps
        contexts = root.get("contexts")
        if isinstance(contexts, list) and len(contexts) > 0:
            try:
                dtps = contexts[-1].get("decode_tps")
                if isinstance(dtps, dict):
                    tps = float(dtps.get("p50") or dtps.get("mean") or 0.0)
                    if tps > 0:
                        return 1000.0 / tps  # latency per token in ms
                elif isinstance(dtps, (int, float)):
                    if float(dtps) > 0:
                        return 1000.0 / float(dtps)
            except Exception:
                pass

    try:
        total = float(report.get("total_time_s") or 0.0)
        runs = int(report.get("runs") or 1)
        if total > 0 and runs > 0:
            return float((total / float(runs)) * 1000.0)
    except Exception:
        return None
    return None


def generate_edge_inference_bridge(
    *,
    out_dir: Path,
    backends: List[str],
    model: str,
    gguf_path: Optional[str],
    runs: int,
    warmup_runs: int,
    gen_tokens: int,
    contexts: str,
) -> Dict[str, Any]:
    m73_dir = (out_dir / "m73_physical").resolve()
    existing: Dict[str, Any] = {}
    if (m73_dir / "edge_inference_bridge.json").exists():
        try:
            existing = _read_json(m73_dir / "edge_inference_bridge.json")
        except Exception:
            existing = {}

    per_backend: Dict[str, Any] = {}
    if isinstance(existing.get("backends"), dict):
        per_backend.update(existing["backends"])

    latencies: list[float] = []
    required = [str(b).strip() for b in backends if str(b).strip() != ""]

    for b in backends:
        b = str(b).strip()
        if b == "":
            continue
        eff_gguf = gguf_path
        if b in {"llama.cpp", "llama_cpp", "llama"} and (eff_gguf is None or str(eff_gguf).strip() == ""):
            eff_gguf = str(model)
        cmd = [
            sys.executable,
            str((Path(__file__).resolve().parents[1] / "agent" / "cli.py").resolve()),
            "pipeline",
            "--mode",
            "llm",
            "--backend",
            b,
            "--model",
            str(model),
            "--contexts",
            str(contexts),
            "--runs",
            str(int(runs)),
            "--warmup-runs",
            str(int(warmup_runs)),
            "--gen-tokens",
            str(int(gen_tokens)),
            "--output-dir",
            str((out_dir / "m73_physical" / f"bridge_{b.replace('/', '_').replace('.', '_')}").resolve()),
        ]
        if eff_gguf is not None and str(eff_gguf).strip() != "":
            cmd.extend(["--gguf-path", str(eff_gguf)])
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        summ = _parse_pipeline_summary(p.stdout or "")
        if summ is None:
            per_backend[b] = {"status": "FAIL", "reason": "missing_pipeline_summary", "returncode": int(p.returncode)}
            continue
        rp = Path(str(summ.get("report_path") or "")).expanduser().resolve()
        if not rp.exists():
            per_backend[b] = {"status": "FAIL", "reason": "missing_report_path", "returncode": int(p.returncode), "report_path": str(rp)}
            continue
        report = _read_json(rp)
        latency = _pipeline_latency_ms(report)
        ok = bool(report.get("ok") is True)
        if latency is None:
            ok = False
        per_backend[b] = {"status": "PASS" if ok else "FAIL", "report_path": str(rp), "latency_ms": float(latency) if latency is not None else None}
        if latency is not None:
            latencies.append(float(latency))

    ok_all = True
    required_latencies: list[float] = []
    for b in required:
        st = per_backend.get(b) if isinstance(per_backend.get(b), dict) else {}
        if str((st or {}).get("status") or "") != "PASS":
            ok_all = False
        lm = (st or {}).get("latency_ms")
        if isinstance(lm, (int, float)):
            required_latencies.append(float(lm))

    edge_latency_ms = float(max(required_latencies)) if required_latencies else 999.0
    bridge_export_success = 1.0 if ok_all else 0.0
    status = "PASS" if (ok_all and edge_latency_ms <= 20.0) else "FAIL"

    data = {
        "status": status,
        "bridge_export_success": float(bridge_export_success),
        "edge_latency_ms": float(edge_latency_ms),
        "backends": per_backend,
        "model": str(model),
        "gguf_path": str(gguf_path) if gguf_path is not None else None,
    }
    _write_json(m73_dir / "edge_inference_bridge.json", data)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument("--mode", default="all", choices=["all", "cloud", "edge"], type=str)
    ap.add_argument("--psi0-seq-lens", default="256,512,1024", type=str)
    ap.add_argument("--psi0-device", default="", type=str)
    ap.add_argument("--psi0-dtype", default="bf16", type=str)
    ap.add_argument("--psi0-full", action="store_true", default=False)
    ap.add_argument("--psi0-num-layers", default=2, type=int)
    ap.add_argument("--psi0-batch-size", default=1, type=int)
    ap.add_argument("--psi0-train-steps", default=1, type=int)
    ap.add_argument("--psi0-ckpt-steps", default=4, type=int)
    ap.add_argument("--bridge-backends", default="llama.cpp,mlx,vllm", type=str)
    ap.add_argument("--bridge-model", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    ap.add_argument("--bridge-gguf-path", default="", type=str)
    ap.add_argument("--bridge-runs", default=3, type=int)
    ap.add_argument("--bridge-warmup-runs", default=1, type=int)
    ap.add_argument("--bridge-gen-tokens", default=32, type=int)
    ap.add_argument("--bridge-contexts", default="128", type=str)
    args = ap.parse_args()

    out_dir = Path(str(args.output_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = []
    for s in str(args.psi0_seq_lens).split(","):
        s = s.strip()
        if s:
            try:
                seq_lens.append(int(s))
            except Exception:
                pass
    if not seq_lens:
        seq_lens = [256, 512, 1024]

    dev = str(args.psi0_device).strip()
    if dev == "":
        dev = "cuda" if platform.system().lower() != "darwin" else "cpu"

    mode = str(args.mode).strip().lower()
    if mode in {"all", "cloud"}:
        generate_cloud_training_psi0(
            out_dir=out_dir,
            seq_lens=seq_lens,
            train_steps=int(args.psi0_train_steps),
            ckpt_steps=int(args.psi0_ckpt_steps),
            num_layers=int(args.psi0_num_layers),
            batch_size=int(args.psi0_batch_size),
            device=dev,
            dtype=str(args.psi0_dtype),
            tiny=not bool(args.psi0_full),
        )

    backends = [b.strip() for b in str(args.bridge_backends).split(",") if b.strip()]
    gguf = str(args.bridge_gguf_path).strip()
    if mode in {"all", "edge"}:
        generate_edge_inference_bridge(
            out_dir=out_dir,
            backends=backends,
            model=str(args.bridge_model),
            gguf_path=gguf if gguf != "" else None,
            runs=int(args.bridge_runs),
            warmup_runs=int(args.bridge_warmup_runs),
            gen_tokens=int(args.bridge_gen_tokens),
            contexts=str(args.bridge_contexts),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
