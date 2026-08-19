import os
import time
from typing import Any, Dict, List, Optional

from cgc_engine.agent.llm1_vllm_client import vllm_chat_completions


def _summarize(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"min": 0.0, "p50": 0.0, "max": 0.0}
    vs = sorted(float(x) for x in values)
    n = len(vs)
    p50 = vs[(n - 1) // 2]
    return {"min": float(vs[0]), "p50": float(p50), "max": float(vs[-1])}


class MindSpeedLLMBackend:
    name = "mindspeed"

    def run_context_bench(
        self,
        model_name: str,
        *,
        contexts: List[int],
        gen_tokens: int,
        warmup_runs: int,
        runs: int,
        enable_hooks: bool,
        enable_ortho_kda: bool,
        ortho_kda_base_dim: int,
        seed: int,
        gguf_path: Optional[str] = None,
        exec_mode: str = "native",
        base_url: str = "",
        api_key: Optional[str] = None,
        timeout_s: int = 120,
        source_hf: str = "",
        mcore_dir: str = "",
        precision: str = "fp8_mixed",
        exec_driver: str = "http",
        subprocess_cmd: str = "",
        subprocess_cwd: str = "",
        subprocess_env_source: str = "",
        subprocess_pythonpath: str = "",
        subprocess_extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if enable_hooks:
            return {"status": "SKIP", "reason": "MindSpeed backend does not support MLXOpsHook"}
        if enable_ortho_kda:
            return {"status": "SKIP", "reason": "MindSpeed backend does not support OrthoKDA hooks"}
        if str(exec_mode) == "inject":
            return {"status": "SKIP", "reason": "MindSpeed backend inject mode is not supported"}

        timeout_s = int(timeout_s)
        driver = str(exec_driver or "").strip().lower()
        if driver == "":
            driver = "http"

        def _make_prompt(ctx: int) -> str:
            approx_chars = int(ctx * 4)
            return ("hello " * max(1, approx_chars // 6)).strip()

        def _run_once_http(ctx: int) -> Dict[str, Any]:
            cloud_url = str(base_url or os.environ.get("MINDSPEED_BASE_URL") or "").strip()
            if cloud_url == "":
                return {"status": "FAIL", "error": "missing mindspeed base url (--mindspeed-base-url or env MINDSPEED_BASE_URL)"}

            key = api_key if api_key is not None else os.environ.get("MINDSPEED_API_KEY")
            prompt = _make_prompt(int(ctx))
            params_1 = {"max_tokens": 1, "temperature": 0.0}
            params_n = {"max_tokens": int(gen_tokens), "temperature": 0.0}

            t0 = time.perf_counter()
            r1 = vllm_chat_completions(
                base_url=cloud_url,
                model=str(model_name),
                messages=[{"role": "user", "content": prompt}],
                timeout_s=int(timeout_s),
                api_key=key,
                extra_body=params_1,
            )
            t_first = float(time.perf_counter() - t0)
            if not bool(r1.get("ok")):
                return {"status": "FAIL", "error": str(r1.get("error") or "request failed")}

            t1 = time.perf_counter()
            r2 = vllm_chat_completions(
                base_url=cloud_url,
                model=str(model_name),
                messages=[{"role": "user", "content": prompt}],
                timeout_s=int(timeout_s),
                api_key=key,
                extra_body=params_n,
            )
            t_total = float(time.perf_counter() - t1)
            if not bool(r2.get("ok")):
                return {"status": "FAIL", "error": str(r2.get("error") or "request failed")}

            decode_s = max(0.0, float(t_total - t_first))
            decode_tps = float(int(gen_tokens) / max(decode_s, 1e-9)) if int(gen_tokens) > 0 else 0.0
            return {
                "status": "PASS",
                "prefill_tps": 0.0,
                "decode_tps": float(decode_tps),
                "peak_memory_gb": 0.0,
                "elapsed_s": float(t_first + t_total),
                "note": {
                    "driver": "http",
                    "base_url": cloud_url,
                    "prompt_chars": int(len(prompt)),
                    "source_hf": str(source_hf).strip() or None,
                    "mcore_dir": str(mcore_dir).strip() or None,
                    "precision": str(precision).strip() or "fp8_mixed",
                },
            }

        def _run_once_subprocess(ctx: int) -> Dict[str, Any]:
            import subprocess
            import shlex

            raw_cmd = str(subprocess_cmd or "").strip()
            if raw_cmd == "":
                return {"status": "FAIL", "error": "missing mindspeed subprocess cmd (--mindspeed-subprocess-cmd)"}

            prompt = _make_prompt(int(ctx))
            subs = {
                "context": str(int(ctx)),
                "gen_tokens": str(int(gen_tokens)),
                "model": str(model_name),
                "mcore_dir": str(mcore_dir),
                "source_hf": str(source_hf),
                "precision": str(precision),
                "prompt": prompt.replace('"', '\\"'),
            }
            for k, v in subs.items():
                raw_cmd = raw_cmd.replace("{" + str(k) + "}", str(v))

            cwd = str(subprocess_cwd or "").strip()
            env_source = str(subprocess_env_source or "").strip()
            if env_source != "":
                cmd: List[str] = ["bash", "-lc", f'source "{env_source}" && {raw_cmd}']
            else:
                cmd = shlex.split(raw_cmd)

            child_env = os.environ.copy()
            extra = subprocess_extra_env or {}
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if v is None:
                        continue
                    child_env[str(k)] = str(v)
            pp = str(subprocess_pythonpath or "").strip()
            if pp != "":
                existing = str(child_env.get("PYTHONPATH") or "").strip()
                child_env["PYTHONPATH"] = f"{pp}:{existing}" if existing != "" else pp

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=cwd if cwd != "" else None,
                    capture_output=True,
                    text=True,
                    timeout=int(timeout_s),
                    env=child_env,
                )
            except Exception as e:
                return {"status": "FAIL", "error": f"subprocess failed: {e}"}
            elapsed_s = float(time.perf_counter() - t0)

            if int(proc.returncode) != 0:
                return {
                    "status": "FAIL",
                    "error": f"subprocess exit={int(proc.returncode)}",
                    "elapsed_s": float(elapsed_s),
                    "stdout_tail": str(proc.stdout or "")[-2000:],
                    "stderr_tail": str(proc.stderr or "")[-2000:],
                }

            decode_tps = float(int(gen_tokens) / max(elapsed_s, 1e-9)) if int(gen_tokens) > 0 else 0.0
            return {
                "status": "PASS",
                "prefill_tps": 0.0,
                "decode_tps": float(decode_tps),
                "peak_memory_gb": 0.0,
                "elapsed_s": float(elapsed_s),
                "note": {
                    "driver": "subprocess",
                    "cmd": str(subprocess_cmd),
                    "cwd": cwd if cwd != "" else None,
                    "env_source": env_source if env_source != "" else None,
                    "prompt_chars": int(len(prompt)),
                    "source_hf": str(source_hf).strip() or None,
                    "mcore_dir": str(mcore_dir).strip() or None,
                    "precision": str(precision).strip() or "fp8_mixed",
                },
            }

        def _run_once(ctx: int) -> Dict[str, Any]:
            if driver in ("subprocess", "local"):
                return _run_once_subprocess(int(ctx))
            return _run_once_http(int(ctx))

        per_ctx: List[Dict[str, Any]] = []
        for ctx in contexts:
            ctx0 = time.perf_counter()
            for _ in range(int(warmup_runs)):
                out = _run_once(int(ctx))
                if out.get("status") == "FAIL":
                    return {"status": "FAIL", "error": out.get("error", "request failed")}

            rows: List[Dict[str, Any]] = []
            for i in range(int(runs)):
                out = _run_once(int(ctx))
                if out.get("status") == "FAIL":
                    return {"status": "FAIL", "error": out.get("error", "request failed")}
                rows.append(
                    {
                        "run": int(i),
                        "elapsed_s": float(out["elapsed_s"]),
                        "prefill_tps": float(out["prefill_tps"]),
                        "decode_tps": float(out["decode_tps"]),
                        "peak_memory_gb": float(out["peak_memory_gb"]),
                    }
                )

            per_ctx.append(
                {
                    "status": "PASS",
                    "context": int(ctx),
                    "elapsed_s": float(time.perf_counter() - ctx0),
                    "prefill_tps": _summarize([float(r["prefill_tps"]) for r in rows]),
                    "decode_tps": _summarize([float(r["decode_tps"]) for r in rows]),
                    "peak_memory_gb": _summarize([float(r["peak_memory_gb"]) for r in rows]),
                    "runs": rows,
                    "note": "remote OpenAI-compatible endpoint; prefill_tps not measured (no token accounting)",
                }
            )

        return {"status": "PASS", "contexts": per_ctx, "inject": {"status": "SKIP"}}
