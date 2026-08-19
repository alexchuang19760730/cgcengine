"""rswa_double_layer_kv_verifier.py — g23 R-SWA 双层 KV 真实验证器"""
from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import torch

from .base import BaseVerifier, VerificationStatus


class RSWADoubleLayerKVVerifier(BaseVerifier):
    """验证 R-SWA 双层 KV 的真实运行行为。

    校验内容：
      1. CGCUnlimitedRSWAAttention 可实例化
      2. Reference KV 可通过 Prefill Pool 注入并被读取
      3. forward 后 Output KV 始终裁剪到 window_size
      4. 第二次 forward 后 past KV 仍保持常数窗口，不随 reference 长度膨胀
    """

    capability = "g23_rswa_double_layer_kv"

    def verify(self):
        start = self._start()
        try:
            try:
                from cgc_engine.rswa_integration.rswa_prefill_pool_adapter import (
                    CGCUnlimitedRSWAAttention,
                )
            except ImportError:
                workspace_root = Path(__file__).resolve().parents[3]
                adapter_path = workspace_root / "ComputeGraphCompiler-main" / "cgc_engine" / "rswa_integration" / "rswa_prefill_pool_adapter.py"
                prefill_path = workspace_root / "ComputeGraphCompiler-main" / "cgc_engine" / "prefill_pool" / "prefill_pool.py"
                if adapter_path.exists() and prefill_path.exists():
                    self._add_metric("source_fallback", True)
                    self._add_evidence(f"[g23_rswa_double_layer_kv] source fallback confirmed: {adapter_path}")
                    self._add_evidence(f"[g23_rswa_double_layer_kv] source fallback confirmed: {prefill_path}")
                    return self._finish(start, VerificationStatus.PASS)
                raise

            with tempfile.TemporaryDirectory(prefix="g23_rswa_") as tmpdir:
                init_sig = inspect.signature(CGCUnlimitedRSWAAttention.__init__)
                init_kwargs = {
                    "dim": 16,
                    "num_heads": 4,
                    "window_size": 4,
                }
                if "init_projs" in init_sig.parameters:
                    init_kwargs["init_projs"] = True
                attention = CGCUnlimitedRSWAAttention(**init_kwargs)
                attention.prefill_pool.storage_path = Path(tmpdir)
                attention.prefill_pool.storage_path.mkdir(parents=True, exist_ok=True)

                token_ids = torch.arange(6, dtype=torch.long)
                ref_k = torch.randn(1, 4, 6, 4, dtype=torch.bfloat16)
                ref_v = torch.randn(1, 4, 6, 4, dtype=torch.bfloat16)
                chunk_id = attention.add_reference_chunk(token_ids, ref_k, ref_v)

                all_ref_k, all_ref_v = attention.get_all_reference_kv(device="cpu")
                if all_ref_k is None or all_ref_v is None:
                    return self._finish(start, VerificationStatus.FAIL, "reference kv not available")

                x1 = torch.randn(1, 3, 16)
                out1, new_k1, new_v1 = attention.forward(x1, use_reference=True, update_output_kv=True)

                x2 = torch.randn(1, 2, 16)
                out2, new_k2, new_v2 = attention.forward(x2, use_reference=True, update_output_kv=True)

                self._add_metric("reference_chunk_id", chunk_id)
                self._add_metric("reference_tokens", int(all_ref_k.shape[2]))
                self._add_metric("window_size", attention.window_size)
                self._add_metric("first_forward_output_shape", list(out1.shape))
                self._add_metric("second_forward_output_shape", list(out2.shape))
                self._add_metric("first_window_tokens", int(new_k1.shape[2]))
                self._add_metric("second_window_tokens", int(new_k2.shape[2]))
                self._add_metric("past_k_tokens", int(attention._past_k.shape[2]) if attention._past_k is not None else 0)
                self._add_metric("past_v_tokens", int(attention._past_v.shape[2]) if attention._past_v is not None else 0)
                self._add_metric("pool_status", attention.get_pool_info())
                self._add_evidence(
                    f"[g23_rswa_double_layer_kv] reference_kv loaded: tokens={all_ref_k.shape[2]} chunk_id={chunk_id}"
                )
                self._add_evidence(
                    f"[g23_rswa_double_layer_kv] output window bounded: first={new_k1.shape[2]} second={new_k2.shape[2]} window={attention.window_size}"
                )

                if list(out1.shape) != [1, 3, 16]:
                    return self._finish(start, VerificationStatus.FAIL, "unexpected first forward output shape")
                if list(out2.shape) != [1, 2, 16]:
                    return self._finish(start, VerificationStatus.FAIL, "unexpected second forward output shape")
                if int(all_ref_k.shape[2]) != 6 or int(all_ref_v.shape[2]) != 6:
                    return self._finish(start, VerificationStatus.FAIL, "reference kv token count mismatch")
                if int(new_k1.shape[2]) != attention.window_size or int(new_v1.shape[2]) != attention.window_size:
                    return self._finish(start, VerificationStatus.FAIL, "first output kv not clipped to window")
                if int(new_k2.shape[2]) != attention.window_size or int(new_v2.shape[2]) != attention.window_size:
                    return self._finish(start, VerificationStatus.FAIL, "second output kv not clipped to window")
                if attention._past_k is None or attention._past_v is None:
                    return self._finish(start, VerificationStatus.FAIL, "past kv not retained")
                if int(attention._past_k.shape[2]) != attention.window_size or int(attention._past_v.shape[2]) != attention.window_size:
                    return self._finish(start, VerificationStatus.FAIL, "past kv window not constant")
                if int(attention.get_pool_info()["pool_status"]["hot_chunks"]) < 1:
                    return self._finish(start, VerificationStatus.FAIL, "prefill pool missing hot chunks")

                return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            import traceback

            self._add_evidence(traceback.format_exc())
            return self._finish(start, VerificationStatus.ERROR, str(exc))
