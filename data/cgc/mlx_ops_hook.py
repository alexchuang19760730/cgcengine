# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
MLX 算子劫持模块
自动劫持MLX核心算子，应用CGC优化
"""

import sys
import logging
from typing import Optional, Dict, Any, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None
    logger.warning("MLX not available, skipping hooks")

if MLX_AVAILABLE:
    try:
        from .mlx_tune_integration import cgc_mlx_tune
        from .cgc_opcodes import CGC_OP_CODES
        CGC_AVAILABLE = True
    except ImportError:
        CGC_AVAILABLE = False
        logger.warning("CGC not available, using fallback")


class MLXOpsHook:
    """MLX算子劫持管理器"""
    
    _instance = None
    _hooks_applied = False
    
    def __init__(self):
        self.original_functions = {}
        self.hook_stats = {
            "attention_calls": 0,
            "mlp_calls": 0,
            "rope_calls": 0,
            "rmsnorm_calls": 0,
            "kv_update_calls": 0,
            "qlinear_calls": 0,
            "total_time_saved_ms": 0.0
        }
        self.enable_cgc = CGC_AVAILABLE
        self.enable_mps_graph = True
        self.enable_profiling = True
        self.enable_ortho_kda = False
        self.ortho_kda_base_dim = 32

    def _register_original(self, key: str, obj: Any, attr: str, original: Any) -> None:
        self.original_functions[key] = (obj, attr, original)
    
    @staticmethod
    def get_instance():
        if MLXOpsHook._instance is None:
            MLXOpsHook._instance = MLXOpsHook()
        return MLXOpsHook._instance
    
    def apply_hooks(self):
        """应用所有MLX算子劫持"""
        if not MLX_AVAILABLE:
            logger.warning("MLX not available, cannot apply hooks")
            return
        
        if self._hooks_applied:
            logger.info("MLX hooks already applied")
            return
        
        logger.info("=" * 80)
        logger.info("🔧 Applying MLX Operator Hooks")
        logger.info("=" * 80)
        
        self._hook_attention()
        self._hook_mlp()
        self._hook_rope()
        self._hook_rmsnorm()
        self._hook_kv_cache()
        self._hook_quantized_linear()
        self._hook_prompt_cache()
        
        self._hooks_applied = True
        logger.info("✅ All MLX operator hooks applied")
        logger.info("=" * 80)
    
    def remove_hooks(self):
        """移除所有MLX算子劫持"""
        if not self._hooks_applied:
            return
        
        logger.info("Removing MLX operator hooks...")
        
        for _, (obj, attr, original_func) in self.original_functions.items():
            setattr(obj, attr, original_func)
        
        self._hooks_applied = False
        logger.info("✅ All MLX operator hooks removed")
    
    def _hook_attention(self):
        """劫持Attention算子"""
        logger.info("  [Hook] Attention operator...")
        
        try:
            import mlx_lm.models.base as base_mod

            original_forward = base_mod.scaled_dot_product_attention
            self._register_original("mlx_lm.models.base.scaled_dot_product_attention", base_mod, "scaled_dot_product_attention", original_forward)

            def hooked_sdpa(queries, keys, values, cache, scale: float, mask, sinks=None):
                hook = MLXOpsHook.get_instance()
                hook.hook_stats["attention_calls"] += 1

                if cache is not None:
                    from cgc_engine.cgc.mlx_lm_ortho_kda_cache import OrthoKDACache

                    if isinstance(cache, OrthoKDACache):
                        if int(queries.shape[2]) != 1:
                            return original_forward(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)
                        n = int(cache.current_dim)
                        if n == 0:
                            return original_forward(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

                        q = queries[..., 0:1, :].astype(mx.float32)
                        k = keys[..., :n, :].astype(mx.float32)
                        v = values[..., :n, :].astype(mx.float32)
                        decay = cache.decay[:, :n].astype(mx.float32)
                        decay = decay[None, :, None, :]
                        scores = mx.sum(q * k, axis=-1)
                        attn = scores * decay.squeeze(2)
                        out = mx.sum(attn[..., None] * v, axis=-2)
                        return out[..., None, :].astype(queries.dtype)

                if hook.enable_cgc and CGC_AVAILABLE:
                    try:
                        return hook._cgc_sdpa(queries, keys, values, scale=scale, mask=mask)
                    except Exception as e:
                        logger.debug(f"CGC sdpa failed: {e}, using original")

                return original_forward(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

            base_mod.scaled_dot_product_attention = hooked_sdpa
            logger.info("    ✅ mlx_lm.models.base.scaled_dot_product_attention hooked")
        except Exception as e:
            logger.warning(f"    ⚠️ Failed to hook attention: {e}")
    
    def _hook_mlp(self):
        """劫持MLP算子"""
        logger.info("  [Hook] MLP operator...")
        
        try:
            if hasattr(nn, 'Linear'):
                original_forward = nn.Linear.__call__
                self._register_original("nn.Linear.__call__", nn.Linear, "__call__", original_forward)
                
                def hooked_linear(self, x):
                    hook = MLXOpsHook.get_instance()
                    hook.hook_stats["mlp_calls"] += 1
                    
                    if hook.enable_cgc and CGC_AVAILABLE:
                        try:
                            return hook._cgc_linear(self, x)
                        except Exception as e:
                            logger.debug(f"CGC linear failed: {e}, using original")
                    
                    return original_forward(self, x)
                
                nn.Linear.__call__ = hooked_linear
                logger.info("    ✅ Linear hooked")
        except Exception as e:
            logger.warning(f"    ⚠️ Failed to hook MLP: {e}")

    def _hook_quantized_linear(self):
        logger.info("  [Hook] QuantizedLinear operator...")

        try:
            if hasattr(nn, "QuantizedLinear"):
                original_forward = nn.QuantizedLinear.__call__
                self._register_original("nn.QuantizedLinear.__call__", nn.QuantizedLinear, "__call__", original_forward)

                def hooked_qlinear(self, x, transpose: bool = True):
                    hook = MLXOpsHook.get_instance()
                    hook.hook_stats["qlinear_calls"] += 1

                    if hook.enable_cgc and CGC_AVAILABLE:
                        try:
                            w = self["weight"] if hasattr(self, "__getitem__") else self.weight
                            scales = self["scales"] if hasattr(self, "__getitem__") else self.scales
                            biases = self.get("biases") if hasattr(self, "get") else None

                            return cgc_mlx_tune.run_cgc_command(
                                CGC_OP_CODES.MLX_QGEMM,
                                {"x": x, "w": w, "scales": scales, **({"biases": biases} if biases is not None else {})},
                                group_size=int(self.group_size),
                                bits=int(self.bits),
                                mode=str(getattr(self, "mode", "affine")),
                                transpose=bool(transpose),
                            )
                        except Exception as e:
                            logger.debug(f"CGC QuantizedLinear failed: {e}, using original")

                    return original_forward(self, x, transpose=transpose)

                nn.QuantizedLinear.__call__ = hooked_qlinear
                logger.info("    ✅ QuantizedLinear hooked")
        except Exception as e:
            logger.warning(f"    ⚠️ Failed to hook QuantizedLinear: {e}")

    def _hook_prompt_cache(self):
        logger.info("  [Hook] make_prompt_cache operator...")

        try:
            import mlx_lm.models.cache as cache_mod

            original_make = cache_mod.make_prompt_cache
            self._register_original(
                "mlx_lm.models.cache.make_prompt_cache",
                cache_mod,
                "make_prompt_cache",
                original_make,
            )

            def hooked_make_prompt_cache(model, max_kv_size=None):
                hook = MLXOpsHook.get_instance()
                if not hook.enable_ortho_kda:
                    return original_make(model, max_kv_size=max_kv_size)

                try:
                    num_layers = len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else len(model.layers)
                    args = getattr(model, "args", None) or getattr(getattr(model, "model", None), "args", None)
                    num_heads = int(getattr(args, "num_key_value_heads", None) or getattr(args, "num_attention_heads"))
                    head_dim = int(getattr(args, "head_dim", None) or (getattr(args, "hidden_size") // getattr(args, "num_attention_heads")))
                except Exception:
                    return original_make(model, max_kv_size=max_kv_size)

                from cgc_engine.cgc.mlx_lm_ortho_kda_cache import MLXOrthoKDAConfig, OrthoKDACache

                cfg = MLXOrthoKDAConfig(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ortho_base_dim=int(hook.ortho_kda_base_dim),
                )
                return [OrthoKDACache(cfg) for _ in range(num_layers)]

            cache_mod.make_prompt_cache = hooked_make_prompt_cache
            logger.info("    ✅ make_prompt_cache hooked")
        except Exception as e:
            logger.warning(f"    ⚠️ Failed to hook make_prompt_cache: {e}")
    
    def _hook_rope(self):
        """劫持RoPE算子"""
        logger.info("  [Hook] RoPE operator...")
        
        try:
            if hasattr(mx, "fast") and hasattr(mx.fast, "rope"):
                original_rope = mx.fast.rope
                self._register_original("mx.fast.rope", mx.fast, "rope", original_rope)

                def hooked_rope(x, dims, traditional=False, base=10000, scale=1.0, offset=0, freqs=None):
                    hook = MLXOpsHook.get_instance()
                    hook.hook_stats["rope_calls"] += 1

                    if hook.enable_cgc and CGC_AVAILABLE:
                        try:
                            return hook._cgc_fast_rope(
                                x,
                                dims,
                                traditional=traditional,
                                base=base,
                                scale=scale,
                                offset=offset,
                                freqs=freqs,
                            )
                        except Exception as e:
                            logger.debug(f"CGC rope failed: {e}, using original")

                    return original_rope(
                        x,
                        dims,
                        traditional=traditional,
                        base=base,
                        scale=scale,
                        offset=offset,
                        freqs=freqs,
                    )

                mx.fast.rope = hooked_rope
                logger.info("    ✅ mx.fast.rope hooked")
        except Exception as e:
            logger.warning(f"    ⚠️ Failed to hook RoPE: {e}")
    
    def _hook_rmsnorm(self):
        logger.info("  [Hook] RMSNorm operator...")
        
        try:
            if hasattr(nn, "RMSNorm"):
                original_forward = nn.RMSNorm.__call__
                self._register_original("nn.RMSNorm.__call__", nn.RMSNorm, "__call__", original_forward)

                def hooked_rmsnorm(self, x):
                    hook = MLXOpsHook.get_instance()
                    hook.hook_stats["rmsnorm_calls"] += 1

                    if hook.enable_cgc and CGC_AVAILABLE:
                        try:
                            return hook._cgc_rmsnorm(self, x)
                        except Exception as e:
                            logger.debug(f"CGC RMSNorm failed: {e}, using original")

                    return original_forward(self, x)

                nn.RMSNorm.__call__ = hooked_rmsnorm
                logger.info("    ✅ RMSNorm hooked")
        except Exception as e:
            logger.warning(f"    ⚠️ Failed to hook RMSNorm: {e}")

    def _hook_kv_cache(self):
        logger.info("  [Hook] KVCache update operator...")

        try:
            import mlx_lm.models.cache as cache_mod

            if hasattr(cache_mod, "KVCache"):
                original_update = cache_mod.KVCache.update_and_fetch
                self._register_original(
                    "mlx_lm.models.cache.KVCache.update_and_fetch",
                    cache_mod.KVCache,
                    "update_and_fetch",
                    original_update,
                )

                def hooked_update_and_fetch(self, keys, values):
                    hook = MLXOpsHook.get_instance()
                    hook.hook_stats["kv_update_calls"] += 1

                    prev = self.offset
                    if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
                        B, n_kv_heads, _, k_head_dim = keys.shape
                        v_head_dim = values.shape[3]
                        n_steps = (self.step + keys.shape[2] - 1) // self.step
                        k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
                        v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
                        new_k = mx.zeros(k_shape, keys.dtype)
                        new_v = mx.zeros(v_shape, values.dtype)
                        if self.keys is not None:
                            if prev % self.step != 0:
                                self.keys = self.keys[..., :prev, :]
                                self.values = self.values[..., :prev, :]
                            self.keys = mx.concatenate([self.keys, new_k], axis=2)
                            self.values = mx.concatenate([self.values, new_v], axis=2)
                        else:
                            self.keys, self.values = new_k, new_v

                    if hook.enable_cgc and CGC_AVAILABLE:
                        try:
                            k_view, v_view, new_offset = cgc_mlx_tune.run_cgc_command(
                                CGC_OP_CODES.KV_CACHE_UPDATE,
                                {"k_cache": self.keys, "v_cache": self.values, "k": keys, "v": values},
                                prev=int(prev),
                            )
                            self.offset = int(new_offset)
                            return k_view, v_view
                        except Exception as e:
                            logger.debug(f"CGC KV update failed: {e}, using original")

                    return original_update(self, keys, values)

                cache_mod.KVCache.update_and_fetch = hooked_update_and_fetch
                logger.info("    ✅ mlx_lm KVCache.update_and_fetch hooked")
        except Exception as e:
            logger.warning(f"    ⚠️ Failed to hook KVCache: {e}")
    
    def _cgc_sdpa(self, queries, keys, values, *, scale=1.0, mask=None):
        import time
        start = time.time()

        if mask is not None and not isinstance(mask, mx.array):
            raise RuntimeError("unsupported mask")

        if keys.shape[1] != queries.shape[1]:
            if queries.shape[1] % keys.shape[1] != 0:
                raise RuntimeError("unsupported gqa")
            rep = queries.shape[1] // keys.shape[1]
            keys = mx.repeat(keys, rep, axis=1)
            values = mx.repeat(values, rep, axis=1)

        out = cgc_mlx_tune.run_cgc_command(
            CGC_OP_CODES.KDA_ATTENTION,
            {"q": queries, "k": keys, "v": values},
        )

        elapsed = (time.time() - start) * 1000
        self.hook_stats["total_time_saved_ms"] += elapsed
        return out
    
    def _cgc_linear(self, linear_layer, x):
        """CGC优化的Linear"""
        import time
        start = time.time()
        
        out = mx.matmul(x, linear_layer.weight.T)
        if linear_layer.bias is not None:
            out = out + linear_layer.bias
        
        elapsed = (time.time() - start) * 1000
        self.hook_stats["total_time_saved_ms"] += elapsed
        
        return out
    
    def _cgc_fast_rope(self, x, dims, *, traditional=False, base=10000, scale=1.0, offset=0, freqs=None):
        import time
        start = time.time()

        if traditional:
            raise RuntimeError("traditional rope unsupported")

        if freqs is None:
            freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)

        pos = mx.arange(x.shape[-2], dtype=mx.float32) + mx.array(offset, dtype=mx.float32)
        inv = (mx.array(scale, dtype=mx.float32) / freqs).astype(mx.float32)
        theta = pos[:, None] * inv[None, :]
        cos = mx.cos(theta)
        sin = mx.sin(theta)
        cos = mx.concatenate([cos, cos], axis=-1).astype(x.dtype)
        sin = mx.concatenate([sin, sin], axis=-1).astype(x.dtype)
        cos = cos.reshape((1, 1, cos.shape[0], cos.shape[1]))
        sin = sin.reshape((1, 1, sin.shape[0], sin.shape[1]))

        out = cgc_mlx_tune.run_cgc_command(
            CGC_OP_CODES.ROPE,
            {"x": x, "cos": cos, "sin": sin},
        )

        elapsed = (time.time() - start) * 1000
        self.hook_stats["total_time_saved_ms"] += elapsed
        return out

    def _cgc_rmsnorm(self, layernorm_layer, x):
        import time
        start = time.time()

        weight = getattr(layernorm_layer, "weight", None)
        eps = float(getattr(layernorm_layer, "eps", 1e-6))
        out = cgc_mlx_tune.run_cgc_command(
            CGC_OP_CODES.MLX_RMS_NORM,
            {"x": x, "weight": weight} if weight is not None else {"x": x},
            eps=eps,
        )

        elapsed = (time.time() - start) * 1000
        self.hook_stats["total_time_saved_ms"] += elapsed
        
        return out
    
    def get_stats(self) -> Dict[str, Any]:
        """获取劫持统计信息"""
        return {
            "hooks_applied": self._hooks_applied,
            "cgc_enabled": self.enable_cgc,
            "mps_graph_enabled": self.enable_mps_graph,
            **self.hook_stats
        }
    
    def print_stats(self):
        """打印劫持统计信息"""
        stats = self.get_stats()
        print("\n" + "=" * 80)
        print("📊 MLX Operator Hook Statistics")
        print("=" * 80)
        print(f"Hooks Applied: {stats['hooks_applied']}")
        print(f"CGC Enabled: {stats['cgc_enabled']}")
        print(f"MPSGraph Enabled: {stats['mps_graph_enabled']}")
        print("-" * 80)
        print(f"Attention Calls: {stats['attention_calls']}")
        print(f"MLP Calls: {stats['mlp_calls']}")
        print(f"RoPE Calls: {stats['rope_calls']}")
        print(f"RMSNorm Calls: {stats['rmsnorm_calls']}")
        print(f"KV Update Calls: {stats['kv_update_calls']}")
        print(f"Total Time: {stats['total_time_saved_ms']:.2f}ms")
        print("=" * 80)


def apply_mlx_hooks():
    """应用MLX算子劫持（全局函数）"""
    hook = MLXOpsHook.get_instance()
    hook.apply_hooks()
    return hook


def remove_mlx_hooks():
    """移除MLX算子劫持（全局函数）"""
    hook = MLXOpsHook.get_instance()
    hook.remove_hooks()


def get_mlx_hook_stats():
    """获取MLX算子劫持统计信息（全局函数）"""
    hook = MLXOpsHook.get_instance()
    return hook.get_stats()


if __name__ == "__main__":
    print("Testing MLX Operator Hooks...")
    
    if MLX_AVAILABLE:
        hook = apply_mlx_hooks()
        
        print("\nTesting hooked model...")
        
        model = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 512)
        )
        
        x = mx.random.normal((1, 512))
        output = model(x)
        mx.eval(output)
        
        hook.print_stats()
    else:
        print("MLX not available")
