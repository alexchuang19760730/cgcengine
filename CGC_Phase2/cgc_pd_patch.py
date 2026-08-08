#!/usr/bin/env python3
"""CGC PD 通用 emit/resume patch — 适用于任何 sglang 模型。

从 V4-Flash deepseek_v4.py 提取 emit 逻辑 + 从 qwen3_vl_resume_patch.py 提取 resume 逻辑，
统一为一套 patch，通过 monkey-patch 适配任意模型。

用法:
  # Cloud (emit) — prefill 后 emit hidden_states 给 edge
  CGC_EMIT_CUT=21 CGC_TRANSPORT=nixl python -m sglang.launch_server ...

  # Edge (resume) — recv hidden_states, 从 cut+1 层 decode
  CGC_RESUME_FROM=21 CGC_TRANSPORT=nixl python -m sglang.launch_server ...

  # 应用 patch (在模型加载后, forward 首次调用前):
  from cgc_pd_patch import apply_pd_patch
  apply_pd_patch(model)  # 自动检测 emit/resume 模式

机制 (与 V4-Flash 一致):
  - emit: 旁路 capture (layer_kv_callback), cloud 继续正常 forward+decode
  - resume: recv hidden_states → 跳过 0..cut 层 → 从 cut+1 开始 forward
  - 每次 forward(EXTEND+DECODE) 都 emit/recv, step 对齐
  - transport: NixlHandoff(VRAM零拷贝) / TcpHandoff / FileHandoff
"""
import os
import sys
import time
import threading
import struct
from typing import Optional, Callable, Any

import torch

# === 全局状态 ===
_CGC_EMIT_CUT = int(os.environ.get("CGC_EMIT_CUT", "-1") or -1)
_CGC_RESUME_CUT = int(os.environ.get("CGC_RESUME_FROM", "-1") or -1)
_CGC_TRANSPORT_MODE = os.environ.get("CGC_TRANSPORT", "file").lower()
_CGC_TRANSPORTS: dict = {}
_CGC_TRANSPORT_LOCK = threading.Lock()
_CGC_EMIT_STEP = 0
_CGC_RESUME_STEP = 0
_CGC_PATCHED = False


def _cgc_rank() -> int:
    """获取当前 TP rank。"""
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return int(_dist.get_rank())
    except Exception:
        pass
    return 0


def _cgc_get_transport(role: str, rank: int):
    """延迟创建 transport (file/tcp/nixl)。"""
    _key = (role, rank, _CGC_TRANSPORT_MODE)
    if _key not in _CGC_TRANSPORTS:
        with _CGC_TRANSPORT_LOCK:
            if _key not in _CGC_TRANSPORTS:
                _repo = os.path.dirname(os.path.abspath(__file__))
                if _repo not in sys.path:
                    sys.path.insert(0, _repo)
                from cgc_handoff_transport import HandoffTransport as _HT
                _port = int(os.environ.get("CGC_TRANSPORT_TCP_PORT", "31000")) + rank
                _CGC_TRANSPORTS[_key] = _HT.make(
                    _CGC_TRANSPORT_MODE,
                    role=role,
                    host="0.0.0.0" if role == "server" else "127.0.0.1",
                    port=_port,
                    connect_host=os.environ.get("CGC_TRANSPORT_TCP_HOST", "127.0.0.1"),
                    rank=rank,
                )
    return _CGC_TRANSPORTS[_key]


def _find_layers(model) -> Optional[list]:
    """自动检测模型的 layers 属性 (支持多种包装层)。"""
    # 直接 layers
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    # model.model.layers (Qwen3LLMModel 包装)
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return layers
    # model.language_model.model.layers (VL 包装)
    lang = getattr(model, "language_model", None)
    if lang is not None:
        inner2 = getattr(lang, "model", lang)
        layers = getattr(inner2, "layers", None)
        if layers is not None:
            return layers
    return None


def _find_embed(model):
    """自动检测 embed_tokens。"""
    for path in [
        lambda m: m.embed_tokens,
        lambda m: m.model.embed_tokens,
        lambda m: m.language_model.model.embed_tokens,
        lambda m: getattr(getattr(m, "model", m), "embed_tokens", None),
    ]:
        try:
            emb = path(model)
            if emb is not None:
                return emb
        except Exception:
            continue
    return None


# === Emit (cloud side) ===

def _apply_emit_patch(model):
    """Patch 模型: 在 emit_cut 层后捕获 hidden_states 并发送给 edge。

    机制: monkey-patch layers[emit_cut].forward, 在原始 forward 后捕获输出。
    Cloud 继续正常 forward (emit 是旁路, 不改变 forward 路径)。
    """
    global _CGC_EMIT_STEP

    layers = _find_layers(model)
    if layers is None:
        print("[CGC_EMIT] WARN: 无法找到 model.layers, emit patch 跳过", flush=True)
        return

    if not (0 <= _CGC_EMIT_CUT < len(layers)):
        print(f"[CGC_EMIT] WARN: emit_cut={_CGC_EMIT_CUT} 超出 layers 范围 "
              f"(0..{len(layers)-1}), emit patch 跳过", flush=True)
        return

    transport = _cgc_get_transport("server", _cgc_rank())
    target_layer = layers[_CGC_EMIT_CUT]
    _orig_forward = target_layer.forward

    print(f"[CGC_EMIT] patching layer {_CGC_EMIT_CUT} "
          f"(model has {len(layers)} layers, rank={_cgc_rank()})", flush=True)

    def _hooked_forward(*args, **kwargs):
        # 调原始 layer forward
        result = _orig_forward(*args, **kwargs)

        # 捕获 hidden_states (layer forward 返回 (hidden_states, ...) 或 hidden_states)
        hs = result[0] if isinstance(result, tuple) else result
        if hs is None or not hasattr(hs, "shape"):
            return result

        _ntok = int(hs.shape[0])
        # NIXL 零拷贝需要 tensor 留在 VRAM; TCP/file 需要 CPU
        if _CGC_TRANSPORT_MODE == "nixl":
            _hs = hs.detach().clone()
        else:
            _hs = hs.detach().cpu().contiguous()

        _payload = {
            "finished_layer": int(_CGC_EMIT_CUT),
            "hidden_states": _hs,
            "step": _CGC_EMIT_STEP,
        }
        try:
            transport.send(_cgc_rank(), _CGC_EMIT_STEP, _payload)
            print(f"[CGC_EMIT] cut={_CGC_EMIT_CUT} step={_CGC_EMIT_STEP} "
                  f"rank={_cgc_rank()} tok={_ntok} "
                  f"shape={tuple(hs.shape)} -> {_CGC_TRANSPORT_MODE}",
                  flush=True)
        except Exception as e:
            print(f"[CGC_EMIT] ERROR send: {e!r}", flush=True)

        return result

    target_layer.forward = _hooked_forward

    # Patch forward_batch 的 forward_mode 检测 (EXTEND=reset step, DECODE=increment)
    # 需要在 model.forward 开头更新 step
    _model_forward = model.forward if hasattr(model, "forward") else None
    if _model_forward is None:
        # 尝试 inner model
        inner = getattr(model, "model", None)
        _model_forward = inner.forward if inner else None

    if _model_forward is not None:
        _orig_model_forward = _model_forward

        def _step_tracking_forward(self, *args, **kwargs):
            global _CGC_EMIT_STEP
            # 检测 forward_mode (EXTEND=1, DECODE=2)
            fb = kwargs.get("forward_batch") or (args[2] if len(args) > 2 else None)
            if fb is not None:
                _mode = int(getattr(fb, "forward_mode", 1))
                if _mode != 2:  # EXTEND = reset
                    _CGC_EMIT_STEP = 0
                # DECODE 不 reset, 自然递增
            return _orig_model_forward(*args, **kwargs)

        # 绑定到正确的对象
        _target = model if hasattr(model, "forward") else getattr(model, "model", model)
        _target.forward = lambda *a, **kw: _step_tracking_forward(_target, *a, **kw)


# === Resume (edge side) ===

def _apply_resume_patch(model):
    """Patch 模型: recv cloud 的 hidden_states, 跳过 0..cut 层, 从 cut+1 开始 forward。

    机制: patch model.forward, 在 forward 开头:
    1. recv hidden_states (transport.recv)
    2. 注入 hidden_states (跳过 embed_tokens + layers 0..cut)
    3. 从 layers[cut+1] 开始 forward
    """
    global _CGC_RESUME_STEP

    layers = _find_layers(model)
    if layers is None:
        print("[CGC_RESUME] WARN: 无法找到 model.layers, resume patch 跳过", flush=True)
        return

    if not (0 <= _CGC_RESUME_CUT < len(layers)):
        print(f"[CGC_RESUME] WARN: resume_cut={_CGC_RESUME_CUT} 超出 layers 范围 "
              f"(0..{len(layers)-1}), resume patch 跳过", flush=True)
        return

    embed = _find_embed(model)
    transport = _cgc_get_transport("client", _cgc_rank())

    # 找到正确的 forward 方法
    _target = model
    _orig_forward = getattr(model, "forward", None)
    if _orig_forward is None:
        _target = getattr(model, "model", model)
        _orig_forward = getattr(_target, "forward", None)

    if _orig_forward is None:
        print("[CGC_RESUME] WARN: 无法找到 model.forward, resume patch 跳过", flush=True)
        return

    _dtype = embed.weight.dtype if embed is not None else torch.float32
    _device = embed.weight.device if embed is not None else torch.device("cuda:0")

    print(f"[CGC_RESUME] patching forward (skip layers 0..{_CGC_RESUME_CUT}, "
          f"resume from {_CGC_RESUME_CUT + 1}, rank={_cgc_rank()})", flush=True)

    def _resume_forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        global _CGC_RESUME_STEP

        # 检测 forward_mode
        _mode = int(getattr(forward_batch, "forward_mode", 1)) if forward_batch else 1
        if _mode != 2:  # EXTEND = reset
            _CGC_RESUME_STEP = 0
        _step = _CGC_RESUME_STEP
        _CGC_RESUME_STEP += 1

        # recv hidden_states from cloud
        try:
            _d = transport.recv(_cgc_rank(), _step, timeout=float(
                os.environ.get("CGC_RESUME_TIMEOUT", "120.0")))
            _hs = _d["hidden_states"].to(device=_device, dtype=_dtype)
            print(f"[CGC_RESUME] cut={_CGC_RESUME_CUT} step={_step} "
                  f"rank={_cgc_rank()} mode={_mode} "
                  f"loaded hidden_states [{tuple(_hs.shape)}] "
                  f"<- {_CGC_TRANSPORT_MODE} transport",
                  flush=True)
        except Exception as e:
            print(f"[CGC_RESUME] WARN step={_step} recv failed: {e!r}, "
                  f"fallback full forward", flush=True)
            return _orig_forward(input_ids, positions, forward_batch, *args, **kwargs)

        # 跳过 embed_tokens + layers 0..cut, 从 cut+1 开始 forward
        hidden_states = _hs

        # 从 cut+1 层开始 forward
        _start = _CGC_RESUME_CUT + 1
        for i in range(_start, len(layers)):
            hidden_states = layers[i](hidden_states, positions, forward_batch,
                                       *args, **kwargs)
            # 处理 tuple 返回
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        return hidden_states

    _target.forward = lambda *a, **kw: _resume_forward(_target, *a, **kw)


# === 入口 ===

def apply_pd_patch(model):
    """自动检测 emit/resume 模式并应用 patch。

    在 sglang 模型加载后调用一次即可。
    - CGC_EMIT_CUT >= 0 → emit patch (cloud side)
    - CGC_RESUME_FROM >= 0 → resume patch (edge side)
    - 两者都 < 0 → 不 patch (正常 forward)
    """
    global _CGC_PATCHED
    if _CGC_PATCHED:
        return
    _CGC_PATCHED = True

    if _CGC_EMIT_CUT >= 0:
        print(f"[CGC_PD] apply emit patch (cut={_CGC_EMIT_CUT}, "
              f"transport={_CGC_TRANSPORT_MODE})", flush=True)
        _apply_emit_patch(model)
    elif _CGC_RESUME_CUT >= 0:
        print(f"[CGC_PD] apply resume patch (from={_CGC_RESUME_CUT}, "
              f"transport={_CGC_TRANSPORT_MODE})", flush=True)
        _apply_resume_patch(model)
    else:
        # 无 patch 模式, 正常 forward
        pass


# === sglang 集成钩子 ===

def patch_sglang_model(model_class):
    """Patch sglang 模型类, 在 forward 首次调用时自动 apply_pd_patch。

    用法 (在 sglang 启动前):
      from cgc_pd_patch import patch_sglang_model
      patch_sglang_model(Qwen3LLMModel)  # 或任何模型类
    """
    _orig_init = model_class.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # 在 init 完成后 (layers 已加载) apply patch
        apply_pd_patch(self)

    model_class.__init__ = _patched_init
    print(f"[CGC_PD] patched {model_class.__name__}.__init__ for auto PD", flush=True)


if __name__ == "__main__":
    # 独立测试: 验证 patch 逻辑
    print(f"CGC_EMIT_CUT={_CGC_EMIT_CUT}")
    print(f"CGC_RESUME_FROM={_CGC_RESUME_CUT}")
    print(f"CGC_TRANSPORT={_CGC_TRANSPORT_MODE}")
    if _CGC_EMIT_CUT >= 0:
        print("Mode: EMIT (cloud)")
    elif _CGC_RESUME_CUT >= 0:
        print("Mode: RESUME (edge)")
    else:
        print("Mode: NONE (normal forward)")
