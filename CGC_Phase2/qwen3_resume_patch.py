#!/usr/bin/env python3
"""CGC Qwen3 resume patch (sglang-native, ported from deepseek_v4.py).

把 deepseek_v4.py 的 CGC_RESUME_FROM resume 机制移植到 sglang Qwen3 MoE，
让云端 sglang server 能从 Mac emit 的 hidden_P resume forward 后 L-P 层，
解开 layer-split 反向 (Mac→cloud) 的 cloud resume 阻塞。

设计 (Design)
============
mirror deepseek_v4.py 的两层 resume:

1. Qwen3MoeForCausalLM.forward  (ForCausalLM 层, CGC_RESUME_FROM 派发)
   - 读 CGC_RESUME_FROM=P → _resume_cut
   - 每 forward 维护 _CGC_RESUME_STEP (EXTEND forward_mode!=2 时 reset 0, decode 递增)
   - _cgc_get_resume_transport(rank, mode) → MacEmitHandoff(role=receiver)
     .recv(rank, step) 取 Mac 推送的 hidden_P payload
   - payload = {finished_layer, hidden_states, residual, step}
   - 调 self.model(..., finished_layer=P, hidden_states_ref=hs, residual_ref=res)

2. Qwen3MoeModel.forward  (Model 层, 层级接续)
   - finished_layer is not None → layer_resume_mode
   - hidden_states_ref 注入: load_hidden_states(hs) 跳过 embed
   - residual_ref 注入: load_hidden_states(res) (Qwen3 层需 hidden+residual 两参,
     不同于 deepseek 的单 hidden 流; residual=None 时首层 prepare_attn 回退)
   - loop_start = max(start_layer, finished_layer+1) 跳过 0..P-1
   - 层循环 self.layers[i](positions, hidden_states, forward_batch, residual)
   - norm + return

非 resume 路径 (CGC_RESUME_FROM 未设) 完整委托原 forward, 零侵入。

与 deepseek_v4.py 的差异 (Qwen3 适配)
   - 无 hc_mult 维度 (deepseek 需 unsqueeze hc_mult, Qwen3 hidden 直接 [N,H])
   - 层 forward 返回 (hidden_states, residual) 两元组 (deepseek 返回 4 元组)
   - residual 需单独注入 (deepseek 仅 hidden_states_ref 单参)
   - transport role=receiver (Mac→cloud 反向; deepseek cloud→edge 用 role=client)

transport
   - CGC_TRANSPORT=mac_emit (默认, layer-split 反向): cloud MacEmitHandoff(receiver)
     TCP server 监听 CGC_MAC_EMIT_PORT+rank, Mac emitter PUT push, cloud recv() pop
   - CGC_TRANSPORT=file: 测试用, torch.load {CGC_HANDOFF_PATH}.rank{r}.step{s}

用法 (Usage)
   apply :  python3 qwen3_resume_patch.py --target <qwen3_moe.py>
   revert:  python3 qwen3_resume_patch.py --target <qwen3_moe.py> --revert

   默认 target = /usr/local/lib/python3.12/dist-packages/sglang/srt/models/qwen3_moe.py

环境变量 (Env, 云端 sglang server)
   CGC_RESUME_FROM=P           从第 P 层 resume (Mac 已跑 0..P-1)
   CGC_TRANSPORT=mac_emit      transport 模式 (默认) | file
   CGC_MAC_EMIT_PORT=31010     receiver 监听 port (base, +rank)
   CGC_RESUME_TIMEOUT=120.0    recv 超时秒
   CGC_HANDOFF_PATH            file 模式路径前缀

依赖: cgc_handoff_transport.py 须在 qwen3_moe.py 同目录或 sys.path
"""

from __future__ import annotations

import os
import sys
import shutil
import datetime

# === 要追加到 qwen3_moe.py 末尾的 patch block ===
# 用原字符串拼接, 避免与 qwen3_moe.py 已有 import 冲突; block 内自含所需 import。
PATCH_BLOCK = r'''

# === CGC Qwen3 resume patch (ported from deepseek_v4.py) =====================
# Enables cloud sglang server to resume forward from Mac-emitted hidden_P
# (layer-split reverse: Mac runs layers 0..P-1, cloud resumes P..end).
# Env: CGC_RESUME_FROM=P, CGC_TRANSPORT=mac_emit|file.
# Mirrors deepseek_v4.py CGC_RESUME_FROM dispatch + Model.forward layer-resume.
# Non-resume path delegates to the original forward (zero-invasive).
import os as _cgc_os
import sys as _cgc_sys
import threading as _cgc_threading
from contextlib import nullcontext as _cgc_nullctx

_CGC_RESUME_STEP = 0
_CGC_TRANSPORTS = {}
_CGC_TRANSPORT_LOCK = _cgc_threading.Lock()
_CGC_PATCH_MARKER = "cgc_qwen3_resume_v1"


def _cgc_rank():
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return int(_dist.get_rank())
    except Exception:
        pass
    return 0


def _cgc_load_hidden_states(ref, device, dtype):
    """Resolve a hidden-states reference into a tensor on device/dtype.

    ref may be: torch.Tensor (moved), bytes/bytearray (BytesIO load),
    str path (torch.load). Same semantics as deepseek_v4.load_hidden_states_from_ref.
    """
    import io
    import torch
    if torch.is_tensor(ref):
        return ref.to(device=device, dtype=dtype)
    if isinstance(ref, (bytes, bytearray)):
        return torch.load(io.BytesIO(ref), map_location=device, weights_only=True)
    if isinstance(ref, str):
        return torch.load(ref, map_location=device, weights_only=True)
    raise TypeError(f"_cgc_load_hidden_states: unsupported ref type {type(ref)!r}")


def _cgc_get_resume_transport(rank, mode):
    """Cloud-side receiver transport, cached per (rank, mode).

    mac_emit: MacEmitHandoff(role=receiver) — TCP server on
              CGC_MAC_EMIT_PORT+rank; Mac emitter PUTs hidden_P; cloud recv()
              pops in-process. This is the layer-split reverse path.
    file:     None — caller falls back to torch.load of the step file.
    """
    _key = ("resume_recv", int(rank), str(mode))
    if _key not in _CGC_TRANSPORTS:
        with _CGC_TRANSPORT_LOCK:
            if _key not in _CGC_TRANSPORTS:
                if mode == "mac_emit":
                    _port = int(_cgc_os.environ.get("CGC_MAC_EMIT_PORT", "31010")) + int(rank)
                    _dir = _cgc_os.path.dirname(_cgc_os.path.abspath(__file__))
                    if _dir not in _cgc_sys.path:
                        _cgc_sys.path.insert(0, _dir)
                    from cgc_handoff_transport import HandoffTransport as _HT
                    _CGC_TRANSPORTS[_key] = _HT.make(
                        "mac_emit",
                        role="receiver",
                        host="0.0.0.0",
                        port=_port,
                        connect_host=_cgc_os.environ.get(
                            "CGC_MAC_EMIT_CLOUD_HOST", "127.0.0.1"
                        ),
                    )
                    print(
                        f"[CGC_RESUME] mac_emit receiver ready on 0.0.0.0:{_port} "
                        f"rank={rank}", flush=True,
                    )
                else:
                    _CGC_TRANSPORTS[_key] = None  # file mode
    return _CGC_TRANSPORTS[_key]


# Save originals so the non-resume path is a zero-invasive delegate.
_Qwen3MoeModel_forward_orig = Qwen3MoeModel.forward
_Qwen3MoeForCausalLM_forward_orig = Qwen3MoeForCausalLM.forward


def _qwen3moe_model_forward_resume(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds=None,
    pp_proxy_tensors=None,
    finished_layer=None,
    hidden_states_ref=None,
    residual_ref=None,
    layer_kv_callback=None,
):
    """Qwen3MoeModel.forward with layer-resume (finished_layer 接续).

    finished_layer is None  -> delegate to original forward (unchanged).
    finished_layer is not None:
      - hidden_states_ref given -> inject (skip embed); else embed normally (fallback)
      - residual_ref given -> inject; else None (first resumed layer treats hs as residual)
      - loop_start = max(start_layer, finished_layer+1) -> skip layers 0..P-1
    Mirrors deepseek_v4.DeepseekV4Model.forward layer-resume, adapted for Qwen3's
    (hidden_states, residual) two-arg layer signature (no hc_mult dim).
    """
    if finished_layer is None:
        return _Qwen3MoeModel_forward_orig(
            self, input_ids, positions, forward_batch, input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    _dtype = self.embed_tokens.weight.dtype
    _device = input_ids.device

    # --- inject hidden_states (skip embed) or embed normally (fallback) ---
    if hidden_states_ref is not None:
        hidden_states = _cgc_load_hidden_states(hidden_states_ref, _device, _dtype)
    elif self.pp_group.is_first_rank:
        hidden_states = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
    else:
        assert pp_proxy_tensors is not None
        hidden_states = pp_proxy_tensors["hidden_states"]

    # --- inject residual (Qwen3 layer needs both hs + residual) ---
    if residual_ref is not None:
        residual = _cgc_load_hidden_states(residual_ref, _device, _dtype)
    elif self.pp_group.is_first_rank:
        residual = None
    elif pp_proxy_tensors is not None:
        residual = pp_proxy_tensors.get("residual")
    else:
        residual = None

    aux_hidden_states = []
    loop_start = max(self.start_layer, int(finished_layer) + 1)
    for i in range(loop_start, self.end_layer):
        ctx = (
            _cgc_nullctx()
            if not get_global_server_args().disable_piecewise_cuda_graph
            else get_global_expert_distribution_recorder().with_current_layer(i)
        )
        with ctx:
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                captured_last_layer_outputs=(
                    aux_hidden_states
                    if getattr(layer, "_is_layer_to_capture", False)
                    else None
                ),
            )
        if layer_kv_callback is not None:
            layer_kv_callback(i, hidden_states, forward_batch)

    if not self.pp_group.is_last_rank:
        return PPProxyTensors({"hidden_states": hidden_states, "residual": residual})
    if hidden_states.shape[0] != 0:
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
    if not aux_hidden_states:
        return hidden_states
    return hidden_states, aux_hidden_states


def _qwen3moe_forcausal_forward_resume(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds=None,
    pp_proxy_tensors=None,
):
    """Qwen3MoeForCausalLM.forward with CGC_RESUME_FROM dispatch (Mac→cloud reverse).

    CGC_RESUME_FROM unset/invalid -> delegate to original forward (unchanged).
    0 <= CGC_RESUME_FROM < end_layer:
      - per-step counter (EXTEND resets to 0, decode increments) mirrors deepseek_v4
      - transport.recv(rank, step) -> {hidden_states, residual, finished_layer}
      - self.model(..., finished_layer=P, hidden_states_ref=hs, residual_ref=res)
      - logits_processor on last rank
    """
    _resume_cut = int(_cgc_os.environ.get("CGC_RESUME_FROM", "-1") or -1)
    _end = getattr(self.model, "end_layer", 10 ** 9)
    if not (0 <= _resume_cut < _end):
        return _Qwen3MoeForCausalLM_forward_orig(
            self, input_ids, positions, forward_batch, input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    global _CGC_RESUME_STEP
    _mode = (
        int(getattr(forward_batch, "forward_mode", 1))
        if forward_batch is not None
        else 1
    )
    if _mode != 2:  # EXTEND = start of a new request -> reset step counter
        _CGC_RESUME_STEP = 0
    _step = _CGC_RESUME_STEP
    _CGC_RESUME_STEP += 1

    _transport_mode = _cgc_os.environ.get("CGC_TRANSPORT", "mac_emit").lower()
    _transport = _cgc_get_resume_transport(_cgc_rank(), _transport_mode)
    _dtype = self.model.embed_tokens.weight.dtype
    _device = input_ids.device
    _hs = None
    _res = None

    if _transport is not None:
        try:
            _d = _transport.recv(
                _cgc_rank(), _step,
                timeout=float(_cgc_os.environ.get("CGC_RESUME_TIMEOUT", "120.0")),
            )
            _hs = _d["hidden_states"].to(device=_device, dtype=_dtype)
            _res = _d.get("residual")
            if _res is not None:
                _res = _res.to(device=_device, dtype=_dtype)
            print(
                f"[CGC_RESUME] from={_resume_cut} step={_step} rank={_cgc_rank()} "
                f"mode={_mode} loaded hidden_states [{tuple(_hs.shape)}] "
                f"<- {_transport_mode} transport",
                flush=True,
            )
        except Exception as _e:
            print(
                f"[CGC_RESUME] WARN from={_resume_cut} step={_step} "
                f"rank={_cgc_rank()} mode={_mode} transport recv failed "
                f"({_e!r}), full forward fallback",
                flush=True,
            )
    else:
        _p = (
            f"{_cgc_os.environ.get('CGC_HANDOFF_PATH', '/data/cgc_handoff.pt')}"
            f".rank{_cgc_rank()}.step{_step}"
        )
        if _cgc_os.path.exists(_p):
            _d = torch.load(_p, map_location=_device, weights_only=True)
            _hs = _d["hidden_states"].to(device=_device, dtype=_dtype)
            _res = _d.get("residual")
            if _res is not None:
                _res = _res.to(device=_device, dtype=_dtype)
            print(
                f"[CGC_RESUME] from={_resume_cut} step={_step} rank={_cgc_rank()} "
                f"mode={_mode} loaded hidden_states [{tuple(_hs.shape)}] <- {_p}",
                flush=True,
            )
        else:
            print(
                f"[CGC_RESUME] WARN from={_resume_cut} step={_step} "
                f"rank={_cgc_rank()} mode={_mode} step file missing, "
                f"full forward fallback",
                flush=True,
            )

    if _hs is not None:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            finished_layer=int(_resume_cut),
            hidden_states_ref=_hs,
            residual_ref=_res,
        )
    else:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            finished_layer=int(_resume_cut),
        )

    aux_hidden_states = None
    if self.capture_aux_hidden_states:
        hidden_states, aux_hidden_states = hidden_states

    if self.pp_group.is_last_rank:
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )
    return hidden_states


# Wire the resume forwards in (non-resume still delegates to the saved originals).
Qwen3MoeModel.forward = _qwen3moe_model_forward_resume
Qwen3MoeForCausalLM.forward = _qwen3moe_forcausal_forward_resume
# === end CGC Qwen3 resume patch =============================================
'''

MARKER = "_CGC_PATCH_MARKER = \"cgc_qwen3_resume_v1\""
DEFAULT_TARGET = (
    "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/qwen3_moe.py"
)


def _is_patched(src: str) -> bool:
    return "_CGC_PATCH_MARKER" in src or "CGC Qwen3 resume patch" in src


def apply(target: str, dry_run: bool = False) -> int:
    with open(target, "r", encoding="utf-8") as f:
        src = f.read()

    if _is_patched(src):
        print(f"[skip] {target} 已含 CGC resume patch marker, 跳过")
        return 0

    # backup
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{target}.bak.{ts}"
    shutil.copy2(target, bak)
    print(f"[backup] {target} -> {bak}")

    patched = src.rstrip() + "\n" + PATCH_BLOCK

    if dry_run:
        print(f"[dry-run] would write {len(patched)} bytes to {target}")
        return 0

    with open(target, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"[patched] {target} (+{len(PATCH_BLOCK)} bytes)")
    print(
        "[note] 确保云端 cgc_handoff_transport.py 在 qwen3_moe.py 同目录或 sys.path; "
        "启动 sglang server 时设 CGC_RESUME_FROM=P CGC_TRANSPORT=mac_emit"
    )
    return 0


def revert(target: str) -> int:
    import glob

    with open(target, "r", encoding="utf-8") as f:
        src = f.read()

    if not _is_patched(src):
        print(f"[skip] {target} 未含 patch marker, 无需 revert")
        return 0

    idx = src.find("# === CGC Qwen3 resume patch (ported from deepseek_v4.py)")
    if idx < 0:
        print(f"[warn] 找不到 patch 起始标记, 改用最近 backup 恢复")
    else:
        head = src[:idx].rstrip() + "\n"
        # 优先写回截断版
        with open(target, "w", encoding="utf-8") as f:
            f.write(head)
        print(f"[reverted] {target} (移除 patch block)")
        return 0

    baks = sorted(glob.glob(f"{target}.bak.*"))
    if not baks:
        print(f"[error] 无 backup 文件可恢复: {target}.bak.*")
        return 1
    shutil.copy2(baks[-1], target)
    print(f"[reverted] {target} <- {baks[-1]}")
    return 0


def main():
    import argparse

    ap = argparse.ArgumentParser(description="CGC Qwen3 resume patch (sglang-native)")
    ap.add_argument(
        "--target", default=DEFAULT_TARGET,
        help=f"qwen3_moe.py 路径 (默认 {DEFAULT_TARGET})",
    )
    ap.add_argument("--revert", action="store_true", help="移除 patch (恢复原文件)")
    ap.add_argument("--dry-run", action="store_true", help="只打印不写")
    args = ap.parse_args()

    if not os.path.exists(args.target):
        print(f"[error] target 不存在: {args.target}")
        return 2

    if args.revert:
        return revert(args.target)
    return apply(args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
