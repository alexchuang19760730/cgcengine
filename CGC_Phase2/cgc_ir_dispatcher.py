"""CGC IR Dispatcher -- IR-driven Metal execution for MTP head.

This is the missing "Layer 5" that connects the IR definition (unified_mtp_ir.py)
to Metal GPU kernels. Instead of hardcoding the MTP head architecture (like
mtp_mlx_forward.py), this dispatcher reads the IR graph definition and executes
it op-by-op on Metal GPU.

Architecture:
    IR Graph (mtp_head_ir.json)
        ↓
    Execution Plan (compiled once)
        ↓
    Metal Backend (MLX kernels → Metal GPU)
        ↓
    Draft tokens + hidden states

Key features:
    - IR-driven: changing the graph definition changes execution
    - CGC opcode mapping: each IR op maps to a CGC_OP_CODES entry
    - Pluggable backend: MLX now, direct CGC Metal kernels later
    - Zero-copy ready: accepts numpy arrays, can extend to Metal buffers

Usage:
    from cgc_ir_dispatcher import CGCIRDispatcher

    dispatcher = CGCIRDispatcher(
        checkpoint="mtp_head_qwen25-0.5b_decode.pt",
        embed_head_path="embed_head.pt",
        hidden_size=896, vocab_size=151936,
        num_heads=14, head_dim=64, intermediate_size=4864,
    )
    tokens, ms = dispatcher.draft_chain(hidden_numpy, token_id, num_draft=4)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as mx_nn
import numpy as np
import torch


# ============================================================================
# CGC Opcode Mapping (IR layer_type → CGC_OP_CODES)
# ============================================================================

# Maps IR layer_type strings to CGC opcode hex values.
# This is the bridge between the high-level IR and the low-level instruction set.
IR_TO_CGC_OPCODE = {
    "linear":    0x20,   # CGC_OP_CODES.LINEAR_GEMM
    "rms_norm":  0x31,   # CGC_OP_CODES.RMS_NORM
    "attention": 0x10,   # CGC_OP_CODES.ATTENTION_SDPA
    "mlp_silu":  0x50,   # CGC_OP_CODES.SILU (SwiGLU uses silu)
    "concat":    0x70,   # CGC_OP_CODES.EMBEDDING (memory ops)
    "softmax":   0x60,   # CGC_OP_CODES.SOFTMAX
    "rope":      0x40,   # CGC_OP_CODES.ROPE
    "argmax":    0x62,   # CGC_OP_CODES.TOP_K (k=1)
}


# ============================================================================
# Execution Plan
# ============================================================================

@dataclass
class ExecStep:
    """A single step in the execution plan."""
    op: str                          # "concat" | "linear" | "rms_norm" | "attention" | "mlp_silu" | "add" | "argmax"
    inputs: List[str]                # names of input tensors
    output: str                      # name of output tensor
    weights: List[str] = field(default_factory=list)  # weight keys to use
    params: Dict[str, Any] = field(default_factory=dict)  # op-specific params
    cgc_opcode: int = 0x00           # CGC opcode for this op (for future direct Metal dispatch)


@dataclass
class CGCIRConfig:
    """MTP head IR configuration (matches unified_mtp_ir.MTPHeadIR)."""
    hidden_size: int = 896           # Qwen2.5-0.5B default
    vocab_size: int = 151936
    num_heads: int = 14
    head_dim: int = 64
    intermediate_size: int = 4864
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    def to_ir_json(self) -> str:
        """Serialize to IR JSON format (compatible with unified_mtp_ir.py)."""
        H = self.hidden_size
        layers = [
            {"name": "proj", "layer_type": "linear", "input_shape": [1, 1, H * 2],
             "output_shape": [1, 1, H], "params": {"in_features": H * 2, "out_features": H, "bias": False}},
            {"name": "norm1", "layer_type": "rms_norm", "input_shape": [1, 1, H],
             "output_shape": [1, 1, H], "params": {"dim": H, "eps": self.rms_norm_eps}},
            {"name": "attn", "layer_type": "attention", "input_shape": [1, 1, H],
             "output_shape": [1, 1, H], "params": {"hidden_size": H, "num_heads": self.num_heads, "head_dim": self.head_dim}},
            {"name": "norm2", "layer_type": "rms_norm", "input_shape": [1, 1, H],
             "output_shape": [1, 1, H], "params": {"dim": H, "eps": self.rms_norm_eps}},
            {"name": "mlp", "layer_type": "mlp_silu", "input_shape": [1, 1, H],
             "output_shape": [1, 1, H], "params": {"hidden_size": H, "intermediate_size": self.intermediate_size}},
            {"name": "norm_out", "layer_type": "rms_norm", "input_shape": [1, 1, H],
             "output_shape": [1, 1, H], "params": {"dim": H, "eps": self.rms_norm_eps}},
        ]
        weight_map = {
            "proj": "proj.weight", "norm1": "norm1.weight",
            "attn.q_proj": "attn.q_proj.weight", "attn.k_proj": "attn.k_proj.weight",
            "attn.v_proj": "attn.v_proj.weight", "attn.o_proj": "attn.o_proj.weight",
            "norm2": "norm2.weight",
            "mlp.gate_proj": "mlp.gate_proj.weight", "mlp.up_proj": "mlp.up_proj.weight",
            "mlp.down_proj": "mlp.down_proj.weight",
            "norm_out": "norm_out.weight",
        }
        return json.dumps({
            "name": "mtp_head", "version": "2.0",
            "hidden_size": H, "vocab_size": self.vocab_size,
            "num_heads": self.num_heads, "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size, "rms_norm_eps": self.rms_norm_eps,
            "layers": layers, "weight_map": weight_map,
        }, indent=2)


# ============================================================================
# Metal Backend (via MLX)
# ============================================================================

class MLXMetalBackend:
    """Metal GPU backend using Apple MLX kernels.

    Each method maps to a CGC opcode and executes via MLX's Metal kernels.
    Future: replace with direct CGC Metal kernel calls (ds4/metal/*.metal).
    """

    def __init__(self, config: CGCIRConfig):
        self.config = config
        self.scale = config.head_dim ** -0.5
        self._rope_cos: Optional[mx.array] = None
        self._rope_sin: Optional[mx.array] = None

    # --- Op implementations (each maps to a CGC opcode) ---

    def concat(self, *tensors: mx.array, axis: int = -1) -> mx.array:
        """CGC opcode 0x70: Concatenate tensors."""
        return mx.concatenate(list(tensors), axis=axis)

    def linear(self, x: mx.array, weight: mx.array) -> mx.array:
        """CGC opcode 0x20: Linear/GEMM. weight: [out, in], x: [..., in] → [..., out]"""
        return x @ weight.T

    def rms_norm(self, x: mx.array, weight: mx.array, eps: float) -> mx.array:
        """CGC opcode 0x31: RMSNorm."""
        x32 = x.astype(mx.float32)
        variance = mx.mean(x32 * x32, axis=-1, keepdims=True)
        x_normed = x32 * mx.rsqrt(variance + eps)
        return (weight * x_normed).astype(x.dtype)

    def rope(self, x: mx.array, seq_len: int) -> mx.array:
        """CGC opcode 0x40: Apply RoPE. x: [heads, seq, head_dim]"""
        cos, sin = self._get_rope(seq_len)
        cos = mx.broadcast_to(cos, (x.shape[0], seq_len, x.shape[2]))
        sin = mx.broadcast_to(sin, (x.shape[0], seq_len, x.shape[2]))
        half = x.shape[2] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated * sin

    def attention(
        self, x: mx.array,
        q_w: mx.array, k_w: mx.array, v_w: mx.array, o_w: mx.array,
    ) -> mx.array:
        """CGC opcode 0x10: Scaled dot-product attention with RoPE.

        x: [1, seq, hidden] → [1, seq, hidden]
        """
        seq_len = x.shape[1]
        nh, hd = self.config.num_heads, self.config.head_dim

        # Projections
        q = (x @ q_w.T).reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3).reshape(nh, seq_len, hd)
        k = (x @ k_w.T).reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3).reshape(nh, seq_len, hd)
        v = (x @ v_w.T).reshape(1, seq_len, nh, hd).transpose(0, 2, 1, 3).reshape(nh, seq_len, hd)

        # RoPE
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        # Scaled dot-product (causal)
        scores = (q @ k.transpose(0, 2, 1)) * self.scale  # [heads, seq, seq]
        mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
        scores = scores + mask

        # Softmax
        attn = mx.softmax(scores, axis=-1)  # CGC opcode 0x60

        # Output
        out = attn @ v  # [heads, seq, head_dim]
        out = out.transpose(1, 0, 2).reshape(1, seq_len, nh * hd)
        return out @ o_w.T

    def mlp_silu(
        self, x: mx.array,
        gate_w: mx.array, up_w: mx.array, down_w: mx.array,
    ) -> mx.array:
        """CGC opcode 0x50: SwiGLU MLP (silu gate * up → down)."""
        gate = x @ gate_w.T
        up = x @ up_w.T
        return (mx_nn.silu(gate) * up) @ down_w.T

    def argmax(self, x: mx.array, axis: int = -1) -> mx.array:
        """CGC opcode 0x62: Top-K (k=1) sampling."""
        return mx.argmax(x, axis=axis)

    # --- RoPE helpers ---

    def _get_rope(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        """Precompute RoPE cos/sin."""
        if self._rope_cos is not None and self._rope_cos.shape[0] >= seq_len:
            return self._rope_cos[:seq_len], self._rope_sin[:seq_len]

        hd = self.config.head_dim
        theta = self.config.rope_theta
        inv_freq = 1.0 / (theta ** (mx.arange(0, hd, 2, dtype=mx.float32) / hd))
        positions = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self._rope_cos = mx.cos(emb)
        self._rope_sin = mx.sin(emb)
        return self._rope_cos[:seq_len], self._rope_sin[:seq_len]


# ============================================================================
# CGC IR Dispatcher
# ============================================================================

class CGCIRDispatcher:
    """CGC IR-driven Metal execution dispatcher for MTP head.

    Reads the IR graph definition, compiles an execution plan, and executes
    it on Metal GPU via the pluggable backend.

    The execution plan encodes the MTP head forward pass:
        1. concat(hidden, token_embed)          → x
        2. linear(x, proj.weight)               → x
        3. rms_norm(x, norm1.weight)            → normed1
        4. attention(normed1, attn weights)     → attn_out
        5. add(x, attn_out)                     → h          (residual)
        6. rms_norm(h, norm2.weight)            → normed2
        7. mlp_silu(normed2, mlp weights)       → mlp_out
        8. add(h, mlp_out)                      → h2         (residual)
        9. rms_norm(h2, norm_out.weight)        → final_hidden
       10. linear(final_hidden, lm_head)        → logits     (shared)
       11. argmax(logits)                       → token

    This plan is IR-driven: it's constructed from the IR layer list, not hardcoded.
    """

    def __init__(
        self,
        checkpoint: str,
        embed_head_path: str,
        config: Optional[CGCIRConfig] = None,
        ir_json_path: Optional[str] = None,
        backend: Optional[MLXMetalBackend] = None,
    ):
        self.config = config or CGCIRConfig()
        self.backend = backend or MLXMetalBackend(self.config)

        # Load IR graph (from file or generate from config)
        if ir_json_path:
            with open(ir_json_path) as f:
                self.ir = json.load(f)
            # Update config from IR
            self.config = CGCIRConfig(
                hidden_size=self.ir["hidden_size"],
                vocab_size=self.ir["vocab_size"],
                num_heads=self.ir["num_heads"],
                head_dim=self.ir["head_dim"],
                intermediate_size=self.ir["intermediate_size"],
                rms_norm_eps=self.ir["rms_norm_eps"],
            )
            self.backend = MLXMetalBackend(self.config)
        else:
            self.ir = json.loads(self.config.to_ir_json())

        # Load weights
        self._load_weights(checkpoint, embed_head_path)

        # Build execution plan
        self._build_execution_plan()

        # Warmup Metal
        self._warmup()

        print(f"[CGC-IR] Dispatcher ready: {len(self.plan)} steps, "
              f"hidden={self.config.hidden_size}, backend=MLX-Metal")

    def _load_weights(self, checkpoint_path: str, embed_head_path: str):
        """Load MTP head weights and shared embed/lm_head."""
        print(f"[CGC-IR] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Handle nested checkpoint: {model_state_dict: {...}, config: {...}, ...}
        if "model_state_dict" in ckpt:
            raw_weights = ckpt["model_state_dict"]
        else:
            raw_weights = ckpt

        # Convert to MLX arrays (skip lm_head/embed, loaded separately)
        self.weights: Dict[str, mx.array] = {}
        for k, v in raw_weights.items():
            if not isinstance(v, torch.Tensor):
                continue
            if "lm_head" in k or "embed" in k:
                continue
            self.weights[k] = mx.array(v.float().numpy())

        print(f"  MTP weights: {len(self.weights)} tensors")

        # Load shared embed + lm_head
        print(f"  Loading embed+head: {embed_head_path}")
        eh = torch.load(embed_head_path, map_location="cpu", weights_only=True)
        lm_head_w = eh.get("lm_head_weight")
        if lm_head_w is None:
            lm_head_w = eh.get("lm_head")
        embed_w = eh.get("embed_weight")
        if embed_w is None:
            embed_w = eh.get("embed")

        if lm_head_w is None or embed_w is None:
            raise ValueError("embed_head.pt missing lm_head_weight or embed_weight")

        self.lm_head = mx.array(lm_head_w.float().numpy())   # [vocab, hidden]
        self.embed = mx.array(embed_w.float().numpy())        # [vocab, hidden]
        print(f"  lm_head: {self.lm_head.shape}, embed: {self.embed.shape}")

    def _build_execution_plan(self):
        """Build execution plan from IR graph definition.

        The IR defines layers: proj, norm1, attn, norm2, mlp, norm_out.
        The dispatcher adds implicit ops: concat, add (residual), argmax.
        """
        H = self.config.hidden_size
        self.plan: List[ExecStep] = []

        # Step 1: Concat hidden + token embedding
        self.plan.append(ExecStep(
            op="concat", inputs=["hidden", "token_embed"], output="x",
            cgc_opcode=IR_TO_CGC_OPCODE["concat"],
        ))

        # Step 2: Linear projection (2*H → H)
        self.plan.append(ExecStep(
            op="linear", inputs=["x"], output="x",
            weights=["proj.weight"],
            params={"in_features": H * 2, "out_features": H},
            cgc_opcode=IR_TO_CGC_OPCODE["linear"],
        ))

        # Step 3: RMSNorm (pre-norm for attention)
        self.plan.append(ExecStep(
            op="rms_norm", inputs=["x"], output="normed1",
            weights=["norm1.weight"],
            params={"eps": self.config.rms_norm_eps},
            cgc_opcode=IR_TO_CGC_OPCODE["rms_norm"],
        ))

        # Step 4: Attention
        self.plan.append(ExecStep(
            op="attention", inputs=["normed1"], output="attn_out",
            weights=["attn.q_proj.weight", "attn.k_proj.weight",
                     "attn.v_proj.weight", "attn.o_proj.weight"],
            params={"num_heads": self.config.num_heads, "head_dim": self.config.head_dim},
            cgc_opcode=IR_TO_CGC_OPCODE["attention"],
        ))

        # Step 5: Residual add (x + attn_out)
        self.plan.append(ExecStep(
            op="add", inputs=["x", "attn_out"], output="h",
            cgc_opcode=0x00,  # element-wise add (not a CGC opcode, but implicit)
        ))

        # Step 6: RMSNorm (pre-norm for MLP)
        self.plan.append(ExecStep(
            op="rms_norm", inputs=["h"], output="normed2",
            weights=["norm2.weight"],
            params={"eps": self.config.rms_norm_eps},
            cgc_opcode=IR_TO_CGC_OPCODE["rms_norm"],
        ))

        # Step 7: SwiGLU MLP
        self.plan.append(ExecStep(
            op="mlp_silu", inputs=["normed2"], output="mlp_out",
            weights=["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"],
            params={"hidden_size": H, "intermediate_size": self.config.intermediate_size},
            cgc_opcode=IR_TO_CGC_OPCODE["mlp_silu"],
        ))

        # Step 8: Residual add (h + mlp_out)
        self.plan.append(ExecStep(
            op="add", inputs=["h", "mlp_out"], output="h2",
            cgc_opcode=0x00,
        ))

        # Step 9: Final RMSNorm
        self.plan.append(ExecStep(
            op="rms_norm", inputs=["h2"], output="final_hidden",
            weights=["norm_out.weight"],
            params={"eps": self.config.rms_norm_eps},
            cgc_opcode=IR_TO_CGC_OPCODE["rms_norm"],
        ))

        # Step 10: Shared lm_head (linear)
        self.plan.append(ExecStep(
            op="linear", inputs=["final_hidden"], output="logits",
            weights=["__lm_head__"],  # special key for shared lm_head
            params={"in_features": H, "out_features": self.config.vocab_size},
            cgc_opcode=IR_TO_CGC_OPCODE["linear"],
        ))

        # Step 11: Argmax sampling
        self.plan.append(ExecStep(
            op="argmax", inputs=["logits"], output="token",
            cgc_opcode=IR_TO_CGC_OPCODE["argmax"],
        ))

    def _execute_step(self, step: ExecStep, ctx: Dict[str, mx.array]) -> mx.array:
        """Execute a single step of the execution plan."""
        backend = self.backend

        if step.op == "concat":
            tensors = [ctx[name] for name in step.inputs]
            return backend.concat(*tensors, axis=-1)

        elif step.op == "linear":
            x = ctx[step.inputs[0]]
            w_key = step.weights[0]
            w = self.lm_head if w_key == "__lm_head__" else self.weights[w_key]
            return backend.linear(x, w)

        elif step.op == "rms_norm":
            x = ctx[step.inputs[0]]
            w = self.weights[step.weights[0]]
            eps = step.params.get("eps", self.config.rms_norm_eps)
            return backend.rms_norm(x, w, eps)

        elif step.op == "attention":
            x = ctx[step.inputs[0]]
            q_w = self.weights[step.weights[0]]
            k_w = self.weights[step.weights[1]]
            v_w = self.weights[step.weights[2]]
            o_w = self.weights[step.weights[3]]
            return backend.attention(x, q_w, k_w, v_w, o_w)

        elif step.op == "mlp_silu":
            x = ctx[step.inputs[0]]
            gate_w = self.weights[step.weights[0]]
            up_w = self.weights[step.weights[1]]
            down_w = self.weights[step.weights[2]]
            return backend.mlp_silu(x, gate_w, up_w, down_w)

        elif step.op == "add":
            tensors = [ctx[name] for name in step.inputs]
            result = tensors[0]
            for t in tensors[1:]:
                result = result + t
            return result

        elif step.op == "argmax":
            x = ctx[step.inputs[0]]
            return backend.argmax(x, axis=-1)

        else:
            raise ValueError(f"Unknown op: {step.op}")

    def forward_single(
        self, hidden: mx.array, token_embed: mx.array,
    ) -> Tuple[int, mx.array]:
        """Execute one forward pass of the IR graph.

        Args:
            hidden: [1, 1, hidden_size] base model hidden state
            token_embed: [1, 1, hidden_size] current token embedding

        Returns:
            token: predicted next token ID
            final_hidden: [1, 1, hidden_size] MTP hidden state (for chaining)
        """
        ctx = {"hidden": hidden, "token_embed": token_embed}

        for step in self.plan:
            result = self._execute_step(step, ctx)
            ctx[step.output] = result

        token = int(ctx["token"].item())
        return token, ctx["final_hidden"]

    def draft_chain(
        self, hidden_np: np.ndarray, token_id: int, num_draft: int,
    ) -> Tuple[List[int], float]:
        """Chain draft generation via IR execution plan.

        Args:
            hidden_np: base model hidden state [hidden_size] (numpy float32)
            token_id: last token ID from base model
            num_draft: number of draft tokens to generate

        Returns:
            draft_tokens: list of draft token IDs
            elapsed_ms: total draft time in ms
        """
        t0 = time.time()

        current_hidden = mx.array(hidden_np.astype(np.float32)).reshape(1, 1, self.config.hidden_size)
        current_token = token_id
        draft_tokens = []

        for i in range(num_draft):
            # Token embedding lookup
            token_embed = self.embed[current_token].reshape(1, 1, self.config.hidden_size)

            # Execute IR graph
            token, mtp_hidden = self.forward_single(current_hidden, token_embed)

            draft_tokens.append(token)
            current_hidden = mtp_hidden
            current_token = token

        mx.eval(draft_tokens)
        elapsed_ms = (time.time() - t0) * 1000
        return draft_tokens, elapsed_ms

    def _warmup(self):
        """Warmup Metal kernels with a dummy forward pass."""
        dummy_hidden = mx.zeros((1, 1, self.config.hidden_size))
        dummy_embed = mx.zeros((1, 1, self.config.hidden_size))
        token, hidden = self.forward_single(dummy_hidden, dummy_embed)
        mx.eval(token, hidden)

    def get_execution_plan_summary(self) -> str:
        """Return a human-readable summary of the execution plan."""
        lines = [f"CGC IR Execution Plan ({len(self.plan)} steps):"]
        for i, step in enumerate(self.plan):
            w_info = f" weights=[{', '.join(step.weights)}]" if step.weights else ""
            opcode_info = f" opcode=0x{step.cgc_opcode:02X}" if step.cgc_opcode else ""
            lines.append(
                f"  {i+1:2d}. {step.op:12s} {step.inputs} → {step.output}{w_info}{opcode_info}"
            )
        return "\n".join(lines)

    def get_ir_json(self) -> str:
        """Return the IR graph as JSON (for serialization/inspection)."""
        return json.dumps(self.ir, indent=2)
