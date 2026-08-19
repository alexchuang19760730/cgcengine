#!/usr/bin/env python3
"""Quick 64-token ppl sanity check for qwen36-r3q4la_ga_e4 with optional
Q36_PRUNE_LAYERS (zero-training layer pruning, mirrors engine semantics).

Differences vs the r4 fp16 script:
  - dense weights are q4-packed (dtype=0) with scaleStart/biasStart metadata
  - experts are 3-bit (BITS=3)
  - Q36_PRUNE_LAYERS="25,24,21,20,5,9,13,17,22,4" skips those layers
    (engine semantics: skip layer entirely; moes indexed by absolute L)
"""
import os, sys, json, time, math, gc
sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack")
from pathlib import Path
from resident_writer import read_resident_bin as _read_resident
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.activations import ACT2FN
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

GTURBO = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r3q4la_ga_e4.gturbo"
HF = "/Volumes/AlexZhuang/qwen36-hf"
BITS = 3
LIMIT = int(os.environ.get("PPL_LIMIT", "64"))
PRUNE = [int(x) for x in os.environ.get("Q36_PRUNE_LAYERS", "").split(",") if x]
PRUNE_SET = set(PRUNE)

def log(m): print(m, flush=True)

def bf16_to_float(bits):
    # fully vectorized: bf16 bit pattern (u16) -> float32
    b = np.asarray(bits, dtype=np.uint32)
    return (b << 16).view(np.float32)

def dequant_rows(packed, scales, biases, bits, cols):
    """Dequant packed rows [rows, cols*bits/8] with per-group scales/biases (bf16 bit patterns).
    Fully vectorized (matches repack_qwen36.quantize_affine roundtrip)."""
    rows = packed.shape[0]
    n_groups = cols // 64
    sc = np.repeat(bf16_to_float(scales.reshape(-1)), 64).reshape(rows, cols)
    bs = np.repeat(bf16_to_float(biases.reshape(-1)), 64).reshape(rows, cols)
    if bits == 4:
        p = packed.astype(np.uint16).reshape(rows, cols // 2)
        lo = p & 0x0F
        hi = p >> 4
        q = np.empty((rows, cols), dtype=np.uint8)
        q[:, 0::2] = lo
        q[:, 1::2] = hi
    else:  # 3-bit LSB-first, 8 vals -> 3 bytes
        p = packed.astype(np.uint16).reshape(rows, cols // 8, 3)
        b0, b1, b2 = p[..., 0], p[..., 1], p[..., 2]
        v0 = b0 & 0x7
        v1 = (b0 >> 3) & 0x7
        v2 = (b0 >> 6) | ((b1 & 0x1) << 2)
        v3 = (b1 >> 1) & 0x7
        v4 = (b1 >> 4) & 0x7
        v5 = (b1 >> 7) | ((b2 & 0x3) << 1)
        v6 = (b2 >> 2) & 0x7
        v7 = (b2 >> 5) & 0x7
        q = np.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=-1).reshape(rows, cols).astype(np.uint8)
    return (q.astype(np.float32) * sc + bs)

layout = json.load(open(f"{GTURBO}/packed_experts/layout.json"))
num_layers = layout["numLayers"]
n_experts = layout["expertsPerLayer"]
ACTIVE = [L for L in range(num_layers) if L not in PRUNE_SET]

info = _read_resident(Path(GTURBO) / "model_weights.bin")

def load_dense_state_dict():
    """Load dense (non-expert) weights; dequant dtype=0 (q4) entries. Skip pruned layers."""
    sd = {}
    with open(os.path.join(GTURBO, "model_weights.bin"), "rb") as f:
        for name, e in info["entries"].items():
            parts = name.split(".")
            # layer-scoped? skip if in pruned set
            if "layers" in parts:
                li = parts.index("layers")
                L = int(parts[li + 1])
                if L in PRUNE_SET:
                    continue
            f.seek(e["fileOffset"])
            blob = f.read(e["sizeBytes"])
            shp = [s for s in e["shape"] if s != 0]
            key = name.replace("model.language_model.", "model.", 1)
            if e["dtype"] == 1:  # fp16
                arr = np.frombuffer(blob, dtype=np.float16).reshape(shp).astype(np.float32)
                sd[key] = torch.from_numpy(arr).to(torch.bfloat16)
            elif e["dtype"] == 2:  # fp32
                arr = np.frombuffer(blob, dtype=np.float32).reshape(shp)
                sd[key] = torch.from_numpy(arr).to(torch.bfloat16)
            elif e["dtype"] == 0 and e.get("scaleSize", 0) > 0:  # q4 packed
                packed = np.frombuffer(blob, dtype=np.uint8)
                # NOTE: scaleOffset/biasOffset are absolute offsets in the file,
                # while `blob` starts at e["fileOffset"] -> subtract fileOffset.
                fo = e["fileOffset"]
                ss, sb = e["scaleOffset"] - fo, e["scaleSize"]
                bs_, bb = e["biasOffset"] - fo, e["biasSize"]
                scales = np.frombuffer(blob[ss:ss + sb], dtype=np.uint16)
                biases = np.frombuffer(blob[bs_:bs_ + bb], dtype=np.uint16)
                # packed layout: [rows, cols*bits/8]
                cols = shp[1]
                rows = shp[0]
                n_pack = cols * 4 // 8  # q4: cols/2 bytes per row
                packed = packed[: rows * n_pack].reshape(rows, n_pack)
                dec = dequant_rows(packed, scales, biases, 4, cols)
                sd[key] = torch.from_numpy(dec).to(torch.bfloat16)
            else:  # dtype 0 without scale metadata (u32 misc) — skip
                pass
    return sd

# ---- expert streaming (3-bit) ----
_layer_counter = [0]
class LocalStreamingQwenExperts(torch.nn.Module):
    _cache = {}
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_dim = config.moe_intermediate_size
        self.hidden_dim = config.hidden_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = _layer_counter[0]
        _layer_counter[0] += 1
    def _expert_tensors(self, e):
        lay = layout["layers"][self.layer_idx]
        ent = lay["experts"][e]
        t = ent["tensors"]
        with open(os.path.join(GTURBO, "packed_experts", lay["file"]), "rb") as f:
            def read(role):
                ri = t[role]
                f.seek(ent["offset"] + ri["offset"])
                raw = f.read(ri["size"])
                if ri["bits"] == 16:
                    return np.frombuffer(raw, dtype=np.uint16).reshape(ri["shape"])
                return raw
            gate_w = read("gate"); gate_s = read("gate_scales"); gate_b = read("gate_biases")
            up_w = read("up"); up_s = read("up_scales"); up_b = read("up_biases")
            down_w = read("down"); down_s = read("down_scales"); down_b = read("down_biases")
        cols = self.hidden_dim
        gate = dequant_rows(np.frombuffer(gate_w, dtype=np.uint8).reshape(-1, cols * BITS // 8), gate_s, gate_b, BITS, cols).astype(np.float32)
        up = dequant_rows(np.frombuffer(up_w, dtype=np.uint8).reshape(-1, cols * BITS // 8), up_s, up_b, BITS, cols).astype(np.float32)
        ic = self.intermediate_dim
        down = dequant_rows(np.frombuffer(down_w, dtype=np.uint8).reshape(-1, ic * BITS // 8), down_s, down_b, BITS, ic).astype(np.float32)
        return (gate, up, down)
    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()
        for row in expert_hit:
            e = row[0].item()
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            w = top_k_weights[token_idx, top_k_pos, None]
            gate, up, down = self._expert_tensors(e)
            g = torch.from_numpy(gate).to(hidden_states.device, hidden_states.dtype)
            u = torch.from_numpy(up).to(hidden_states.device, hidden_states.dtype)
            d = torch.from_numpy(down).to(hidden_states.device, hidden_states.dtype)
            gate_o = F.linear(hidden_states[token_idx], g)
            up_o = F.linear(hidden_states[token_idx], u)
            h = self.act_fn(gate_o) * up_o
            out = F.linear(h, d) * w
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final

class _EmptyExperts(torch.nn.Module):
    """Placeholder: dense params omitted (streamed). State dict loads nothing for experts."""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_up_proj = torch.nn.Parameter(torch.empty(1, 1, 1))
        self.down_proj = torch.nn.Parameter(torch.empty(1, 1, 1))

M.Qwen3_5MoeExperts = _EmptyExperts
def _patched_init_weights(self, module):
    if isinstance(module, _EmptyExperts):
        return
    return _orig_init_weights(self, module)
_orig_init_weights = M.Qwen3_5MoePreTrainedModel._init_weights
M.Qwen3_5MoePreTrainedModel._init_weights = _patched_init_weights
# routing path (used by the model forward) routes through Qwen3_5MoeExperts;
# the streaming implementation is installed AFTER model construction below.

log(f"building model ({num_layers} layers, {n_experts} experts, {BITS}-bit, prune={sorted(PRUNE_SET)})...")
cfg = AutoConfig.from_pretrained(HF)
text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
# Build in bf16 (halves dense init footprint; state dict is bf16 anyway)
torch.set_default_dtype(torch.bfloat16)
model = M.Qwen3_5MoeForCausalLM(text_cfg)
model.eval()

t1 = time.time()
sd = load_dense_state_dict()
missing = model.load_state_dict(sd, strict=False)
log(f"dense loaded ({time.time()-t1:.0f}s); missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
del sd; gc.collect()

# Install streaming expert forward on every expert module (keeps _EmptyExperts params)
# layer_idx = absolute layer index (engine semantics: moes[L] indexed by absolute L)
for li, layer in enumerate(model.model.layers):
    layer.layer_idx = li
    for name, mod in layer.named_modules():
        if isinstance(mod, _EmptyExperts):
            mod.__class__ = LocalStreamingQwenExperts
            mod.layer_idx = li

GPU_IDS = [248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198, 248045, 74455, 198]

def _run(model, ids):
    t0 = time.time()
    with torch.no_grad():
        out = model(ids)
    lg = out.logits[0, :-1]
    labels = ids[0, 1:]
    n = labels.shape[0]
    ssum = 0.0
    for i in range(0, n, 128):
        ssum += F.cross_entropy(lg[i:i + 128].float(), labels[i:i + 128]).item() * min(128, n - i)
    nll = ssum / n
    ppl = math.exp(nll)
    log(f"quick ppl: nll={nll:.4f} ppl={ppl:.3f} ({time.time()-t0:.0f}s)")
    log(f"logits finite={torch.isfinite(lg).all().item()} | range [{lg.min().item():.2f},{lg.max().item():.2f}]")
    top1 = lg.argmax(-1)
    log(f"top1-acc vs labels: {(top1 == labels).float().mean().item():.3f} ({n} tokens)")
    lg_last = out.logits[0, -1]
    top5 = torch.topk(lg_last, 5)
    log("CPU logits top5: " + " ".join(f"{idx.item()}:{val.item():.2f}" for idx, val in zip(top5.indices, top5.values)))
    log("=== SANITY DONE ===")

# Engine semantics: pruned layers are skipped entirely (residual passes through).
if PRUNE_SET:
    _orig_layer_forward = M.Qwen3_5MoeDecoderLayer.forward
    def _pruned_layer_forward(self, hidden_states, *args, **kwargs):
        if self.layer_idx in PRUNE_SET:
            return hidden_states
        return _orig_layer_forward(self, hidden_states, *args, **kwargs)
    M.Qwen3_5MoeDecoderLayer.forward = _pruned_layer_forward

def main():
    text = os.environ.get("PPL_TEXT", "")
    if text:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(GTURBO, "tokenizer"))
        ids = torch.tensor([tok.encode(text)], dtype=torch.long)
        log(f"corpus: {ids.shape[1]} tokens (tokenizer)")
    else:
        ids = torch.tensor([GPU_IDS], dtype=torch.long)
        log(f"gpu ids: {ids.shape[1]} tokens")
    _run(model, ids)

if __name__ == "__main__":
    main()
