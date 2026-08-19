#!/usr/bin/env python3
"""linear_attn_sensitivity.py — DeltaNet linear_attn 量化敏感度 profile。

在 Python 中逐 token 复刻引擎 (deltanet.metal) 的 fp16 管线:
  project(GEMV fp32 acc → fp16) → conv1d(silu, fp16 状态滚动) →
  prepare(q/k/v fp16, g/beta fp32) → delta_rule(h fp32) → output(fp16 y)

1. 先用 layer 0 + inputs_layer0.bin 对拍 metal_layer0_y/h.bin (引擎 fp16 实跑),
   验证 sim 与引擎一致 (fp16 舍入级误差).
2. 然后对每个 deltanet 层 (0..38), 对 {qkv, z, out} × {3, 4, 8 bit} 分别量化,
   跑 50 token, 测 y/h 相对误差 (vs 自身 fp16 sim) → 敏感度表, 决定各 tensor
   用 4-bit 还是 3-bit.

用法:
  python3 linear_attn_sensitivity.py [--validate] [--layers 0,3,7] [--quick]
"""
import sys, struct
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "prime-agent-worktrees/qwen36-repack"))
from resident_writer import read_resident_bin
from repack_qwen36 import quantize_affine

TEST = Path("/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/models/qwen36-test")
META = TEST / "metal_out"

EPS = 1e-6
SCALE = 1.0 / np.sqrt(128.0)


def load_fp16(path, shape):
    d = np.fromfile(path, dtype=np.float16)
    return d.reshape(shape)


def load_fp32(path, shape):
    d = np.fromfile(path, dtype=np.float32)
    return d.reshape(shape)


class LayerWeights:
    def __init__(self, resident, raw, layer):
        base = f"model.language_model.layers.{layer}.linear_attn"
        def get(name, shape):
            e = resident["entries"][f"{base}.{name}"]
            return np.frombuffer(raw[e["fileOffset"]:e["fileOffset"] + e["sizeBytes"]],
                                 dtype=np.float16).reshape(shape).copy()
        self.W_qkv = get("in_proj_qkv.weight", (8192, 2048))
        self.W_z = get("in_proj_z.weight", (4096, 2048))
        self.W_b = get("in_proj_b.weight", (32, 2048))
        self.W_a = get("in_proj_a.weight", (32, 2048))
        self.W_out = get("out_proj.weight", (2048, 4096))
        self.conv_w = get("conv1d.weight", (8192, 1, 4))
        self.A_log = get("A_log", (32,))
        self.dt_bias = get("dt_bias", (32,))
        self.norm = get("norm.weight", (128,))


def dequant_q(t, bits):
    """4/3-bit affine quant → fp32 dequant (与 quantize_affine/引擎 dequant 同款)."""
    q, s, b = quantize_affine(t.astype(np.float32), bits=bits, group_size=64)
    s = (s.astype(np.uint32) << 16).view(np.float32)
    b = (b.astype(np.uint32) << 16).view(np.float32)
    rows, cols = t.shape
    ng = cols // 64
    qv = np.zeros((rows, cols), dtype=np.float32)
    if bits == 4:
        # quantize_affine: q4 = qf.reshape(rows, ng, 32, 2); packed byte k (per
        # row) = lo|hi<<4, k = g*32+p → col 2p (lo), col 2p+1 (hi).
        q2 = q.reshape(rows, ng, 32)
        lo = (q2 & 0x0F).astype(np.float32).reshape(rows, cols // 2)
        hi = ((q2 >> 4) & 0x0F).astype(np.float32).reshape(rows, cols // 2)
        qv[:, 0::2] = lo
        qv[:, 1::2] = hi
    else:  # 3-bit — packed [rows, cols*3/8] u8, 每 8 值 3 bytes LSB-first
        q3 = q.reshape(rows, cols // 8, 3).astype(np.uint32)
        window = q3[..., 0] | (q3[..., 1] << 8) | (q3[..., 2] << 16)  # [rows, cols/8]
        # qv[:, c] = (window[:, c//8] >> ((c%8)*3)) & 0x7
        qv = ((window[:, np.arange(cols) // 8] >> ((np.arange(cols) % 8) * 3)) & 0x7).astype(np.float32)
    return qv * s[:, np.arange(cols) // 64] + b[:, np.arange(cols) // 64]


def run_layer(w: LayerWeights, x_all, qcfg=None):
    """x_all: [T, 2048] fp16 → y_all [T, 2048] fp16, h_all [T,32,128,128] fp32."""
    if qcfg is None:
        qcfg = {}
    W_qkv = dequant_q(w.W_qkv, qcfg["qkv"]) if "qkv" in qcfg else w.W_qkv.astype(np.float32)
    W_z = dequant_q(w.W_z, qcfg["z"]) if "z" in qcfg else w.W_z.astype(np.float32)
    W_out = dequant_q(w.W_out, qcfg["out"]) if "out" in qcfg else w.W_out.astype(np.float32)
    W_b = w.W_b.astype(np.float32)
    W_a = w.W_a.astype(np.float32)

    T = x_all.shape[0]
    conv_state = np.zeros((8192, 4), dtype=np.float16)
    h = np.zeros((32, 128, 128), dtype=np.float32)
    y_all = np.zeros((T, 2048), dtype=np.float16)
    h_all = np.zeros((T, 32, 128, 128), dtype=np.float32)
    for t in range(T):
        x = x_all[t].astype(np.float32)
        # 1. project (fp32 acc → fp16)
        qkv = (W_qkv @ x).astype(np.float16)     # [8192]
        z = (W_z @ x).astype(np.float16)         # [4096]
        b = (W_b @ x).astype(np.float16)         # [32]
        a = (W_a @ x).astype(np.float16)         # [32]
        # 2. conv1d + silu
        cw = w.conv_w.astype(np.float32).reshape(8192, 4)  # [C,4]
        cs = conv_state.astype(np.float32)
        acc = cw[:, 0] * cs[:, 0] + cw[:, 1] * cs[:, 1] + cw[:, 2] * cs[:, 2] + cw[:, 3] * qkv.astype(np.float32)
        qkv_post = (acc / (1.0 + np.exp(-acc))).astype(np.float16)
        # roll state: [s1, s0, x, 0]
        conv_state = np.stack([cs[:, 1], cs[:, 2], qkv.astype(np.float32), np.zeros(8192)], axis=1).astype(np.float16)
        # 3. prepare
        av = a.astype(np.float32)
        bv = b.astype(np.float32)
        A = np.exp(w.A_log.astype(np.float32))            # [32]
        sp = np.maximum(av + w.dt_bias.astype(np.float32), 0) + np.log1p(np.exp(-np.abs(av + w.dt_bias.astype(np.float32))))
        g = -A * sp                                        # [32]
        beta = 1.0 / (1.0 + np.exp(-bv))                   # [32]
        qp = qkv_post.astype(np.float32)
        q_k = np.zeros((32, 128), dtype=np.float16)
        k_k = np.zeros((32, 128), dtype=np.float16)
        v_k = np.zeros((32, 128), dtype=np.float16)
        for vh in range(32):
            ks = vh // 2
            qsrc = ks * 128
            ksrc = 2048 + ks * 128
            vsrc = 4096 + vh * 128
            q16 = qp[qsrc:qsrc + 128]
            k16 = qp[ksrc:ksrc + 128]
            # kernel: q_norm = rsqrt(max(q_sum,eps)) * kDeltaScale; q_k = qkv_post*q_norm
            # (k 无 scale — 只有 l2norm)
            qn = np.sqrt(1.0 / np.maximum((q16 * q16).sum(), EPS)) * SCALE
            kn = np.sqrt(1.0 / np.maximum((k16 * k16).sum(), EPS))
            q_k[vh] = (q16 * qn).astype(np.float16)
            k_k[vh] = (k16 * kn).astype(np.float16)
            v_k[vh] = qkv_post[vsrc:vsrc + 128]
        # 4. delta rule
        qf = q_k.astype(np.float32)
        kf = k_k.astype(np.float32)
        vf = v_k.astype(np.float32)
        gv = np.exp(g)
        o = np.zeros((32, 128), dtype=np.float32)
        for vh in range(32):
            hh = h[vh]
            hh = hh * gv[vh]
            kv_mem = hh.T @ kf[vh]           # [128]
            delta = (vf[vh] - kv_mem) * beta[vh]
            hh += np.outer(kf[vh], delta)
            h[vh] = hh
            o[vh] = hh.T @ qf[vh]
        # 5. output
        nw = w.norm.astype(np.float32)       # [128]
        zf = z.reshape(32, 128).astype(np.float32)
        o_normed = np.zeros((32, 128), dtype=np.float16)
        for vh in range(32):
            v = o[vh]
            mean_sq = (v * v).mean()
            rms = v * np.sqrt(1.0 / (mean_sq + EPS))
            gated = rms * nw * (zf[vh] / (1.0 + np.exp(-zf[vh])))
            o_normed[vh] = gated.astype(np.float16)
        y = (W_out @ o_normed.astype(np.float32).reshape(-1)).astype(np.float16)
        y_all[t] = y
        h_all[t] = h
    return y_all, h_all


def main():
    args = [a for a in sys.argv[1:]]
    validate_only = "--validate" in args
    quick = "--quick" in args
    layers_arg = None
    if "--layers" in args:
        layers_arg = [int(x) for x in args[args.index("--layers") + 1].split(",")]

    resident = read_resident_bin(TEST / "model_weights.bin")
    raw = (TEST / "model_weights.bin").read_bytes()
    x_all = load_fp16(META / "inputs_layer0.bin", (50, 2048))

    # ---- 0. validate layer 0 sim vs Metal fp16 ----
    w0 = LayerWeights(resident, raw, 0)
    y_sim, h_sim = run_layer(w0, x_all)
    y_metal = load_fp16(META / "metal_layer0_y.bin", (50, 2048))
    h_metal = load_fp32(META / "metal_layer0_h.bin", (50, 32, 128, 128))
    dy = np.abs(y_sim.astype(np.float32) - y_metal.astype(np.float32)).max()
    dh = np.abs(h_sim - h_metal).max()
    print(f"[validate] layer0 sim vs Metal fp16: max|dy|={dy:.6f} max|dh|={dh:.6f}")
    print(f"[validate] h magnitude: mean|h|={np.abs(h_metal).mean():.4f} max|h|={np.abs(h_metal).max():.4f}")
    if validate_only:
        ok = dy < 0.02 and dh < 1e-3
        print("VALIDATE", "PASS" if ok else "FAIL")
        return 0

    # ---- 1. per-tensor sensitivity on layer 0 ----
    print("\n=== layer 0: per-tensor quantization sensitivity (50 tok, vs fp16 sim) ===")
    y_ref, h_ref = y_sim, h_sim
    cfgs = [
        ("qkv@4",  {"qkv": 4}), ("qkv@3", {"qkv": 3}),
        ("z@4",    {"z": 4}),   ("z@3",   {"z": 3}),
        ("out@4",  {"out": 4}), ("out@3", {"out": 3}),
        ("qkv@4+z@4+out@4", {"qkv": 4, "z": 4, "out": 4}),
        ("qkv@3+z@4+out@4", {"qkv": 3, "z": 4, "out": 4}),
        ("z@4+out@4",       {"z": 4, "out": 4}),
        ("z@3+out@4",       {"z": 3, "out": 4}),
    ]
    print(f"{'config':<18}{'y rel err':>12}{'y max abs':>12}{'h rel err':>12}")
    for name, cfg in cfgs:
        y_q, h_q = run_layer(w0, x_all, cfg)
        yrel = np.abs(y_q.astype(np.float32) - y_ref.astype(np.float32)).mean() / (np.abs(y_ref.astype(np.float32)).mean() + 1e-6)
        ymax = np.abs(y_q.astype(np.float32) - y_ref.astype(np.float32)).max()
        hrel = np.abs(h_q - h_ref).mean() / (np.abs(h_ref).mean() + 1e-6)
        print(f"{name:<18}{yrel:>12.3%}{ymax:>12.5f}{hrel:>12.3%}")

    # ---- 2. per-layer profile (all deltanet layers) ----
    layers = layers_arg if layers_arg else ([0, 3, 7, 15, 27, 38] if quick else range(0, 40, 4))
    print(f"\n=== per-layer sensitivity ({len(layers)} layers, same x, 50 tok) ===")
    print(f"{'layer':<6}{'qkv@4 y%':>10}{'qkv@3 y%':>10}{'z@4 y%':>10}{'out@4 y%':>10}{'all@4 y%':>10}")
    rows = []
    for L in layers:
        if L % 4 == 3:
            continue  # gatedattn
        w = LayerWeights(resident, raw, L)
        y_ref, h_ref = run_layer(w, x_all)
        def yrel(cfg):
            y_q, _ = run_layer(w, x_all, cfg)
            return np.abs(y_q.astype(np.float32) - y_ref.astype(np.float32)).mean() / (np.abs(y_ref.astype(np.float32)).mean() + 1e-6)
        q4 = yrel({"qkv": 4}); q3 = yrel({"qkv": 3})
        z4 = yrel({"z": 4}); out4 = yrel({"out": 4}); all4 = yrel({"qkv": 4, "z": 4, "out": 4})
        rows.append((L, q4, q3, z4, out4, all4))
        print(f"{L:<6}{q4:>10.2%}{q3:>10.2%}{z4:>10.2%}{out4:>10.2%}{all4:>10.2%}")
    arr = np.array(rows)
    if len(arr):
        print("\n=== aggregate (mean over layers) ===")
        print(f"qkv@4 y%: {arr[:,1].mean():.2%}  qkv@3 y%: {arr[:,2].mean():.2%}  "
              f"z@4: {arr[:,3].mean():.2%}  out@4: {arr[:,4].mean():.2%}  all@4: {arr[:,5].mean():.2%}")


if __name__ == "__main__":
    main()
