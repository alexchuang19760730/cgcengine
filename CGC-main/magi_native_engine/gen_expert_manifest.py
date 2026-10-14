#!/usr/bin/env python3
"""
gen_expert_manifest.py — 生成 P2 四接口桩的专家页清单（GGUF 实测核准输入）

读取 Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf 的 tensor 布局，
把每层 ffn_{gate,up,down}_exps 按文件顺序打包成 ~1.7MiB 页（对齐 P1a 实测
带宽粒度：随机 4 线程 6.2 GB/s），输出 JSON manifest 供 moe_inject.c 桩读取。

用法：
  /opt/homebrew/bin/python3.13 gen_expert_manifest.py \
      ~/Documents/flashkv-devserver/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf \
      --out expert_manifest.json

口径：tensor 的 data_offset / n_bytes 来自 gguf 库（与 gguf_verified.json 同源）；
专家维 = shape[-1]；每专家字节 = n_bytes / n_experts（含量化开销，整数向上取整）。
"""
import argparse, json, sys, os
sys.path.insert(0, "/opt/homebrew/lib/python3.13/site-packages")
from gguf import GGUFReader

PAGE = 1782576  # 1.7 MiB，P1a 实测粒度
ALIGN = 4096

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--out", default="expert_manifest.json")
    ap.add_argument("--page", type=int, default=PAGE)
    args = ap.parse_args()

    r = GGUFReader(args.gguf)
    layers = {}
    expert_sizes = {}   # (layer, matrix) -> per-expert bytes
    for t in r.tensors:
        name = t.name.decode() if isinstance(t.name, bytes) else str(t.name)
        # blk.<L>.ffn_<m>_exps.weight
        parts = name.split(".")
        if len(parts) < 4 or not parts[1].isdigit() or not parts[2].startswith("ffn_") or not parts[2].endswith("_exps"):
            continue
        layer, matrix = int(parts[1]), parts[2].replace("ffn_", "").replace("_exps", "")
        n_exp = t.shape[-1]
        per_exp = (t.n_bytes + n_exp - 1) // n_exp
        layers.setdefault(layer, {})[matrix] = {
            "offset": int(t.data_offset),
            "n_bytes": int(t.n_bytes),
            "n_experts": int(n_exp),
            "per_expert": int(per_exp),
            "tensor_type": int(t.tensor_type),
        }
        expert_sizes[(layer, matrix)] = per_exp

    L = sorted(layers)
    # 页打包：每矩阵内按文件顺序连续切页（页内可能跨专家边界，页粒度 = 1.7MiB）
    pages = []
    for layer in L:
        for matrix in ("gate", "up", "down"):
            e = layers[layer][matrix]
            off, size = e["offset"], e["n_bytes"]
            n = 0
            while size > 0:
                chunk = min(size, args.page)
                # 页起点对齐 4KiB
                poff = (off + ALIGN - 1) // ALIGN * ALIGN
                pages.append({"layer": layer, "matrix": matrix, "page_idx": n,
                              "offset": poff, "size": chunk,
                              "n_experts": max(1, int(chunk / e["per_expert"]))})
                off += chunk; size -= chunk; n += 1

    manifest = {
        "model": os.path.basename(args.gguf),
        "L": L[-1] + 1 if L else 0,
        "experts_per_layer": layers[L[0]]["gate"]["n_experts"] if L else 0,
        "top_k": 8,
        "expert_bytes_MB": round(layers[L[0]]["gate"]["per_expert"] / 1e6, 4) if L else 0,
        "page_bytes": args.page,
        "pages": pages,
    }
    # 每层页数（gate+up+down）
    manifest["pages_per_layer"] = {}
    for layer in L:
        manifest["pages_per_layer"][str(layer)] = sum(
            1 for pg in pages if pg["layer"] == layer)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    # 二进制版（C 桩 moe_pool_create 直接 fread，packed 32B 与 #pragma pack(1) 对齐）：
    #   moe_manifest_hdr_t {int n_layer, experts_per_layer, n_pages, page_bytes, expert_bytes}  = 20B
    #   moe_page_t         {int layer, matrix, page_idx; int64 offset, size; int n_experts}     = 32B '<iiiqqi'
    import struct
    bin_out = args.out.rsplit(".", 1)[0] + ".bin" if "." in args.out else args.out + ".bin"
    with open(bin_out, "wb") as f:
        f.write(struct.pack("<iiiii", manifest["L"], manifest["experts_per_layer"],
                            len(pages), args.page,
                            layers[L[0]]["gate"]["per_expert"] if L else 0))
        for pg in pages:
            f.write(struct.pack("<iiiqqi", pg["layer"], 0 if pg["matrix"] == "gate" else (1 if pg["matrix"] == "up" else 2),
                                pg["page_idx"], pg["offset"], pg["size"], pg["n_experts"]))
    n_pages = len(pages)
    total_bytes = sum(layers[l][m]["n_bytes"] for l in layers for m in layers[l])
    total_experts = manifest["L"] * manifest["experts_per_layer"] if manifest["L"] else 0
    print("manifest 已写 %s" % args.out)
    print("  L=%d · 每层专家=%d · 专家页=%d · 页大小=%d B · 专家总字节=%.2f GB"
          % (manifest["L"], manifest["experts_per_layer"], n_pages, args.page,
             total_bytes / 1e9))
    print("  每层页数样例:", {k: manifest["pages_per_layer"][k] for k in list(manifest["pages_per_layer"])[:3]})

if __name__ == "__main__":
    main()
