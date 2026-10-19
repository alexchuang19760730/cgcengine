#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
l4_slab_geometry.py — C 线阶段 1 只读分析：L4 pool_ext 页/slot 交叠 + 冷页回收粒度。

不接触引擎、不分配 GPU。输入是启动 server 时开 CGC_POOL_SPLIT_DBG=1 后 stderr 里的
两类诊断行（阶段 1 补丁在原 CGC-POOL-SPLIT-ADOPT 之外补了一行 CGC-POOL-SPLIT-GEOM）：

    CGC-POOL-SPLIT-ADOPT: layer=3 kind=0 slots=143 stride=1122335 base=0x... buf=0x... ok=1
    CGC-POOL-SPLIT-GEOM: layer=3 kind=0 page=16384 base_mod_pg=0 stride_mod_pg=8191 \
                         slice_mod_pg=... stride_pg=68.5000

它回答阶段 1 的问题②③（静态可算部分）：
  1. 每个 kind 的 stride 是不是 16KiB 页整数倍；
  2. 每个 layer 切片 base 是不是页对齐、slots*stride 是不是页整数倍；
  3. 以绝对页切分整个 kind 大 Metal buffer，每个 16KiB 页横跨几个 (layer, slot)
     —— 这决定阶段 2 MADV_FREE_REUSABLE 的最小安全回收单元：
       * 一页只覆盖 1 个 slot  -> 该 slot 冷即可回收；
       * 一页覆盖 N 个 slot（含跨 layer 边界页）-> 这些 slot 必须全部无 owner 才能回收。
  4. 打印该把 buffer 切成多大的“回收 slab”（连续 slot 分组，边界恰好页对齐）。

用法：
  python3 l4_slab_geometry.py server_stderr.log          # 解析真实日志
  python3 l4_slab_geometry.py --self-test                # 合成几何自测（无需日志）
"""
import re
import sys
import argparse

PAGE = 16384  # macOS Apple Silicon 16 KiB 页

GEOM_RE = re.compile(
    r"CGC-POOL-SPLIT-GEOM:\s*layer=(\d+)\s+kind=(\d+)\s+page=(\d+)\s+"
    r"base_mod_pg=(\d+)\s+stride_mod_pg=(\d+)\s+slice_mod_pg=(\d+)\s+"
    r"stride_pg=([0-9.]+)")
ADOPT_RE = re.compile(
    r"CGC-POOL-SPLIT-ADOPT:\s*layer=(\d+)\s+kind=(\d+)\s+slots=(\d+)\s+"
    r"stride=(\d+)\s+base=0x([0-9a-fA-F]+)")


def parse(path):
    """返回 {kind: {layer: {slots, stride, base, ...}}}，按 layer 排序。"""
    adopt = {}
    geom = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = ADOPT_RE.search(line)
            if m:
                layer, kind, slots, stride, basehex = m.groups()
                adopt.setdefault(int(kind), {})[int(layer)] = {
                    "slots": int(slots), "stride": int(stride),
                    "base": int(basehex, 16),
                }
                continue
            m = GEOM_RE.search(line)
            if m:
                layer, kind, page, bmod, smod, slicemod, spg = m.groups()
                geom.setdefault(int(kind), {})[int(layer)] = {
                    "page": int(page), "base_mod": int(bmod),
                    "stride_mod": int(smod), "slice_mod": int(slicemod),
                    "stride_pg": float(spg),
                }
    # 合并：以 adopt 为骨架（有 slots/stride/base），geom 补对齐字段
    out = {}
    for kind in sorted(set(adopt) | set(geom)):
        layers = sorted(set(adopt.get(kind, {})) | set(geom.get(kind, {})))
        out[kind] = []
        for layer in layers:
            rec = {"layer": layer}
            if layer in adopt.get(kind, {}):
                rec.update(adopt[kind][layer])
            if layer in geom.get(kind, {}):
                rec.update(geom[kind][layer])
            out[kind].append(rec)
    return out


def analyze_kind(kind, recs, page=PAGE):
    """对一个 kind 做整 buffer 页/slot 交叠分析。"""
    print(f"\n{'='*78}\n kind={kind}   layers={len(recs)}   page={page} B\n{'='*78}")
    if not recs:
        print("  （无数据）")
        return
    # 用绝对 base 排序，重建 layer 顺序；同一 kind 应只在一个大 buffer 里 layer 顺序排列
    recs = sorted(recs, key=lambda r: r["base"])
    base0 = recs[0]["base"]
    strides = {r["stride"] for r in recs if "stride" in r}
    slots_set = {r["slots"] for r in recs if "slots" in r}
    print(f"  buffer 起始 base0 = 0x{base0:x}  (base0 mod page = {base0 % page})")
    print(f"  distinct stride = {sorted(strides)} B")
    print(f"  distinct slots/layer = {sorted(slots_set)}")
    for r in recs[:3]:
        spg = r.get("stride_pg", r["stride"] / page)
        print(f"    layer {r['layer']:>2}: stride={r['stride']} B = {spg:.4f} 页, "
              f"slots={r['slots']}, base_mod={r['base'] % page}, "
              f"slice_mod={(r['slots']*r['stride']) % page}")
    if len(recs) > 3:
        print(f"    ... 共 {len(recs)} 层")

    # 构造全局 (layer, slot) 字节区间（相对 base0）
    spans = []  # (off_begin, off_end, layer, slot)
    for r in recs:
        lb = r["base"] - base0
        st = r["stride"]
        for s in range(r["slots"]):
            spans.append((lb + s * st, lb + (s + 1) * st, r["layer"], s))
    spans.sort()
    buf_end = spans[-1][1]
    n_pages = (buf_end + page - 1) // page
    print(f"  整 kind buffer 大小 = {buf_end} B = {buf_end/1024/1024:.2f} MiB, "
          f"共 {n_pages} 页, {len(spans)} 个 slot 实例")

    # 扫每一页覆盖的 slot 数
    import bisect
    begins = [sp[0] for sp in spans]
    owner_counts = {}        # 一页覆盖的 slot 数 -> 页数
    cross_layer_pages = 0
    misaligned_cover = 0
    max_cover = 0
    example_shared = None
    for p in range(n_pages):
        lo, hi = p * page, (p + 1) * page
        # 可能与 [lo,hi) 相交的 span：start < hi 且 end > lo
        i = bisect.bisect_left(begins, hi) - 1
        owners = []
        j = i
        while j >= 0:
            b, e, l, s = spans[j]
            if e <= lo:
                break
            if b < hi and e > lo:
                owners.append((l, s))
            j -= 1
        if not owners:
            owner_counts[0] = owner_counts.get(0, 0) + 1
            continue
        c = len(owners)
        owner_counts[c] = owner_counts.get(c, 0) + 1
        max_cover = max(max_cover, c)
        if len({o[0] for o in owners}) > 1:
            cross_layer_pages += 1
        if c >= 2:
            misaligned_cover += 1
            if example_shared is None:
                example_shared = (p, owners)
    print(f"\n  页覆盖 slot 数分布（页数）:")
    for c in sorted(owner_counts):
        print(f"    覆盖 {c} 个 slot 的页 : {owner_counts[c]:>7} 页 "
              f"({100*owner_counts[c]/max(1,n_pages):5.1f}%)")
    print(f"  一页最多横跨 {max_cover} 个 slot；跨 layer 边界页 {cross_layer_pages} 个；"
          f"共享页（≥2 slot）{misaligned_cover} 个")
    if example_shared:
        p, owners = example_shared
        print(f"  共享页示例: 第 {p} 页 [{p*page:#x},{(p+1)*page:#x}) 覆盖 "
              f"{owners[:6]}{'...' if len(owners)>6 else ''}")

    # 结论：回收粒度
    stride = sorted(strides)[0]
    if stride % page == 0 and all(r["base"] % page == 0 for r in recs) \
            and all((r["slots"] * r["stride"]) % page == 0 for r in recs):
        print("\n  ✅ stride 与 layer 边界均页对齐：可按【单 slot】直接回收，无邻居误伤。")
    else:
        print("\n  ⚠ stride/layer 边界非页对齐：单 slot 回收会误伤相邻 slot。")
        print("    阶段 2 必须以【绝对 16KiB 页】为单位，仅当该页覆盖的所有 slot 都无 owner")
        print("    （全冷）才对该页 MADV_FREE_REUSABLE；跨 layer 边界页需两层相邻 slot 同时冷。")
        # 估连续多少 slot 凑整页（同层内）
        for k in range(1, 257):
            if (k * stride) % page == 0:
                print(f"    同层内：每 {k} 个连续 slot 的字节宽度 = {k*stride} B "
                      f"恰好页对齐（k*stride 是页整数倍）。")
                break
        else:
            print("    同层内 stride 与页在 256 slot 内无公倍数（以运行时逐页 owner 集合为准）。")


def analyze_whole(data, page=PAGE):
    """全区域：三个 kind 交织在同一连续 Metal 分配里，合并所有 tensor 按绝对页算覆盖。
    这是生产模式（CGC_POOL_SPLIT=0）的真实结构；按 kind 隔离会把其他 kind/gap 误算成空页。"""
    import bisect
    recs = []
    for kind, rs in data.items():
        for r in rs:
            if "stride" in r and "slots" in r and "base" in r:
                recs.append((r["layer"], kind, r["slots"], r["stride"], r["base"]))
    if not recs:
        print("\n（全区域分析：ADOPT 行缺失，无法重建）")
        return
    spans = []
    for (l, k, sl, st, b) in recs:
        for s in range(sl):
            spans.append((b + s * st, b + (s + 1) * st, l, k, s))
    spans.sort()
    base0 = spans[0][0]
    end = spans[-1][1]
    npg = (end - base0 + page - 1) // page
    begins = [x[0] for x in spans]
    from collections import Counter
    cov = Counter()
    cross = 0
    example = None
    for p in range(npg):
        lo, hi = base0 + p * page, base0 + (p + 1) * page
        i = bisect.bisect_left(begins, hi) - 1
        owners = []
        j = i
        while j >= 0:
            b, e, l, k, s = spans[j]
            if e <= lo:
                break
            if b < hi and e > lo:
                owners.append((l, k, s))
            j -= 1
        cov[len(owners)] += 1
        if len(owners) >= 2 and len({(o[0], o[1]) for o in owners}) > 1:
            cross += 1
        if len(owners) >= 2 and example is None:
            example = (p, owners)
    print("=" * 78)
    print(f" 全区域（3 kind 交织）连续 pool = {(end-base0)/1024/1024:.0f} MiB, {npg} 页, "
          f"{len(spans)} 个 (slot×kind)")
    print(f" pool base0 = 0x{base0:x}  (base0 mod page = {base0 % page})")
    print("=" * 78)
    for c in sorted(cov):
        print(f"   覆盖 {c} slot 的页 : {cov[c]:>7} ({100*cov[c]/max(1,npg):5.1f}%)")
    print(f"  跨 tensor(layer/kind) 共享页 = {cross}")
    if example:
        print(f"  共享页示例 第 {example[0]} 页: {example[1][:4]}")
    print("\n  回收口径：覆盖 1 slot 的页可在该 slot 冷时独立回收；覆盖 2 slot 的页需相邻 slot")
    print("  同时冷；覆盖 0 slot 的页是 gap/dense，不属于 pool、不可动。")


def self_test():
    print("="*78)
    print(" 自测：合成几何（模拟 IQ2_S gate/up stride≈68.5 页、143 slot/层、40 层）")
    print("="*78)
    page = PAGE
    # 1.0703 MiB ≈ 68.5 页 -> stride = 68*page + page//2 = 1122304（取半页残差，近似注释 1.0703MiB）
    stride = 68 * page + page // 2
    slots = 143
    n_layers = 40
    base0 = 0x100000000
    recs = []
    off = 0
    for l in range(n_layers):
        recs.append({"layer": l, "slots": slots, "stride": stride,
                     "base": base0 + off, "page": page,
                     "base_mod": (base0 + off) % page,
                     "stride_mod": stride % page,
                     "slice_mod": (slots * stride) % page,
                     "stride_pg": stride / page})
        off += slots * stride
    analyze_kind(0, recs, page)

    print("\n" + "="*78)
    print(" 自测：页对齐对照组（stride 恰好整数页、base 对齐）")
    print("="*78)
    stride2 = 70 * page
    recs2 = []
    off = 0
    for l in range(4):
        recs2.append({"layer": l, "slots": 140, "stride": stride2,
                      "base": base0 + off, "page": page,
                      "base_mod": 0, "stride_mod": 0, "slice_mod": 0,
                      "stride_pg": stride2 / page})
        off += 140 * stride2
    analyze_kind(3, recs2, page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="server stderr 日志（含 CGC-POOL-SPLIT-GEOM/ADOPT）")
    ap.add_argument("--self-test", action="store_true", help="用合成几何自测")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.log:
        ap.error("需要提供日志路径，或用 --self-test")
    data = parse(args.log)
    if not data:
        print("未解析到任何 CGC-POOL-SPLIT-GEOM/ADOPT 行。\n"
              "请确认启动 server 时设了 CGC_POOL_SPLIT_DBG=1 并已应用阶段1几何补丁。",
              file=sys.stderr)
        sys.exit(2)
    # 生产模式：先全区域（真实结构），再按 kind 看 stride 对齐
    analyze_whole(data)
    for kind in sorted(data):
        analyze_kind(kind, data[kind])


if __name__ == "__main__":
    main()
