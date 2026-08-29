#!/usr/bin/env python3
# decode_breakdown.py — 深度 decode 段分解（2026-08-29）
# 把 steady MTP decode 的 ~110ms/step 拆成 verify compute / catch-up fill / graph submit
# 資料源：CGC-SEG 週期打印（每 160 hooks 印累積平均）→ 逐窗口 delta = 該窗口真實平均
import re
import sys

def parse(path):
    rows = []
    for line in open(path, errors="ignore"):
        m = re.match(r"CGC-SEG: wait ([\d.]+) cb ([\d.]+) submit ([\d.]+) us \((\d+)\)", line)
        if m:
            rows.append(tuple(float(m.group(i)) for i in (1, 2, 3)) + (int(m.group(4)),))
    return rows

def windows(rows):
    # cum_avg*n = cum_total; delta of totals between prints = window sum (160 hooks)
    out = []
    for a, b in zip(rows, rows[1:]):
        n_a, n_b = a[3], b[3]
        dn = n_b - n_a
        if dn <= 0:
            continue
        dw = b[0] * n_b - a[0] * n_a
        dc = b[1] * n_b - a[1] * n_a
        ds = b[2] * n_b - a[2] * n_a
        out.append((n_b, dw / dn, dc / dn, ds / dn))
    return out

def main(path, prefill_hooks, steps, decode_wall_s, label=""):
    rows = parse(path)
    if not rows:
        print(f"{label}: NO CGC-SEG DATA in {path}")
        return
    wins = windows(rows)
    dec = [w for w in wins if w[0] > prefill_hooks]
    pre = [w for w in wins if w[0] <= prefill_hooks]
    tot_hooks = rows[-1][3]
    dec_hooks = tot_hooks - prefill_hooks

    def agg(sel, idx):
        if not sel:
            return 0.0
        # hooks-weighted mean over selected windows (window = 160 hooks each)
        return sum(w[idx] for w in sel) / len(sel)

    dw, dc, ds = agg(dec, 1), agg(dec, 2), agg(dec, 3)
    hooks_per_step = dec_hooks / steps
    print(f"===== {label} =====")
    print(f"total hooks={tot_hooks}  prefill_hooks≈{prefill_hooks}  decode_hooks={dec_hooks}  steps={steps}")
    print(f"hooks/step = {hooks_per_step:.1f}   decode wall = {decode_wall_s:.2f}s")
    print(f"decode-phase per-hook us: wait(GPU compute)={dw:.1f}  cb(topk hook+fill)={dc:.1f}  submit(encode)={ds:.1f}")
    sw, sc, ss = dw * hooks_per_step / 1000, dc * hooks_per_step / 1000, ds * hooks_per_step / 1000
    wall_ms = decode_wall_s * 1000 / steps
    print(f"per-step ms:  verify_compute={sw:.1f}  catch_up_fill={sc:.1f}  graph_submit={ss:.1f}  sum={sw+sc+ss:.1f}")
    print(f"measured step wall = {wall_ms:.1f} ms   residual(CPU split/sampler/KV/其他) = {wall_ms-sw-sc-ss:.1f} ms")
    print(f"佔比: verify {sw/wall_ms*100:.0f}%  fill {sc/wall_ms*100:.0f}%  submit {ss/wall_ms*100:.0f}%  residual {(wall_ms-sw-sc-ss)/wall_ms*100:.0f}%")
    if pre:
        pw, pc, ps = agg(pre, 1), agg(pre, 2), agg(pre, 3)
        print(f"(prefill 對照 per-hook us: wait={pw:.1f} cb={pc:.1f} submit={ps:.1f})")

if __name__ == "__main__":
    # 用法：decode_breakdown.py <err檔> <prefill_hooks> <steps> <decode_wall_s> <label>
    #   prefill_hooks：prefill 結束時的 CGC-SEG hook 計數（stall 版可由 watchdog print 對齊；
    #                  本 steady 配置實測 ≈ 2400：153 chunks × ~16 hooks/chunk，runs 間 bit-identical）
    #   steps        ：decode 步數 = decoded_tokens - n_accept（MTP n_max=2：370）
    #   decode_wall_s：decoded N tokens in M seconds（err 檔 perf print）
    # 無參數時跑內建示例（2026-08-29 實測資料）
    if len(sys.argv) >= 6:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
    else:
        # dlk2a_P1: clean 26.126 t/s（cold→warm session 早期）
        main("/tmp/dlk2a_P1.err", 2400, 370, 42.180, "P1 clean 26.13 t/s")
        # dlk2_P8: polluted 19.055 t/s（session 後期 page-cache 壓縮）
        main("/tmp/dlk2_P8.err", 2400, 370, 57.833, "P8 polluted 19.06 t/s")
