#!/usr/bin/env python3
"""Fix repack bug: resident weights were stored as FP16 bytes but tagged BF16.

Streams model_weights.bin, converting every dtype=1 (BF16-tagged) entry from
FP16 bytes to true BF16 bytes, writing model_weights.bin.fixed (original
preserved). Per-entry chunked conversion bounds memory (~64MB at a time).
Usage: convert_resident_fp16_to_bf16.py <model_dir>...
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack')
from resident_writer import read_resident_bin

CHUNK = 64 * 1024 * 1024  # 64MB per conversion chunk

def vec_fp16_to_f32(raw: np.ndarray) -> np.ndarray:
    """fp16 u16 bits -> fp32 values (vectorized, handles subnormals)."""
    u = raw.astype(np.uint32)
    sign = (u >> 15) & 1
    exp = (u >> 10) & 0x1F
    frac = u & 0x3FF
    f32 = np.zeros_like(u, dtype=np.float32)
    m_norm = exp != 0
    exp_b = ((exp.astype(np.int64) - 15 + 127).astype(np.uint32) & 0xFF)
    bits_norm = (sign << 31) | (exp_b << 23) | (frac << 13)
    f32[m_norm] = bits_norm[m_norm].view(np.float32)
    m_sub = (exp == 0) & (frac != 0)
    f32[m_sub] = frac[m_sub].astype(np.float32) * np.float32(2.0 ** -24) * \
        np.where(sign[m_sub] == 1, -1.0, 1.0)
    m_inf = (exp == 0x1F) & (frac == 0)
    f32[m_inf] = np.where(sign[m_inf] == 1, -np.inf, np.inf)
    m_nan = (exp == 0x1F) & (frac != 0)
    f32[m_nan] = np.nan
    return f32

def to_bf16_bytes(f32: np.ndarray) -> np.ndarray:
    bits = f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    bias = np.uint32(0x7FFF) + lsb
    return ((bits + bias) >> 16).astype(np.uint16).tobytes()

def convert(p: Path):
    info = read_resident_bin(p)
    entries = info['entries']
    data_start = info['indexSize']
    dst = p.with_name('model_weights.bin.fixed')
    print(f'{p}: {len(entries)} entries, index={data_start}B data={info["residentSize"]/1e6:.0f}MB -> {dst.name}')
    with open(p, 'rb') as fin, open(dst, 'wb') as fout:
        # copy index region verbatim
        fin.seek(0)
        fout.write(fin.read(data_start))
        converted = 0
        for name, e in entries.items():
            off = e['fileOffset']; sz = e['sizeBytes']
            if e['dtype'] != 1:
                # copy verbatim
                fin.seek(off)
                remaining = sz
                while remaining > 0:
                    chunk = fin.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    fout.write(chunk)
                    remaining -= len(chunk)
                continue
            assert sz % 2 == 0
            n_total = sz // 2
            fin.seek(off)
            done = 0
            while done < n_total:
                n = min(CHUNK // 2, n_total - done)
                raw = np.frombuffer(fin.read(n * 2), dtype=np.uint16)
                f32 = vec_fp16_to_f32(raw)
                fout.write(to_bf16_bytes(f32))
                done += n
            converted += 1
        print(f'  converted {converted} BF16-tagged entries (fp16->bf16)')

if __name__ == '__main__':
    for d in sys.argv[1:]:
        if d.startswith('--'):
            continue
        convert(Path(d) / 'model_weights.bin')
    print('done')
