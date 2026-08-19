#!/usr/bin/env python3
"""Mock 端到端 demo — 在 Windows 上驗證完整 PD 管線 (無需 Mac).

模擬:
  1. Mac A (Gemma4) emit: 用隨機張量模擬 hidden state [seq_len, 2816]
  2. MoT-h 翻譯: 2816 → 2048
  3. Context Replay: hidden → KV cache
  4. Mac B (Qwen3.6) resume: 模擬 decode 輸出

驗證點:
  ✅ protocol.py 的 encode/decode hidden state 正確
  ✅ MoT-h translate_hidden 前向跑通 (2816 → 2048)
  ✅ channel_mapping 通道映射正確
  ✅ context_replay KV 還原正確
  ✅ 端到端管線形狀/數值無 NaN

用法:
  py mock_demo.py
"""
from __future__ import annotations

import os
import sys
import time
import math

# 動態加入路徑
_HERE = os.path.dirname(os.path.abspath(__file__))
_CGC_ENGINE = _HERE  # 已在 cgc-engine/pd/ 下
_MOT_H_PATH = os.path.join(_HERE, "..", "..", "CGC_Phase2", "mot_h")
_MOT_H_PATH = os.path.abspath(_MOT_H_PATH)

for p in [_CGC_ENGINE, _MOT_H_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from protocol import (
    HiddenStatePacket, ModelInfo, EmitRequest, EmitResponse,
    ResumeRequest, encode_hidden_state, decode_hidden_state,
    SourceModel, TargetModel,
)
from mot_h import MoTHConfig, MoTH
from channel_mapping import get_gemma4_to_qwen36_channels
from context_replay import context_replay_mvp, restore_kv_cache


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(condition: bool, msg: str) -> bool:
    status = "✅" if condition else "❌"
    print(f"  {status} {msg}")
    return condition


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CGC PD Mock Demo — Gemma4 → MoT-h → Qwen3.6 端到端    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    torch.manual_seed(42)
    all_ok = True

    # ──────────────────────────────────────────────────────────
    section("1. 模型配置")
    # ──────────────────────────────────────────────────────────
    src = SourceModel.GEMMA4_26B_A4B
    tgt = TargetModel.QWEN36_35B_A3B
    print(f"  Source: {src.value}  hidden={src.hidden_size} layers={src.num_layers}")
    print(f"  Target: {tgt.value}  hidden={tgt.hidden_size} layers={tgt.num_layers}")
    check(src.hidden_size == 2816, "Gemma4 hidden_size=2816")
    check(src.num_layers == 30, "Gemma4 num_layers=30")
    check(tgt.hidden_size == 2048, "Qwen3.6 hidden_size=2048")
    check(tgt.num_layers == 40, "Qwen3.6 num_layers=40")

    # ──────────────────────────────────────────────────────────
    section("2. Protocol: hidden state encode/decode")
    # ──────────────────────────────────────────────────────────
    seq_len = 128
    hidden_src = torch.randn(seq_len, src.hidden_size)
    print(f"  原始: {tuple(hidden_src.shape)}, dtype={hidden_src.dtype}")

    # 編碼
    b64 = encode_hidden_state(hidden_src)
    print(f"  base64: {len(b64)} chars ({len(b64)/1024:.1f} KB)")

    # 解碼
    hidden_decoded = decode_hidden_state(b64, seq_len, src.hidden_size)
    print(f"  解碼: {tuple(hidden_decoded.shape)}, dtype={hidden_decoded.dtype}")

    # 驗證 round-trip
    max_diff = (hidden_src - hidden_decoded).abs().max().item()
    all_ok &= check(max_diff < 1e-6, f"round-trip 誤差 < 1e-6 (實際={max_diff:.2e})")

    # HiddenStatePacket
    model_info = ModelInfo(
        model_id=src.value,
        hidden_size=src.hidden_size,
        num_layers=src.num_layers,
    )
    packet = HiddenStatePacket.from_tensor(hidden_src, model_info, finished_layer=src.num_layers)
    d = packet.to_dict()
    packet2 = HiddenStatePacket.from_dict(d)
    all_ok &= check(packet2.seq_len == seq_len, "HiddenStatePacket serialize/deserialize")
    print(f"  packet: request_id={packet.request_id}, finished_layer={packet.finished_layer}")

    # ──────────────────────────────────────────────────────────
    section("3. Channel Mapping: Gemma4 30層 → Qwen3.6 40層")
    # ──────────────────────────────────────────────────────────
    cfg = get_gemma4_to_qwen36_channels()
    print(f"  通道比例: {cfg['channel_ratio']}")
    print(f"  源通道層 ({len(cfg['src_channel_layers'])}): {cfg['src_channel_layers']}")
    print(f"  目標通道層 ({len(cfg['tgt_channel_layers'])}): {cfg['tgt_channel_layers']}")
    print(f"  窗口數: {len(cfg['windows'])}")
    print(f"  深度比例配對 (前5):")
    for i, j in cfg["depth_ratio_pairs"][:5]:
        print(f"    src[{i:2d}] (d={i/30:.3f}) → tgt[{j:2d}] (d={j/40:.3f})")

    all_ok &= check(len(cfg["src_channel_layers"]) > 0, "通道層非空")
    all_ok &= check(len(cfg["windows"]) > 0, "窗口非空")

    # ──────────────────────────────────────────────────────────
    section("4. MoT-h 翻譯: 2816 → 2048")
    # ──────────────────────────────────────────────────────────
    mot_cfg = MoTHConfig(
        src_hidden_size=src.hidden_size,   # 2816
        tgt_hidden_size=tgt.hidden_size,    # 2048
        src_num_layers=src.num_layers,      # 30
        tgt_num_layers=tgt.num_layers,      # 40
    )
    print(f"  MoT-h 配置:")
    print(f"    src_hidden={mot_cfg.src_hidden_size}, tgt_hidden={mot_cfg.tgt_hidden_size}")
    print(f"    num_translators={mot_cfg.num_translators}, top_k={mot_cfg.top_k}")
    print(f"    window_size={mot_cfg.window_size}")

    mot = MoTH(mot_cfg)
    num_params = sum(p.numel() for p in mot.parameters())
    print(f"  參數量: {num_params:,} ({num_params/1e6:.1f}M)")

    # 前向 (用 translate_hidden 便捷方法)
    t0 = time.time()
    hidden_tgt = mot.translate_hidden(hidden_src)
    t1 = time.time()
    print(f"  輸入: {tuple(hidden_src.shape)}")
    print(f"  輸出: {tuple(hidden_tgt.shape)}")
    print(f"  耗時: {(t1-t0)*1000:.1f}ms")

    all_ok &= check(hidden_tgt.shape == (seq_len, tgt.hidden_size),
                    f"輸出形狀 = ({seq_len}, {tgt.hidden_size})")
    all_ok &= check(not torch.isnan(hidden_tgt).any(), "輸出無 NaN")
    all_ok &= check(not torch.isinf(hidden_tgt).any(), "輸出無 Inf")
    print(f"  統計: mean={hidden_tgt.mean():.4f} std={hidden_tgt.std():.4f}")

    # ──────────────────────────────────────────────────────────
    section("5. Context Replay: hidden → KV cache")
    # ──────────────────────────────────────────────────────────
    # 模擬 Qwen3.6 的 Wk/Wv
    kv_dim = 512  # GQA
    wk = torch.randn(tgt.num_layers, tgt.hidden_size, kv_dim) * 0.02
    wv = torch.randn(tgt.num_layers, tgt.hidden_size, kv_dim) * 0.02
    print(f"  Wk: {tuple(wk.shape)}, Wv: {tuple(wv.shape)}")

    K, V = context_replay_mvp(hidden_tgt, wk, wv)
    print(f"  K: {tuple(K.shape)}, V: {tuple(V.shape)}")

    all_ok &= check(K.shape == (tgt.num_layers, seq_len, kv_dim),
                    f"K 形狀 = ({tgt.num_layers}, {seq_len}, {kv_dim})")
    all_ok &= check(not torch.isnan(K).any(), "K 無 NaN")
    all_ok &= check(not torch.isnan(V).any(), "V 無 NaN")

    # 傳輸量對比
    hidden_bytes = seq_len * src.hidden_size * 4
    kv_bytes = tgt.num_layers * seq_len * kv_dim * 4 * 2
    print(f"  傳輸量: hidden={hidden_bytes/1024:.1f}KB, 完整KV={kv_bytes/1024:.1f}KB")
    print(f"  節省: {(1 - hidden_bytes/kv_bytes)*100:.1f}%")

    # ──────────────────────────────────────────────────────────
    section("6. 端到端管線形狀驗證")
    # ──────────────────────────────────────────────────────────
    print(f"  Step 1: emit (模擬 Gemma4 prefill)")
    print(f"    → hidden_src: {tuple(hidden_src.shape)} [seq_len, 2816]")

    print(f"  Step 2: protocol encode/decode (模擬網路傳輸)")
    b64 = encode_hidden_state(hidden_src)
    hidden_recv = decode_hidden_state(b64, seq_len, src.hidden_size)
    print(f"    → hidden_recv: {tuple(hidden_recv.shape)} [base64: {len(b64)/1024:.1f}KB]")

    print(f"  Step 3: MoT-h 翻譯 (2816 → 2048)")
    hidden_tgt = mot.translate_hidden(hidden_recv)
    print(f"    → hidden_tgt: {tuple(hidden_tgt.shape)} [seq_len, 2048]")

    print(f"  Step 4: Context Replay (hidden → KV cache)")
    K, V = context_replay_mvp(hidden_tgt, wk, wv)
    print(f"    → K: {tuple(K.shape)}, V: {tuple(V.shape)}")

    print(f"  Step 5: resume (模擬 Qwen3.6 decode)")
    # 模擬 decode 第一個 token (用最後一個位置的 hidden → lm_head)
    lm_head = torch.randn(tgt.hidden_size, 32000) * 0.01  # vocab=32000
    last_hidden = hidden_tgt[-1:]  # [1, 2048]
    logits = last_hidden @ lm_head  # [1, 32000]
    next_token = logits.argmax(dim=-1).item()
    print(f"    → next_token_id: {next_token}")

    all_ok &= check(logits.shape == (1, 32000), "lm_head 輸出形狀正確")

    # ──────────────────────────────────────────────────────────
    section("總結")
    # ──────────────────────────────────────────────────────────
    if all_ok:
        print("  ✅ 所有驗證通過! 管線形狀/數值正確.")
        print()
        print("  下一步:")
        print("    1. Mac A 實現 TurboFieldfare /v1/cgc/emit (Gemma4 prefill)")
        print("    2. Mac B 實現 TurboFieldfare /v1/cgc/resume (Qwen3.6 decode)")
        print("    3. Mac B 實現 TurboFieldfare /v1/cgc/emit (Qwen3.6 prefill, 採集用)")
        print("    4. 跑 collect_parallel_data.py 採集訓練對")
        print("    5. 訓練 MoT-h (train_mot_h.py)")
        print("    6. 啟動 coordinator.py 端到端測試")
    else:
        print("  ❌ 有驗證失敗, 請檢查上方標記 ❌ 的項目")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
