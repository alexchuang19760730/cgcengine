import os, sys
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#!/usr/bin/env python3
"""Mock 绔埌绔?demo 鈥?鍦?Windows 涓婇璀夊畬鏁?PD 绠＄窔 (鐒￠渶 Mac).

妯℃摤:
  1. Mac A (Gemma4) emit: 鐢ㄩ毃姗熷嫉閲忔ā鎿?hidden state [seq_len, 2816]
  2. MoT-h 缈昏: 2816 鈫?2048
  3. Context Replay: hidden 鈫?KV cache
  4. Mac B (Qwen3.6) resume: 妯℃摤 decode 杓稿嚭

椹楄瓑榛?
  鉁?protocol.py 鐨?encode/decode hidden state 姝ｇ⒑
  鉁?MoT-h translate_hidden 鍓嶅悜璺戦€?(2816 鈫?2048)
  鉁?channel_mapping 閫氶亾鏄犲皠姝ｇ⒑
  鉁?context_replay KV 閭勫師姝ｇ⒑
  鉁?绔埌绔绶氬舰鐙€/鏁稿€肩劇 NaN

鐢ㄦ硶:
  py mock_demo.py
"""
from __future__ import annotations

import os
import sys
import time
import math

# 鍕曟厠鍔犲叆璺緫
_HERE = os.path.dirname(os.path.abspath(__file__))
_CGC_ENGINE = _HERE  # 宸插湪 cgc-engine/pd/ 涓?
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
    status = "鉁? if condition else "鉂?
    print(f"  {status} {msg}")
    return condition


def main():
    print("鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晽")
    print("鈺? CGC PD Mock Demo 鈥?Gemma4 鈫?MoT-h 鈫?Qwen3.6 绔埌绔?   鈺?)
    print("鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨暆")

    torch.manual_seed(42)
    all_ok = True

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    section("1. 妯″瀷閰嶇疆")
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    src = SourceModel.GEMMA4_26B_A4B
    tgt = TargetModel.QWEN36_35B_A3B
    print(f"  Source: {src.value}  hidden={src.hidden_size} layers={src.num_layers}")
    print(f"  Target: {tgt.value}  hidden={tgt.hidden_size} layers={tgt.num_layers}")
    check(src.hidden_size == 2816, "Gemma4 hidden_size=2816")
    check(src.num_layers == 30, "Gemma4 num_layers=30")
    check(tgt.hidden_size == 2048, "Qwen3.6 hidden_size=2048")
    check(tgt.num_layers == 40, "Qwen3.6 num_layers=40")

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    section("2. Protocol: hidden state encode/decode")
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    seq_len = 128
    hidden_src = torch.randn(seq_len, src.hidden_size)
    print(f"  鍘熷: {tuple(hidden_src.shape)}, dtype={hidden_src.dtype}")

    # 绶ㄧ⒓
    b64 = encode_hidden_state(hidden_src)
    print(f"  base64: {len(b64)} chars ({len(b64)/1024:.1f} KB)")

    # 瑙ｇ⒓
    hidden_decoded = decode_hidden_state(b64, seq_len, src.hidden_size)
    print(f"  瑙ｇ⒓: {tuple(hidden_decoded.shape)}, dtype={hidden_decoded.dtype}")

    # 椹楄瓑 round-trip
    max_diff = (hidden_src - hidden_decoded).abs().max().item()
    all_ok &= check(max_diff < 1e-6, f"round-trip 瑾ゅ樊 < 1e-6 (瀵﹂殯={max_diff:.2e})")

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

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    section("3. Channel Mapping: Gemma4 30灞?鈫?Qwen3.6 40灞?)
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cfg = get_gemma4_to_qwen36_channels()
    print(f"  閫氶亾姣斾緥: {cfg['channel_ratio']}")
    print(f"  婧愰€氶亾灞?({len(cfg['src_channel_layers'])}): {cfg['src_channel_layers']}")
    print(f"  鐩閫氶亾灞?({len(cfg['tgt_channel_layers'])}): {cfg['tgt_channel_layers']}")
    print(f"  绐楀彛鏁? {len(cfg['windows'])}")
    print(f"  娣卞害姣斾緥閰嶅皪 (鍓?):")
    for i, j in cfg["depth_ratio_pairs"][:5]:
        print(f"    src[{i:2d}] (d={i/30:.3f}) 鈫?tgt[{j:2d}] (d={j/40:.3f})")

    all_ok &= check(len(cfg["src_channel_layers"]) > 0, "閫氶亾灞ら潪绌?)
    all_ok &= check(len(cfg["windows"]) > 0, "绐楀彛闈炵┖")

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    section("4. MoT-h 缈昏: 2816 鈫?2048")
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    mot_cfg = MoTHConfig(
        src_hidden_size=src.hidden_size,   # 2816
        tgt_hidden_size=tgt.hidden_size,    # 2048
        src_num_layers=src.num_layers,      # 30
        tgt_num_layers=tgt.num_layers,      # 40
    )
    print(f"  MoT-h 閰嶇疆:")
    print(f"    src_hidden={mot_cfg.src_hidden_size}, tgt_hidden={mot_cfg.tgt_hidden_size}")
    print(f"    num_translators={mot_cfg.num_translators}, top_k={mot_cfg.top_k}")
    print(f"    window_size={mot_cfg.window_size}")

    mot = MoTH(mot_cfg)
    num_params = sum(p.numel() for p in mot.parameters())
    print(f"  鍙冩暩閲? {num_params:,} ({num_params/1e6:.1f}M)")

    # 鍓嶅悜 (鐢?translate_hidden 渚挎嵎鏂规硶)
    t0 = time.time()
    hidden_tgt = mot.translate_hidden(hidden_src)
    t1 = time.time()
    print(f"  杓稿叆: {tuple(hidden_src.shape)}")
    print(f"  杓稿嚭: {tuple(hidden_tgt.shape)}")
    print(f"  鑰楁檪: {(t1-t0)*1000:.1f}ms")

    all_ok &= check(hidden_tgt.shape == (seq_len, tgt.hidden_size),
                    f"杓稿嚭褰㈢媭 = ({seq_len}, {tgt.hidden_size})")
    all_ok &= check(not torch.isnan(hidden_tgt).any(), "杓稿嚭鐒?NaN")
    all_ok &= check(not torch.isinf(hidden_tgt).any(), "杓稿嚭鐒?Inf")
    print(f"  绲辫▓: mean={hidden_tgt.mean():.4f} std={hidden_tgt.std():.4f}")

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    section("5. Context Replay: hidden 鈫?KV cache")
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # 妯℃摤 Qwen3.6 鐨?Wk/Wv
    kv_dim = 512  # GQA
    wk = torch.randn(tgt.num_layers, tgt.hidden_size, kv_dim) * 0.02
    wv = torch.randn(tgt.num_layers, tgt.hidden_size, kv_dim) * 0.02
    print(f"  Wk: {tuple(wk.shape)}, Wv: {tuple(wv.shape)}")

    K, V = context_replay_mvp(hidden_tgt, wk, wv)
    print(f"  K: {tuple(K.shape)}, V: {tuple(V.shape)}")

    all_ok &= check(K.shape == (tgt.num_layers, seq_len, kv_dim),
                    f"K 褰㈢媭 = ({tgt.num_layers}, {seq_len}, {kv_dim})")
    all_ok &= check(not torch.isnan(K).any(), "K 鐒?NaN")
    all_ok &= check(not torch.isnan(V).any(), "V 鐒?NaN")

    # 鍌宠几閲忓皪姣?
    hidden_bytes = seq_len * src.hidden_size * 4
    kv_bytes = tgt.num_layers * seq_len * kv_dim * 4 * 2
    print(f"  鍌宠几閲? hidden={hidden_bytes/1024:.1f}KB, 瀹屾暣KV={kv_bytes/1024:.1f}KB")
    print(f"  绡€鐪? {(1 - hidden_bytes/kv_bytes)*100:.1f}%")

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    section("6. 绔埌绔绶氬舰鐙€椹楄瓑")
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print(f"  Step 1: emit (妯℃摤 Gemma4 prefill)")
    print(f"    鈫?hidden_src: {tuple(hidden_src.shape)} [seq_len, 2816]")

    print(f"  Step 2: protocol encode/decode (妯℃摤缍茶矾鍌宠几)")
    b64 = encode_hidden_state(hidden_src)
    hidden_recv = decode_hidden_state(b64, seq_len, src.hidden_size)
    print(f"    鈫?hidden_recv: {tuple(hidden_recv.shape)} [base64: {len(b64)/1024:.1f}KB]")

    print(f"  Step 3: MoT-h 缈昏 (2816 鈫?2048)")
    hidden_tgt = mot.translate_hidden(hidden_recv)
    print(f"    鈫?hidden_tgt: {tuple(hidden_tgt.shape)} [seq_len, 2048]")

    print(f"  Step 4: Context Replay (hidden 鈫?KV cache)")
    K, V = context_replay_mvp(hidden_tgt, wk, wv)
    print(f"    鈫?K: {tuple(K.shape)}, V: {tuple(V.shape)}")

    print(f"  Step 5: resume (妯℃摤 Qwen3.6 decode)")
    # 妯℃摤 decode 绗竴鍊?token (鐢ㄦ渶寰屼竴鍊嬩綅缃殑 hidden 鈫?lm_head)
    lm_head = torch.randn(tgt.hidden_size, 32000) * 0.01  # vocab=32000
    last_hidden = hidden_tgt[-1:]  # [1, 2048]
    logits = last_hidden @ lm_head  # [1, 32000]
    next_token = logits.argmax(dim=-1).item()
    print(f"    鈫?next_token_id: {next_token}")

    all_ok &= check(logits.shape == (1, 32000), "lm_head 杓稿嚭褰㈢媭姝ｇ⒑")

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    section("绺界祼")
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if all_ok:
        print("  鉁?鎵€鏈夐璀夐€氶亷! 绠＄窔褰㈢媭/鏁稿€兼纰?")
        print()
        print("  涓嬩竴姝?")
        print("    1. Mac A 瀵︾従 TurboFieldfare /v1/cgc/emit (Gemma4 prefill)")
        print("    2. Mac B 瀵︾従 TurboFieldfare /v1/cgc/resume (Qwen3.6 decode)")
        print("    3. Mac B 瀵︾従 TurboFieldfare /v1/cgc/emit (Qwen3.6 prefill, 鎺￠泦鐢?")
        print("    4. 璺?collect_parallel_data.py 鎺￠泦瑷撶反灏?)
        print("    5. 瑷撶反 MoT-h (train_mot_h.py)")
        print("    6. 鍟熷嫊 coordinator.py 绔埌绔脯瑭?)
    else:
        print("  鉂?鏈夐璀夊け鏁? 璜嬫鏌ヤ笂鏂规瑷?鉂?鐨勯爡鐩?)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
