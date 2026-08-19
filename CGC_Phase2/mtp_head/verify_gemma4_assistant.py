#!/usr/bin/env python3
"""端側驗證: Python 同構 Gemma4Assistant vs 引擎 MTP probe 的 accept 率對拍。

方法:
1. 用引擎跑 gemma4-r3 + MTP (draft=3), 從 log 抓 accept 率 (基準)
2. 用 Python 同構模型 + 引擎導出的 (backbone_hidden, token) 資料做 draft 預測
   對照 target 主模型 greedy token, 算 Python head 的 accept rate

驗證目標: Python head 的 accept ≈ 引擎 head 的 accept (58%) 才是對齊成功。
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gemma4_assistant import load_gemma4_assistant


def main():
    model_dir = "/Users/alexhuang/Documents/flashkv0516/models/gemma-4-mtp-head"
    head, info = load_gemma4_assistant(model_dir, device="cpu")
    print("missing:", len(info["missing"]), "unexpected:", len(info["unexpected"]))
    print("params:", sum(p.numel() for p in head.parameters()) / 1e6, "M")

    # ---- 端側真實資料收集: 用引擎產生 (hidden, token) 對 ----
    # 這裡先用簡化路徑: 引擎 MTP log 已有 accept 基準 58% (2026-08-11 實測)
    # Python head 對齊驗證需要真實 backbone hidden, 由引擎導出 (下一步)
    print("Python head loaded OK. 等待引擎導出真實 decode hidden 進行對拍。")


if __name__ == "__main__":
    main()
