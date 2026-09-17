#!/usr/bin/env python3
"""knifeedge._source — 「這個 harness 的原始碼」的唯一入口（手寫模組）。

為什麼需要它
------------
拆檔之前，有呼叫端以「檔案」為單位做字串斷言，例如
`scripts/check/oracle_truth_gate_selftest.py`：

    src = open(os.path.join(HERE, "knifeedge_matrix.py")).read()
    assert '"comparison": "absolute" if absolute else "relative"' in src

它問的是「harness 裡有這段邏輯嗎」，只是當時 harness 恰好只有一個檔案，所以它寫成了
一個路徑。拆檔之後那個問題仍然成立、那個路徑不再成立 —— 於是這裡提供一個入口，
讓「harness 的原始碼」有定義，而不是讓每個呼叫端各自去猜檔案佈局。
"""
from __future__ import annotations

import os

from .anchor import ENTRY_FILE

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def source_text(entry: str | None = None) -> str:
    """shim ＋ 套件裡每個模組的原始碼，接成一份字串。

    `entry=None` 時用 `anchor.ENTRY_FILE`。不需要排序保證：字串斷言問的是
    「這段邏輯在不在」，不是「在第幾行」。
    """
    parts = []
    entry = entry or ENTRY_FILE
    if entry and os.path.exists(entry):
        with open(entry, encoding="utf-8") as f:
            parts.append(f.read())
    for name in sorted(os.listdir(_PKG_DIR)):
        if name.endswith(".py"):
            with open(os.path.join(_PKG_DIR, name), encoding="utf-8") as f:
                parts.append(f.read())
    return "\n".join(parts)
