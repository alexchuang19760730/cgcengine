#!/usr/bin/env python3
"""knifeedge.anchor — 路徑錨點（手寫模組，不由 split_module.py 產生）。

為什麼這個模組是手寫的
----------------------
原檔 `scripts/check/knifeedge_matrix.py` 第 78 行是：

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

`__file__` 在 `scripts/check/`，往上三層剛好是 repo 根。搬進 `scripts/check/knifeedge/`
之後，**同一個表達式**往上三層會落到 `scripts/` —— 值變了，而沒有任何語法、import 或
型別檢查會發現（`eng-bound-0007`：自帶錨點只保證相對於自己，不保證相對於世界）。

所以這裡把它展開成看得見的三步，並多一個 `ENTRY_FILE`。`record_provenance()` 原本用
`open(os.path.abspath(__file__))` 記「跑的是哪支腳本」的摘要，搬過來之後 `__file__` 會指向
本模組 —— 同一個錯誤類別，同一個修法。

兩個值都由 `agent_harness/shared/check_module_split.py` 在 runtime 斷言與搬遷前相同：
`ROOT` 必須等於搬遷前的絕對路徑，`ENTRY_FILE` 必須是呼叫者實際執行的那個檔案。
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))       # .../scripts/check/knifeedge
CHECK_DIR = os.path.dirname(_HERE)                       # .../scripts/check
ROOT = os.path.dirname(os.path.dirname(CHECK_DIR))       # repo 根

# 呼叫者執行的那個檔案。`__file__` 不是它 —— 本模組的 `__file__` 指向這個子目錄。
ENTRY_FILE = os.path.join(CHECK_DIR, "knifeedge_matrix.py")

GGUF = os.path.join(ROOT, "models", "gguf")
SERVER_MATCH = "build/bin/llama-server"
RESULT_DIR = os.path.join(ROOT, "Backup", "knifeedge_matrix")
DEFAULT_PORT = 8080
