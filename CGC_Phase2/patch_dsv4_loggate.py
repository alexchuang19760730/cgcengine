#!/usr/bin/env python3
"""Gate all logger.info() calls in dsv4 trace files behind DSV4_TRACE env var.
Uses __import__('os') to avoid touching `from __future__` import ordering.
"""
import re, sys, shutil, py_compile

GATE = "__import__('os').environ.get('DSV4_TRACE')"

for path in sys.argv[1:]:
    try:
        s = open(path).read()
    except FileNotFoundError:
        print("MISSING", path); continue
    orig = s
    # 1) gate every logger.info( -> if <GATE>: logger.info(
    s2 = re.sub(r'^(\s+)logger\.info\(',
                r'\1if ' + GATE + r': logger.info(',
                s, flags=re.M)
    # 2) gate invalid_count .item() assignment (moe.py)
    s2 = s2.replace(
        'invalid_count = int(invalid_mask.sum().item())',
        'invalid_count = int(invalid_mask.sum().item()) if ' + GATE + ' else 0')
    if s2 == orig:
        print("NO_CHANGE", path); continue
    shutil.copy2(path, path + ".bak_loggate")
    open(path, "w").write(s2)
    py_compile.compile(path, doraise=True)
    print("GATED", path, "logger.info:", orig.count("logger.info("))
