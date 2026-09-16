#!/usr/bin/env python3
"""Delivery self-check for a docs/*.html whitepaper (cgc-whitepaper-delivery checklist).

Checks: HTML tag balance, markdown residue outside pre/code, stray code fences, and that every
repo path the document names actually exists. Run before committing a whitepaper.
"""
import html.parser
import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
bad = 0


class C(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.st, self.err = [], []
        self.void = {"meta", "br", "hr", "img", "link", "input"}

    def handle_starttag(self, t, a):
        if t not in self.void:
            self.st.append((t, self.getpos()))

    def handle_endtag(self, t):
        if t in self.void:
            return
        if self.st and self.st[-1][0] == t:
            self.st.pop()
        else:
            self.err.append((t, self.getpos()))


c = C()
c.feed(s)
print(f"unclosed : {c.st[:5]}")
print(f"mismatched: {c.err[:5]}")
bad += len(c.st) + len(c.err)

body = re.sub(r"<(pre|code)\b[^>]*>.*?</\1>", "", s, flags=re.S)
res = re.findall(r"\*\*[^*\n]{1,90}\*\*|`[^`\n]{1,90}`", body)
print(f"markdown residue outside pre/code: {res or 'none'}")
bad += len(res)

fences = s.count("```")
print(f"code fences (```): {fences}")
bad += fences

miss = []
for m in sorted(set(re.findall(r"(?:docs|scripts|agent_harness|Backup|src)/[A-Za-z0-9_./+-]+", s))):
    if not Path(m.rstrip(".")).exists():
        miss.append(m)
print(f"missing paths: {miss or 'none'}")

print(f"\n{'OK' if bad == 0 else 'NEEDS FIX'}  ({bad} finding(s))")
sys.exit(1 if bad else 0)
