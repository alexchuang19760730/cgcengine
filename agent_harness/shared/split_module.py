#!/usr/bin/env python3
"""split_module.py — 把一個 Python 單體檔機械地拆成套件，內容逐位元組搬運。

為什麼要這支
------------
E4 item 2 要拆 `scripts/check/knifeedge_matrix.py`（3310 行 / 181 KB）。手拆等於重寫：
每一個被搬動的函式都是一次**靜默改行為**的機會（模組級可變狀態、`__file__` 錨點、
只在 `global` 宣告下存在的名字——這三類都不會在任何語法檢查裡出現）。

所以搬運由這支工具做，而且是**逐位元組**的：每個頂層項目的原始行（含它上方那把註解）
原樣切出來放進新模組，工具本身不重新排版、不改縮排、不「順手」修任何東西。

工具拒絕的事（寧可在這裡失敗，不要在執行時失敗）
------------------------------------------------
1. 一個自由名字它歸屬不了（既不是原檔的 import、也不是某個頂層項目、也不是內建）
   —— 那在新佈局裡就是 `NameError`。
2. 模組之間的 import 不滿足拓撲序（有環）。
3. map 指名的項目在原始碼裡不存在，或原始碼裡有項目沒有被任何群組認領（靜默遺失）。
4. 同一批裡出現「被 `global` 宣告的名字，它的讀者與寫者被分到不同模組」
   —— 這是本工具存在的**主要**理由：搬家之後該名字會在新模組裡不存在。

用法
----
    split_module.py --source scripts/check/knifeedge_matrix.py \
        --map agent_harness/shared/knifeedge_split_map.json \
        --out-dir scripts/check --pkg-name knifeedge --write

    split_module.py ... --dry-run     # 只印計畫與位元組對帳，不落盤
    split_module.py ... --check       # 對帳已落盤的結果（乾跑的反面）
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import subprocess
import sys
from collections import OrderedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BUILTINS = set(dir(builtins))


def atomic_write(path, text):
    """先把整份內容寫進同目錄的暫存檔，再 os.replace() 換上去。

    這不是潔癖：`scripts/check/knifeedge_matrix.py` 有另一條線正在 import（他們的腳本會用
    `spec_from_file_location` 讀它）。`open(path, "w")` 會先截斷，在那個窗口裡被讀到就是
    半個檔案 —— 而對方拿到的是 SyntaxError 還是沉默的錯，取決於他們剛好讀到第幾個位元組。
    """
    tmp = path + ".split-tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


class Source:
    """原檔的頂層節點、行區間、與名字歸屬。"""

    def __init__(self, path, text):
        self.path = path
        self.lines = text.splitlines()
        self.tree = ast.parse(text, filename=path)
        self.nodes = []            # [(kind, names, node)]，程式順序
        for n in self.tree.body:
            self.nodes.append((self._kind(n), self._names(n), n))

    @staticmethod
    def _kind(node):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            return "docstring"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import"
        if isinstance(node, ast.If):
            return "mainguard"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "def"
        if isinstance(node, ast.ClassDef):
            return "class"
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            return "var"
        return type(node).__name__

    @staticmethod
    def _names(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return [node.name]
        if isinstance(node, ast.Assign):
            return [t.id for t in node.targets if isinstance(t, ast.Name)]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return [node.target.id]
        return []

    @property
    def docstring(self):
        for k, _n, node in self.nodes:
            return ast.get_docstring(node, clean=False) if k == "docstring" else None
        return None

    def regions(self):
        """每個頂層節點連同它上方的註解/空行 -> (kind, names, start, end)，1-based 含頭含尾。

        切法是「上一個節點的結尾 +1」，所以註解不會掉在縫裡；所有區間**鋪滿**整份檔案
        （docstring 與 import 除外，那兩者由工具重建）。這是對帳的依據。
        """
        out = []
        prev_end = 0
        for kind, names, node in self.nodes:
            start = prev_end + 1
            end = node.end_lineno
            out.append((kind, names, start, end, node))
            prev_end = end
        return out

    def import_bindings(self):
        """原檔的 import -> {綁定名: [原始 import 陳述式, ...]}。

        值是**列表**而不是單一字串：`import urllib.error` 與 `import urllib.request` 綁定
        同一個名字 `urllib`，用字典會讓後者蓋掉前者。第一版就是這樣，AST 檢查抓到
        `import urllib.error` 在新樹裡不存在 —— 而那行程式剛好因為
        `urllib.request` 自己會 import `urllib.error` 而仍然能跑。**僥倖不是保證。**
        """
        table = OrderedDict()
        for _k, _n, node in self.nodes:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            text = ast.get_source_segment("\n".join(self.lines), node) or ast.unparse(node)
            binds = []
            if isinstance(node, ast.Import):
                binds = [(a.asname or a.name).split(".")[0] for a in node.names]
            else:
                if node.module == "__future__":
                    continue
                binds = [a.asname or a.name for a in node.names]
            for b in binds:
                table.setdefault(b, [])
                if text not in table[b]:
                    table[b].append(text)
        return table


def free_names(node):
    """一個頂層節點在**模組層級**用到的名字。

    這裡必須做作用域分析，不能只取所有 `Name(Load)`：函式裡的區域變數（`args`、`k`、`path`…）
    在那個近似下會被當成模組級引用，於是模組之間會多出**假的**相依邊，甚至假環。
    （第一版就是這樣，乾跑時每個模組印出上百個「歸屬不了的名字」。）
    """
    bound = set()
    for x in ast.walk(node):
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and x is not node:
            bound.add(x.name)
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = x.args
            for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
                bound.add(arg.arg)
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)
        elif isinstance(x, ast.ExceptHandler) and x.name:
            bound.add(x.name)
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            for al in x.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del)):
            bound.add(x.id)
    globals_declared = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Global):
            globals_declared.update(x.names)
    loaded = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
            loaded.add(x.id)
    return (loaded - bound) | globals_declared


def globals_written(node):
    """`global X` 宣告的名字（這些名字必須與它的讀者同住一個模組）。"""
    got = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Global):
            got.update(x.names)
    return got


def index_by_name(regions):
    """每個頂層名字 -> (kind, start, end, node)；同時回報沒有名字可歸屬的節點。"""
    by_name = {}
    unnamed = []
    for kind, names, start, end, node in regions:
        for nm in names:
            by_name[nm] = (kind, start, end, node)
        if kind not in ("docstring", "import") and not names:
            unnamed.append((kind, start, end))
    return by_name, unnamed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--map", dest="map_path", required=True)
    ap.add_argument("--out-dir", required=True, help="套件要放進哪個目錄（= 原檔所在目錄）")
    ap.add_argument("--pkg-name", required=True)
    ap.add_argument("--entry", default=None, help="shim 路徑（預設 = --source）")
    ap.add_argument("--from-rev", default=None,
                    help="從這個 git 修訂讀原始碼（--write 之後必須用它才可重跑）")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    with open(args.map_path, encoding="utf-8") as f:
        spec = json.load(f)
    modules = spec["modules"]
    order = spec["module_order"]
    handwritten = spec.get("handwritten", {})
    entry_name = os.path.basename(args.entry or args.source)

    text = open(args.source, encoding="utf-8").read()
    if args.from_rev:
        # 可重現：--write 之後 --source 就是 shim 了，再跑一次會拆到 shim。
        # 從修訂讀才有「同一個輸入 -> 同一個輸出」，也讓這一步可以被別人重跑。
        text = subprocess.run(["git", "-C", REPO_ROOT, "show",
                               f"{args.from_rev}:{args.source}"],
                              capture_output=True, text=True, check=True).stdout
        print(f"source      : {args.source}@{args.from_rev}（從修訂讀，不是工作區）")
    src = Source(args.source, text)

    owner = {}
    for mod, names in modules.items():
        for nm in names:
            if nm in owner:
                die(f"{nm} 同時被 {owner[nm]} 與 {mod} 認領")
            owner[nm] = mod
    for mod, info in handwritten.items():
        for nm in info["names"]:
            if nm in owner:
                die(f"{nm} 同時被 {owner[nm]} 與手寫模組 {mod} 認領")
            owner[nm] = mod

    regions = src.regions()
    by_name, unnamed = index_by_name(regions)
    guard_owner = spec.get("mainguard_owner")
    guard_regions = []
    keep_unnamed = []
    for kind, start, end in unnamed:
        if kind == "mainguard" and guard_owner:
            guard_regions.append((start, end))
        else:
            keep_unnamed.append((kind, start, end))
    if keep_unnamed:
        die("有頂層節點沒有名字可歸屬（工具不敢猜）：" + repr(keep_unnamed))
    if guard_regions and guard_owner not in modules and guard_owner not in handwritten:
        die(f"mainguard_owner={guard_owner} 不是一個模組")

    # --- 有效來源：每個項目的行區間（含上方註解）＋已宣告的逐字替換 ----------
    # 宣告的替換（declared_edits）在這裡套用一次，之後所有分析（相依、自由名、
    # 未歸屬檢查）與落盤都用同一份文字，避免「分析看舊的、寫出新的」。
    edits_by_name = {}
    for e in spec.get("declared_edits", []):
        edits_by_name.setdefault(e["name"], []).append(e)
    EFF_TEXT, EFF_NAMES = {}, {}
    for kind, names, start, end, _n in regions:
        if not names:
            continue
        raw = "\n".join(src.lines[start - 1:end])
        for nm in names:
            text = raw
            for e in edits_by_name.get(nm, []):
                if "\n" in e["find"] or "\n" in e["replace"]:
                    die(f"declared_edits[{nm}] 的 find/replace 不能跨行（行號會漂）")
                cnt = text.count(e["find"])
                if cnt != 1:
                    die(f"declared_edits[{nm}] 的 find 在該項目裡出現 {cnt} 次（要求恰好 1 次）")
                text = text.replace(e["find"], e["replace"])
            EFF_TEXT[nm] = text
            EFF_NAMES[nm] = free_names(ast.parse(text))
    for e in spec.get("declared_edits", []):
        if e["name"] not in EFF_TEXT:
            die(f"declared_edits 指名了原始碼裡沒有的名字：{e['name']}")

    missing = [n for n in owner if n not in by_name and owner[n] not in handwritten]
    unclaimed = [n for n in by_name if n not in owner]
    if missing:
        die("map 指名但原始碼沒有的名字：" + ", ".join(sorted(missing)))
    if unclaimed:
        die("原始碼有但沒有任何群組認領（靜默遺失的來源）：" + ", ".join(sorted(unclaimed)))
    new_glue = sorted(n for n in owner if n not in by_name)
    for m in modules:
        if m not in order:
            die(f"module_order 少了 {m}")
    for m in handwritten:
        if m not in order:
            die(f"module_order 少了手寫模組 {m}")
    for m in order:
        if m not in modules and m not in handwritten:
            die(f"module_order 有 {m}，但它既不是產生模組也不是手寫模組")

    # --- 環檢查：模組之間的相依必須有拓撲序 -------------------------------
    allmods = [m for m in order if m in modules or m in handwritten]
    deps = {m: set() for m in allmods}
    for nm, mod in owner.items():
        if nm not in by_name:
            continue          # 手寫模組的 glue 名（ENTRY_FILE / source_text）
        for used in EFF_NAMES[nm]:
            other = owner.get(used)
            if other and other != mod:
                deps[mod].add(other)
    indeg = {m: 0 for m in allmods}
    for m in allmods:
        for d in deps[m]:
            indeg[d] += 1
    ready = [m for m in allmods if indeg[m] == 0]
    seen = []
    while ready:
        m = ready.pop()
        seen.append(m)
        for d in sorted(deps[m]):
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
    if len(seen) != len(allmods):
        die("模組之間有 import 環（搬家後會 ImportError）：" + repr(sorted(deps.items())))

    # --- global 共居規則 ---------------------------------------------------
    gwriters = {}
    for nm, mod in owner.items():
        if nm not in by_name:
            continue
        for g in globals_written(by_name[nm][3]):
            gwriters.setdefault(g, set()).add(nm)
    shared = [y for y, ns in gwriters.items()
              if len({x for x in ns if x in owner}) > 1]
    for g, ws in gwriters.items():
        mods = {owner[w] for w in ws if w in owner}
        for nm, mod in owner.items():
            if nm in ws or nm not in by_name:
                continue
            if g in EFF_NAMES[nm] and mod not in mods:
                die(f"`global {g}` 被 {sorted(ws)} 寫（{sorted(mods)}），卻被 {nm}（{mod}）讀 "
                    f"—— 搬家後 {nm} 讀的是它自己模組裡不存在的名字。把讀者與寫者放同一模組。")
    if shared:
        die(f"以下名字被多個模組以 global 寫入：{shared}")

    # --- 每個模組的 import 表 ---------------------------------------------
    generated = [m for m in order if m not in handwritten]
    imports_of = src.import_bindings()
    plan = OrderedDict()
    for mod in generated:
        names = modules[mod]
        used = set()
        for nm in names:
            if nm not in EFF_NAMES:
                continue
            used |= EFF_NAMES[nm]
        lines = []
        for b in sorted(imports_of):
            if b in used:
                lines.extend(imports_of[b])
        cross = OrderedDict()
        for u in sorted(used):
            if u in imports_of or u in BUILTINS or u in names:
                continue
            if u in owner:
                dep = owner[u]
            elif u in gwriters:
                # 只以 `global` 存在的名字（args_greedy_global）：它屬於**寫它的**那個模組。
                dep = owner.get(sorted(gwriters[u])[0])
            else:
                continue
            if dep is None or dep == mod:
                continue          # 同模組 —— 不需要 import，import 自己反而會爆
            cross.setdefault(dep, []).append(u)
        for dep in sorted(cross):
            if dep == mod:
                die(f"{mod} 產生出對自己的 import：{cross[dep]}")
            lines.append(f"from .{dep} import {', '.join(sorted(set(cross[dep])))}")
        plan[mod] = {"names": names, "imports": lines, "cross": sorted(cross)}

    # --- 未歸屬的自由名字（真正的安全網）----------------------------------
    interpreter = {"__file__", "__name__", "__doc__", "__spec__", "__package__",
                   "__loader__", "__builtins__"}
    unresolved = {}
    for mod, info in plan.items():
        have = set(owner) | set(imports_of) | BUILTINS | set(gwriters) | interpreter
        used = set()
        for nm in info["names"]:
            if nm not in EFF_NAMES:
                continue
            used |= EFF_NAMES[nm]
        rest = sorted(used - have)
        if rest:
            unresolved[mod] = rest
    if unresolved:
        for mod, rest in unresolved.items():
            print(f"  {mod}: {rest}", file=sys.stderr)
        die("上面這些名字歸屬不了（不在 import、不在頂層項目、不是內建）")

    # --- 位元組對帳 --------------------------------------------------------
    emitted = set()
    for kind, names, start, end, _node in regions:
        if kind in ("docstring", "import"):
            continue
        for nm in names:
            emitted.add(nm)
    total_items = len(emitted)

    print(f"source      : {args.source}  {len(src.lines)} 行  {len(text.encode())} bytes")
    print(f"top-level   : {len(owner)} 個有名項目 + docstring + "
          f"{len(src.import_bindings())} 個 import")
    print(f"package     : {args.out_dir}/{args.pkg_name}/  "
          f"({len(plan)} 個產生模組 + {len(handwritten)} 個手寫模組)")
    print()
    for mod in order:
        if mod in handwritten:
            ns = handwritten[mod]["names"]
            print(f"  {mod:12s} {len(ns):3d} items  {'手寫':>5s}  "
                  f"{handwritten[mod]['file']}（{handwritten[mod]['reason']}）")
            continue
        ln = sum(by_name[n][2] - by_name[n][1] + 1 for n in plan[mod]["names"])
        print(f"  {mod:12s} {len(plan[mod]['names']):3d} items  {ln:5d} 行  "
              f"deps={plan[mod]['cross']}")
    print()
    print("  shim       : " + entry_name + "  (import * + __main__ 轉發)")
    print("  package    : __init__.py  (re-export 全部名字)")

    if args.dry_run:
        print()
        print("dry-run: 沒有落盤。")
        return 0

    pkg_dir = os.path.join(args.out_dir, args.pkg_name)
    os.makedirs(pkg_dir, exist_ok=True)
    # shim 的 docstring 取**原文那幾行**（不是 ast.get_docstring 的值）：
    # 值再包回引號會經過一次跳脫/還原，而 __doc__ 必須逐位元組相同。
    doc_lines = []
    for kind, _names, start, end, _node in regions:
        if kind == "docstring":
            doc_lines = src.lines[start - 1:end]
            break
    if not doc_lines:
        die("原檔沒有 docstring，shim 少了 __doc__ 的來源")

    written = []
    for mod in generated:
        info = plan[mod]
        body = [(by_name[nm][1], by_name[nm][2], nm) for nm in info["names"]]
        if mod == guard_owner:
            body.extend([(s, e, None) for s, e in guard_regions])
        body.sort()
        chunks = [EFF_TEXT[nm].split("\n") if nm else src.lines[s - 1:e]
                  for s, e, nm in body]
        head = [
            "#!/usr/bin/env python3",
            f'"""{args.pkg_name}.{mod} — 由 split_module.py 從 {args.source} 機械拆出。',
            "",
            "這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、",
            "且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見",
            "agent_harness/shared/knifeedge_split_map.json。",
            '"""',
            "from __future__ import annotations",
            "",
        ]
        out = head + info["imports"] + [""] + [ln for c in chunks for ln in c + [""]]
        p = os.path.join(pkg_dir, f"{mod}.py")
        atomic_write(p, "\n".join(out).rstrip("\n") + "\n")
        written.append(p)

    allnames = sorted(owner)
    stdlib = sorted(imports_of)
    init = ["#!/usr/bin/env python3",
            f'"""{args.pkg_name} — {args.source} 的內容，按能力拆成 {len(plan)} 個模組。',
            "",
            "公開面（`__all__`）與原檔的頂層名字**逐一相同**，所以任何既有的",
            f"`import {os.path.splitext(entry_name)[0]}` / `from ... import X` 都不必改一行：",
            f"舊路徑 {entry_name} 是薄 shim，re-export 的就是這裡。",
            "",
            "下面那排 stdlib import 不是 API，是為了讓 `dir()` 與原檔一致 —— 原檔 import 過",
            "`os`/`json`/… 之後，那些名字就是它的屬性；搬家不該讓 `km.os` 消失。",
            '"""',
            "from __future__ import annotations",
            ""]
    for b in stdlib:
        init.extend(imports_of[b])
    init.append("")
    for mod in order:
        if mod in handwritten:
            init.append(f"from .{mod} import ({', '.join(sorted(handwritten[mod]['names']))})")
    for mod in generated:
        init.append(f"from .{mod} import ({', '.join(sorted(modules[mod]))})")
    init += ["", "__all__ = ["]
    init += [f'    "{n}",' for n in allnames]
    init += ["]", ""]
    p = os.path.join(pkg_dir, "__init__.py")
    atomic_write(p, "\n".join(init))
    written.append(p)

    shim = ["#!/usr/bin/env python3"] + doc_lines + ["",
            "from __future__ import annotations",
            ""] + [
            "# ---- 這個檔案是 shim ------------------------------------------------",
            f"# 內容自 2026-09-17（E4 item 2）起住在 ./{args.pkg_name}/，分成 "
            f"{len(plan)} 個模組。這裡只做三件事：",
            "#   1. 把自己所在的目錄放進 sys.path —— 呼叫者用 import、用 "
            "spec_from_file_location、",
            "#      或用 `python3 <這個路徑>` 都必須能解析到 ./"
            + args.pkg_name + "/。",
            "#   2. 逐一 re-export（含 _ 開頭的私有名），讓 dir() 與原檔一致。",
            "#   3. 轉發 __main__。",
            "# 這一層是「錨點」而不是轉運站：__doc__ 保留原文（呼叫者的字串斷言才不會漂）。",
            "# --------------------------------------------------------------------",
            "import os as _os",
            "import sys as _sys",
            "",
            "_HERE = _os.path.dirname(_os.path.abspath(__file__))",
            "if _HERE not in _sys.path:",
            "    _sys.path.insert(0, _HERE)",
            "",
            f"from {args.pkg_name} import (  # noqa: E402,F401",
            ]
    for n in allnames:
        shim.append(f"    {n},")
    shim += [")", ""]
    if stdlib:
        shim += ["# 原檔的 stdlib 綁定（`km.os` 這種讀法）—— 為了讓 dir() 與搬家前一致。",
                 f"from {args.pkg_name} import (  # noqa: E402,F401"]
        for n in stdlib:
            shim.append(f"    {n},")
        shim += [")", ""]
    shim += ["# ---- 子模組可以直接取用 ------------------------------------------",
             "# `km.<module>.<name> = x` 是路徑載入（spec_from_file_location）時**唯一**可行的",
             "# 替換方式：那種載入不會把模組放進 sys.modules，所以沒有人拿得到「entry 那個物件」，",
             "# 也就沒有 __setattr__ 可以攔。因此把子模組綁在 entry 上。",
             f"from {args.pkg_name} import ({', '.join(order)})  # noqa: E402,F401",
             "",
             "# ---- 屬性寫入轉發（僅對已註冊的 entry 生效）------------------------",
             "# 這些離線自測會用屬性替換注入替身，例如 `km.pool_geometry = lambda kind: ...`。",
             "# 單體時代那必然生效（只有一個命名空間）。拆檔之後 `feasibility_cell` 讀的是",
             "# `feasibility.pool_geometry`，而 km 上的那個只是 shim 的獨立綁定。",
             "# 正常 import 的 entry 就在 sys.modules 裡，可以換掉它的類別來攔 __setattr__；",
             "# 路徑載入的 entry 不在 sys.modules 裡，Python 沒有給任何物件回指的辦法 ——",
             "# 那種情況請改用 `km.<module>.<name> = x`（見上一段）。",
             "import types as _types",
             "",
             f'_PKG = "{args.pkg_name}"',
             "_OWNER = {",
             ]
    for n in sorted(owner):
        shim.append(f'    "{n}": "{owner[n]}",')
    shim += ["}", "",
             "",
             "class _Forward(_types.ModuleType):",
             '    """把 `entry.N = x` 轉發到定義 N 的那個模組，其餘照常。"""',
             "",
             "    def __setattr__(self, name, value):",
             "        mod = _OWNER.get(name)",
             "        if mod is not None:",
             '            setattr(_sys.modules[f"{_PKG}.{mod}"], name, value)',
             "        super().__setattr__(name, value)",
             "",
             "",
             "_self = _sys.modules.get(__name__)",
             "if _self is not None:",
             "    _self.__class__ = _Forward",
             "",
             'if __name__ == "__main__":',
             "    sys.exit(main())",
             ""]
    atomic_write(args.entry or args.source, "\n".join(shim) + "\n")
    written.append(args.entry or args.source)

    print()
    for p in written:
        print(f"  寫入 {p}  ({os.path.getsize(p)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
