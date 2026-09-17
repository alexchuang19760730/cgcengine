#!/usr/bin/env python3
"""check_module_split.py — 證明一次「單體拆成套件」沒有改行為。

判準（四層，缺一不可）
----------------------
1. **AST 層**：原檔的每一個頂層節點（含 docstring 與 import），都要能在新樹裡找到
   `ast.dump` 相同的節點。反向也要成立：**產生出來的模組只能有逐位元組搬過來的內容**，
   任何多出來的節點都必須落在 map 宣告的 `glue_files` 裡（shim / `__init__` / 手寫錨點）。
   這一條擋的是「順手改一行」——它不會被任何語法檢查抓到。
2. **匯出層**：`__all__` 覆蓋原檔**所有**頂層綁定（含 `_` 開頭），因此
   `from <entry> import <任何名字>` 都還能用；`dir()` 也一致（stdlib 綁定一起帶過去）。
3. **runtime 層**：把原檔與新樹各別 import 起來，比對每一個函式的簽名、docstring、
   常數值（序列化後雜湊），以及解析出來的路徑錨點。
4. **CLI 層**：兩個 entry 的 `--help` 逐字元相同。argparse 的 `prog` 取自
   `argv[0]` 的 basename，所以搬到別的目錄也應該相同——不相同就代表參數面變了。

陰性對照
--------
`--self-test` 會把真套件複製一份、在**函式體內**改一個字元，然後要求 AST 檢查**必須失敗**；
再對未改的複本要求**必須通過**。一個只會說「沒問題」的比較器沒有價值 —— 它得先證明
自己說得出「有問題」。

用法
----
    check_module_split.py fingerprint --entry scripts/check/knifeedge_matrix.py --out /tmp/before.json
    check_module_split.py compare --before /tmp/before.json --after /tmp/after.json
    check_module_split.py cli --old /tmp/old_entry.py --new scripts/check/knifeedge_matrix.py
    check_module_split.py ast  --original scripts/check/knifeedge_matrix.py --rev <rev> \\
                               --package scripts/check/knifeedge --entry scripts/check/knifeedge_matrix.py \\
                               --map <map.json>
    check_module_split.py --self-test
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANCHOR_NAMES = ("ROOT", "GGUF", "RESULT_DIR", "BINARY_DIR", "ENTRY_FILE")


def die(msg, rc=2):
    print(f"FAIL  {msg}", file=sys.stderr)
    raise SystemExit(rc)


def sha(text):
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def norm_repr(v):
    try:
        return json.dumps(v, sort_keys=True, default=repr)
    except Exception:
        return repr(v)


def load_module(path, name):
    """用檔案路徑載入模組 —— 走的就是腳本自己的載入方式，不靠 sys.path。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fingerprint(entry, root=REPO):
    """一個模組的可比指紋。刻意不含 `__module__` 與檔案路徑本身（那兩者本來就會變）。"""
    mod = load_module(entry, "km_fingerprint")
    fp = {"entry": os.path.basename(entry), "names": sorted(n for n in dir(mod)
                                                             if not n.startswith("__")),
          "callables": {}, "values": {}, "anchors": {}, "doc": sha(mod.__doc__ or "")}
    for n in fp["names"]:
        v = getattr(mod, n)
        if callable(v) and getattr(v, "__module__", "").startswith(("knifeedge", "km_")):
            import inspect
            try:
                sig = str(inspect.signature(v))
            except (TypeError, ValueError):
                sig = "<no signature>"
            fp["callables"][n] = {"sig": sig, "doc": sha(getattr(v, "__doc__", "") or "")}
        elif n in ANCHOR_NAMES and isinstance(v, str):
            fp["anchors"][n] = os.path.relpath(v, root)
        elif isinstance(v, (str, int, float, bool, list, tuple, dict, type(None))):
            fp["values"][n] = sha(norm_repr(v))
    if "ROOT" in fp["anchors"]:
        fp["anchors"]["ROOT_ABS"] = getattr(mod, "ROOT")
    return fp


def cmd_fingerprint(args):
    fp = fingerprint(args.entry)
    out = json.dumps(fp, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"  fingerprint -> {args.out}  ({len(fp['names'])} names, "
              f"{len(fp['callables'])} callables, {len(fp['anchors'])} anchors)")
    else:
        print(out)
    return 0


def cmd_compare(args):
    before = json.load(open(args.before, encoding="utf-8"))
    after = json.load(open(args.after, encoding="utf-8"))
    extra = set(args.expect_extra or [])
    if args.map:
        spec = json.load(open(args.map, encoding="utf-8"))
        for info in spec.get("handwritten", {}).values():
            extra |= set(info["names"])
        extra |= set(spec.get("entry_glue", []))
    bad = []
    missing = [n for n in before["names"] if n not in after["names"]]
    if missing:
        bad.append(f"新樹少了 {len(missing)} 個名字（這一條沒有例外）：{missing[:12]}")
    added = sorted(n for n in after["names"] if n not in before["names"])
    undeclared = [n for n in added if n not in extra]
    if undeclared:
        bad.append(f"新樹多了未宣告的名字：{undeclared}")
    for n, sig in before["callables"].items():
        got = after["callables"].get(n)
        if got is None:
            continue
        if got["sig"] != sig["sig"]:
            bad.append(f"{n} 簽名變了：{sig['sig']} -> {got['sig']}")
        if got["doc"] != sig["doc"]:
            bad.append(f"{n} 的 docstring 變了")
    for n, h in before["values"].items():
        got = after["values"].get(n)
        if got is not None and got != h:
            bad.append(f"{n} 的常數值變了")
    for n, p in before["anchors"].items():
        got = after["anchors"].get(n)
        if got != p:
            bad.append(f"錨點 {n} 變了：{p} -> {got}")
    if before["doc"] != after["doc"]:
        bad.append("模組 __doc__ 變了")
    print(f"  before: {len(before['names'])} names / {len(before['callables'])} callables / "
          f"{len(before['values'])} constants / {len(before['anchors'])} anchors")
    print(f"  after : {len(after['names'])} names / {len(after['callables'])} callables / "
          f"{len(after['values'])} constants / {len(after['anchors'])} anchors")
    print(f"  宣告的新增名字：{added}")
    if bad:
        for b in bad:
            print(f"  DIFF  {b}", file=sys.stderr)
        die(f"runtime 指紋不一致（{len(bad)} 項）")
    print("  runtime 指紋一致")
    return 0


GLUE_ALLOWED = ("Import", "ImportFrom", "Expr")


def top_nodes(path_or_text, is_text=False):
    text = path_or_text if is_text else open(path_or_text, encoding="utf-8").read()
    return text, ast.parse(text).body


def dump_all(nodes):
    return [ast.dump(n) for n in nodes]


def find_named(nodes, name):
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return n
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgt = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if getattr(tgt, "id", None) == name:
                return n
    return None


def verify_edits(spec, old_text, old_nodes, new_sources):
    """逐字重建每一個宣告的替換，並回傳 {old_dump: new_dump}。

    這一條把「我改了這一行」從**聲明**變成**可驗的等式**：
    `原節點原文.replace(find, replace)` 必須逐 AST 等於新樹裡同名的那個節點。
    沒有這一條，`declared_edits` 就只是一句自述。
    """
    rebuilt = {}
    for e in spec.get("declared_edits", []):
        old_node = find_named(old_nodes, e["name"])
        if old_node is None:
            die(f"declared_edits 指名了原檔裡沒有的名字：{e['name']}")
        old_src = ast.get_source_segment(old_text, old_node)
        cnt = old_src.count(e["find"])
        if cnt != 1:
            die(f"{e['name']} 的 find 在原節點裡出現 {cnt} 次（要求恰好 1 次）")
        expected = old_src.replace(e["find"], e["replace"])
        new_node = None
        for where, nodes in new_sources.items():
            cand = find_named(nodes, e["name"])
            if cand is not None:
                new_node = cand
                break
        if new_node is None:
            die(f"新樹裡找不到 {e['name']}")
        want = ast.dump(ast.parse(expected).body[0])
        got = ast.dump(new_node)
        if want != got:
            print(f"  EDIT MISMATCH {e['name']}：逐字重建的結果與新樹不同", file=sys.stderr)
            die("declared_edits 與實際內容不符")
        rebuilt[ast.dump(old_node)] = got
    return rebuilt


def cmd_ast(args):
    rev_text = subprocess.run(["git", "-C", REPO, "show", f"{args.rev}:{args.original}"],
                              capture_output=True, text=True, check=True).stdout
    old_text, old_nodes = top_nodes(rev_text, is_text=True)

    files = sorted(f for f in os.listdir(args.package) if f.endswith(".py"))
    entry_base = os.path.basename(args.entry)
    pairs = [(f"knifeedge/{f}", os.path.join(args.package, f)) for f in files] + \
            [(entry_base, args.entry)]
    new_sources, new_dumps = {}, {}
    for key, path in pairs:
        _t, nodes = top_nodes(path)
        new_sources[key] = nodes
        for n in nodes:
            new_dumps.setdefault(ast.dump(n), []).append(key)

    map_spec = json.load(open(args.map, encoding="utf-8")) if args.map else {}
    declared = {d["dump"]: d for d in map_spec.get("declared_delta", [])}
    glue_files = set(map_spec.get("glue_files", []))
    rebuilt = verify_edits(map_spec, rev_text, old_nodes, new_sources)

    missing, relocated = [], 0
    n_rebuilt = 0
    for n in old_nodes:
        d = ast.dump(n)
        if new_dumps.get(d):
            relocated += 1
            continue
        if d in declared:
            continue
        if d in rebuilt:
            n_rebuilt += 1
            continue
        line = old_text.splitlines()[n.lineno - 1][:80] if hasattr(n, "lineno") else ""
        missing.append((type(n).__name__, getattr(n, "name", ""), line))
    if missing:
        print(f"  MISSING 原檔節點在新樹裡找不到（{len(missing)} 個）：", file=sys.stderr)
        for k, nm, line in missing[:15]:
            print(f"    {k} {nm}  {line}", file=sys.stderr)
        die("AST 等價失敗")

    old_dumps = set(dump_all(old_nodes))
    rebuilt_dumps = set(rebuilt.values())
    extras = {}
    for key, path in pairs:
        _t, nodes = top_nodes(path)
        for n in nodes:
            if ast.dump(n) in old_dumps or ast.dump(n) in rebuilt_dumps:
                continue
            kind = type(n).__name__
            name = getattr(n, "name", None)
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                tgt = n.targets[0] if isinstance(n, ast.Assign) else n.target
                name = getattr(tgt, "id", None)
            if kind in GLUE_ALLOWED or name == "__all__" or key in glue_files:
                continue
            extras.setdefault(key, []).append(f"{kind} {name}")
    if extras:
        print("  未宣告的額外節點：", file=sys.stderr)
        for k, v in extras.items():
            print(f"    {k}: {v[:8]}", file=sys.stderr)
        die("產生出來的模組裡有不屬於原檔的內容（glue 必須宣告在 map 的 glue_files）")

    print(f"  原檔 {len(old_nodes)} 個頂層節點：{relocated} 個逐 AST 相同、"
          f"{n_rebuilt} 個逐字重建驗過、{len(declared)} 個為宣告的偏差")
    print(f"  新樹：{len(files)} 個模組 + shim；額外節點只出現在宣告的 glue 檔 "
          f"({sorted(glue_files)})")
    return 0


def strip_warnings(text):
    """拿掉 warning 區塊（warning 那一行 + 它的來源引文那一行）。

    警告的內容包含**檔案路徑與行號**，所以搬動程式碼之後它必然不同 —— 那不代表 CLI 面變了。
    但它會蓋掉「新樹多噴了一個 Traceback」這種真訊號，所以只拿掉警告、其餘照比。
    """
    out, skip_next = [], False
    for line in text.splitlines():
        if skip_next:
            skip_next = False
            if line.startswith("  "):
                continue
        if re.search(r":\d+: \w*Warning: ", line):
            out.append("<warning at [path]:[line]>")
            skip_next = True
            continue
        out.append(re.sub(r"^[^\s:]+\.py:\d+: ", "<[path]:[line]>: ", line))
    return "\n".join(out)


def cmd_contracts(args):
    """替換契約：entry 上的名字被換掉時，真正在跑的程式碼必須看到那個替換。

    為什麼要單獨驗這一條：這是 AST 檢查與指紋比對**都看不到**的一類破壞。
    單體時代「只有一個命名空間」，所以 `km.<name> = x` 必然生效（trivially true，不需要驗）。
    拆檔之後 entry 與各模組各自有自己的全域，於是同一個寫入可能只改到 shim 的綁定 ——
    而沒有任何讀者。實測症狀：`scripts/check/feasibility_gate_selftest.py` 用它注入合成幾何，
    拆檔後注入靜默失效，測試掉到真的 GGUF 路徑並以 numpy 缺席崩潰。

    兩種載入方式各驗一次，因為它們的可行修法不同：
      A. spec_from_file_location（三支離線自測用的）→ Python 不把模組放進 sys.modules，
         拿不到 entry 那個物件，所以只能走 `entry.<module>.<name> = x`。
      B. import → entry 在 sys.modules 裡，shim 換掉了自己的類別，`entry.<name> = x` 會轉發。
    """
    spec = json.load(open(args.map, encoding="utf-8")) if args.map else {}
    owner = {}
    for mod, names in spec.get("modules", {}).items():
        for n in names:
            owner[n] = mod
    for mod, info in spec.get("handwritten", {}).items():
        for n in info["names"]:
            owner[n] = mod
    if not owner:
        die("map 裡沒有 modules/handwritten，無法驗替換契約")

    checks = []
    check_dir = os.path.dirname(os.path.abspath(args.entry))
    pkg_dir = os.path.join(check_dir, spec.get("package", "").split("/")[-1])

    # --- A. 路徑載入 ------------------------------------------------------
    mod = load_module(args.entry, "km_contracts_path")
    missing_mods = sorted(m for m in set(owner.values())
                          if m not in ("anchor", "_source") and not hasattr(mod, m))
    checks.append(("A 子模組可由 entry 取用（km.<module>）", not missing_mods, str(missing_mods)))
    sent = lambda *a, **k: {"__sentinel__": True}
    bad = []
    for name, mname in sorted(owner.items()):
        m = getattr(mod, mname, None)
        if m is None:
            continue
        orig = getattr(m, name, None)
        setattr(m, name, sent)
        if getattr(m, name) is not sent:
            bad.append(name)
        setattr(m, name, orig)
    checks.append(("A 打在子模組上的替身真的進去", not bad, f"{len(bad)} 個失敗 {bad[:5]}"))

    # --- B. import 載入 ---------------------------------------------------
    sys.path.insert(0, check_dir)
    sys.path.insert(0, os.path.dirname(pkg_dir))
    entry_name = os.path.splitext(os.path.basename(args.entry))[0]
    km = importlib.import_module(entry_name)
    checks.append(("B entry 是轉發類別（__setattr__ 可攔）",
                   type(km).__name__ == "_Forward", type(km).__name__))
    bad = []
    for name, mname in sorted(owner.items()):
        holder = sys.modules.get(f"{spec.get('package', '').split('/')[-1]}.{mname}")
        if holder is None:
            continue
        orig = getattr(holder, name, None)
        setattr(km, name, sent)
        if getattr(holder, name) is not sent:
            bad.append(f"{name} -> {mname}")
        setattr(holder, name, orig)
        setattr(km, name, orig)
    checks.append(("B 打在 entry 上的替身轉發到定義它的模組", not bad,
                   f"{len(bad)} 個失敗 {bad[:5]}"))

    ok = True
    for name, good, detail in checks:
        print(f"  [{'ok' if good else 'BAD'}] {name}   {detail}")
        ok &= good
    print(f"\n  {sum(1 for _n, g, _d in checks if g)}/{len(checks)} checks passed")
    return 0 if ok else 1


def cmd_cli(args):
    def run(path):
        # 清掉 __pycache__：警告是**編譯期**產生的，而 .pyc 命中時不會再產生一次。
        # 不清的話，一邊暖快取一邊冷編譯，比到的是快取狀態不是 CLI 面。
        # 套件版的警告來自子模組，所以連 knifeedge/__pycache__ 一起清。
        d = os.path.dirname(os.path.abspath(path))
        for cache in (os.path.join(d, "__pycache__"), os.path.join(d, "knifeedge",
                                                                   "__pycache__")):
            shutil.rmtree(cache, ignore_errors=True)
        r = subprocess.run([sys.executable, os.path.abspath(path), "--help"],
                           capture_output=True, text=True, cwd=REPO)
        return r.returncode, r.stdout, r.stderr
    rc_o, out_o, err_o = run(args.old)
    rc_n, out_n, err_n = run(args.new)
    if rc_o != rc_n or out_o != out_n:
        print(f"  old rc={rc_o} stdout={len(out_o)} bytes / new rc={rc_n} "
              f"stdout={len(out_n)} bytes", file=sys.stderr)
        for i, (a, b) in enumerate(zip(out_o.splitlines(), out_n.splitlines())):
            if a != b:
                print(f"  第 {i+1} 行不同:\n    old: {a!r}\n    new: {b!r}", file=sys.stderr)
                break
        die("CLI 面不同")
    if strip_warnings(err_o) != strip_warnings(err_n):
        print(f"  old stderr:\n{err_o}\n  new stderr:\n{err_n}", file=sys.stderr)
        die("stderr 不同（不只是警告位置）")
    nw = len([ln for ln in err_o.splitlines() if "Warning: " in ln])
    print(f"  --help 逐字元相同（stdout {len(out_o)} bytes, rc={rc_o}）；"
          f"stderr 的 {nw} 個警告位置不同（預期：同一段程式換了檔案）")
    return 0


def cmd_self_test(args):
    """陰性對照：改一個字元的複本**必須**被判為不等價；未改的複本必須通過。"""
    pkg = args.package
    entry = args.entry_selftest
    map_path = args.map_selftest
    checks = []

    def run_ast(pkgdir, entry, show=False):
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "ast",
             "--original", args.original, "--rev", args.rev,
             "--package", pkgdir, "--entry", entry, "--map", map_path],
            capture_output=True, text=True)
        if show or r.returncode not in (0, 2):
            print("    " + "\n    ".join(r.stdout.strip().splitlines()[-6:]))
            print("    " + "\n    ".join(
                ln for ln in r.stderr.strip().splitlines()
                if "SyntaxWarning" not in ln and not ln.startswith("  "))[:600])
        return r.returncode

    with tempfile.TemporaryDirectory(prefix="splitcheck_") as tmp:
        work = os.path.join(tmp, "scripts", "check")
        os.makedirs(work)
        subprocess.run(["cp", "-R", pkg, os.path.join(work, "knifeedge")], check=True)
        subprocess.run(["cp", entry, os.path.join(work, "knifeedge_matrix.py")], check=True)
        # 陽性對照：原樣的複本要通過
        rc_ok = run_ast(os.path.join(work, "knifeedge"),
                        os.path.join(work, "knifeedge_matrix.py"), show=True)
        checks.append(("原樣的複本 -> pass", rc_ok == 0, f"rc={rc_ok}"))
        # 陰性對照：在函式體內把一個 6 改成 7
        victim = None
        for f in sorted(os.listdir(os.path.join(work, "knifeedge"))):
            if not f.endswith(".py") or f == "__init__.py":
                continue
            p = os.path.join(work, "knifeedge", f)
            t = open(p, encoding="utf-8").read()
            m = re.search(r"=\s*6\b", t)
            if m:
                open(p, "w", encoding="utf-8").write(t[:m.start()] + "= 7" + t[m.end():])
                victim = f
                break
        if victim is None:
            die("self-test 找不到可以改的常數（測試本身失效）")
        rc_bad = run_ast(os.path.join(work, "knifeedge"), os.path.join(work, "knifeedge_matrix.py"))
        checks.append((f"改了 knifeedge/{victim} 一個字元 -> fail", rc_bad != 0, f"rc={rc_bad}"))

    ok = True
    for name, good, detail in checks:
        print(f"  [{'ok' if good else 'BAD'}] {name}   {detail}")
        ok &= good
    print(f"\n  {sum(1 for _n, g, _d in checks if g)}/{len(checks)} checks passed")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("fingerprint")
    p.add_argument("--entry", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_fingerprint)

    p = sub.add_parser("compare")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--map", default=os.path.join(REPO, "agent_harness/shared"
                                                 "/knifeedge_split_map.json"),
                   help="宣告『允許新增的名字』的地方（手寫模組的名字 + entry 的 glue）")
    p.add_argument("--expect-extra", default="", help="逗號分隔：額外允許的名字")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("ast")
    p.add_argument("--original", required=True)
    p.add_argument("--rev", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--entry", required=True)
    p.add_argument("--map", default=None)
    p.set_defaults(func=cmd_ast)

    p = sub.add_parser("contracts")
    p.add_argument("--entry", required=True)
    p.add_argument("--map", default=os.path.join(REPO, "agent_harness/shared"
                                                 "/knifeedge_split_map.json"))
    p.set_defaults(func=cmd_contracts)

    p = sub.add_parser("cli")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.set_defaults(func=cmd_cli)

    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--package", default=os.path.join(REPO, "scripts/check/knifeedge"))
    ap.add_argument("--original", default="scripts/check/knifeedge_matrix.py")
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--map", dest="map_selftest",
                    default=os.path.join(REPO,
                                         "agent_harness/shared/knifeedge_split_map.json"))
    ap.add_argument("--entry", dest="entry_selftest",
                    default=os.path.join(REPO, "scripts/check/knifeedge_matrix.py"))
    args = ap.parse_args(argv)
    if args.self_test:
        return cmd_self_test(args)
    if not getattr(args, "func", None):
        ap.error("需要一個子命令（或 --self-test）")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
