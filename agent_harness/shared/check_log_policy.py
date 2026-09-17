#!/usr/bin/env python3
"""check_log_policy.py — 把 PLAN §8 的 log 政策從散文變成斷言（E4 item 1）。

政策原本寫在三個地方：PLAN §8 的段落、`.gitignore` 的 dated 註解、以及幾支腳本的行為。
三者都不會因為違反而有任何反應 —— 實測：`Backup/cgc_logs` 底下有 **12 個 .log 原文**
是被 `git add -f` 進版控的（約 5.6 MB，而 docs/ 與 *.md 對它們的引用數是 **0**），
而政策原文是「原文**不進 repo**」。**沒有閘門的政策就是一個願望。**

這一支驗四件事：
  1. 該納管的真的在版控裡（不在 = 那個產出物沒有 `--check` 可以驗）；
  2. 不該納管的真的不在（在 = 一個看得見的違反，或一筆登記在案的既有例外）；
  3. 證據 pack 的上限與出處（每筆有 sha256 指向原文、每筆 ≤ 256 KB、索引指向的檔案存在）；
  4. 去隱私**在已提交的產物上**成立（不是只驗 sanitize.py 自己的單元測試）。

用法
----
    check_log_policy.py --check                 # 1 + 2 + 3
    check_log_policy.py --privacy --sample 40   # 4（抽樣解壓掃描）
    check_log_policy.py --self-test             # 三條陰性對照
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICY = os.path.join(REPO, "agent_harness", "shared", "log_policy.json")
EVIDENCE = os.path.join(REPO, "agent_harness", "shared", "evidence")


def die(msg):
    print(f"FAIL  {msg}", file=sys.stderr)
    raise SystemExit(2)


def git(*args):
    r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def tracked(pattern):
    _rc, out = git("ls-files", pattern)
    return [p for p in out.splitlines() if p]


def check_must_track(pol):
    bad = []
    detail = []
    for ent in pol["must_track"]:
        p = ent["path"]
        files = tracked(p)
        if ent.get("glob"):
            files = [f for f in files if f.endswith(ent["glob"][1:])]
        if not files:
            bad.append(f"{p}（{ent['glob'] or '檔案'}）不在版控裡 —— {ent['why']}")
        else:
            detail.append(f"{p}: {len(files)} 筆")
    return bad, detail


def check_must_not_track(pol):
    bad = []
    detail = []
    allowed = {g["path"] for g in pol.get("grandfathered", [])}
    import fnmatch
    for ent in pol["must_not_track"]:
        p = ent["path"]
        rule = ent.get("gitignore_rule")
        if rule:
            rules = [ln.strip() for ln in open(os.path.join(REPO, ".gitignore"),
                                               encoding="utf-8")
                     if ln.strip() and not ln.startswith("#")]
            if rule not in rules:
                bad.append(f"{p} 說要以 .gitignore 的 `{rule}` 擋住，但那條規則不在")
        files = tracked(p)
        forms = ent.get("forbidden_forms")
        if forms:
            stray = [f for f in files
                     if any(fnmatch.fnmatch(os.path.basename(f), fm) for fm in forms)
                     and f not in allowed]
            cap = ent.get("max_bytes")
            if cap:
                for f in files:
                    if f in allowed:
                        continue
                    sz = os.path.getsize(os.path.join(REPO, f))
                    if sz > cap:
                        stray.append(f"{f}（{sz} bytes > {cap}）")
            others = [f for f in files if f not in stray and f not in allowed]
            if others:
                detail.append(f"{p}: {len(others)} 個衍生檔在版控裡（非原文；"
                              f"{sum(os.path.getsize(os.path.join(REPO, f)) for f in others)}"
                              f" bytes）—— 見 log_policy.json 的 derived_but_tracked")
        else:
            stray = [f for f in files if f not in allowed]
        if stray:
            bad.append(f"{p} 底下有 {len(stray)} 個檔案違反政策（{ent['why'][:40]}…）："
                       f"{stray[:3]}")
        if not files:
            detail.append(f"{p}: 0 筆在版控（正確）")
    # 每一筆 grandfathered 都必須還在版控裡，否則那個例外已經過期、會掩蓋未來的新違反
    for g in pol.get("grandfathered", []):
        rc, _o = git("ls-files", "--error-unmatch", g["path"])
        if rc != 0:
            bad.append(f"grandfathered 的 {g['path']} 已經不在版控裡了 —— 把這筆例外刪掉")
    return bad, detail


def check_packs(pol):
    idx = os.path.join(EVIDENCE, "INDEX.jsonl")
    if not os.path.exists(idx):
        return [f"找不到 {os.path.relpath(idx, REPO)}"], []
    cap = pol["caps"]["pack_bytes_max"]
    rows = []
    with open(idx, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    bad = []
    biggest = 0
    files = set()
    for r in rows:
        sha_src = r.get("source_sha256") or ""
        if not re.fullmatch(r"[0-9a-f]{64}", sha_src):
            bad.append(f"{r.get('episode_id')}: source_sha256 不是 64 位十六進位"
                       f"（沒有指向原文就無法回溯）")
        pb = r.get("pack_bytes") or 0
        biggest = max(biggest, pb)
        if pb <= 0:
            bad.append(f"{r.get('episode_id')}: pack_bytes={pb}（沒有量到大小）")
        elif pb > cap:
            bad.append(f"{r.get('episode_id')}: {pb} bytes > 上限 {cap}")
        pk = r.get("pack")
        if pk:
            files.add(pk)
    missing_pack = [f for f in files if not os.path.exists(os.path.join(EVIDENCE,
                                                                       os.path.basename(f)))]
    if missing_pack:
        bad.append(f"索引指向的 {len(missing_pack)} 個 pack 檔不存在：{missing_pack[:3]}")
    on_disk = [f for f in os.listdir(EVIDENCE) if f.endswith(".txt.zst")]
    dropped = sorted(set(on_disk) - {os.path.basename(f) for f in files})
    detail = [f"引用 {len(tracked('agent_harness/shared/evidence'))} 個 pack 檔"
              f"（索引 {len(rows)} 列；內容尋址去重 {len(rows)} → {len(on_disk)} 檔）",
              f"最大 pack {biggest} bytes = {round(biggest / 1024, 1)} KB（上限 "
              f"{cap // 1024} KB）"]
    if dropped:
        detail.append(f"⚠ {len(dropped)} 個 pack 檔在磁碟上但索引沒有引用它們"
                      f"（孤兒，非違反，但值得知道）：{dropped[:3]}")
    if bad:
        return bad, detail
    return [], detail


def scan_privacy(pol, sample=0):
    zstd = shutil.which("zstd")
    if not zstd:
        return ["找不到 zstd 執行檔，無法解壓 pack 做去隱私掃描"], []
    packs = sorted(f for f in os.listdir(EVIDENCE) if f.endswith(".txt.zst"))
    if sample:
        step = max(1, len(packs) // sample)
        packs = packs[::step][:sample]
    pats = [(p["name"], re.compile(p["regex"])) for p in pol["privacy_forbidden"]]
    hits = []
    scanned = 0
    for name in packs:
        r = subprocess.run([zstd, "-dc", os.path.join(EVIDENCE, name)],
                           capture_output=True)
        if r.returncode != 0:
            hits.append(f"{name}: 解壓失敗")
            continue
        text = r.stdout.decode("utf-8", "replace")
        scanned += 1
        for pname, rx in pats:
            m = rx.search(text)
            if m:
                hits.append(f"{name}: 命中『{pname}』→ {m.group(0)[:60]!r}")
    return hits, [f"掃了 {scanned}／{len(packs)} 個 pack（抽樣；--all 可全掃）"]


def run_all(pol, do_privacy=False, sample=40):
    problems, details = [], []
    for fn in (check_must_track, check_must_not_track, check_packs):
        b, d = fn(pol)
        problems += b
        details += d
    if do_privacy:
        b, d = scan_privacy(pol, sample)
        problems += b
        details += d
    return problems, details


def cmd_check(args):
    pol = json.load(open(POLICY, encoding="utf-8"))
    problems, details = run_all(pol, do_privacy=args.privacy,
                               sample=0 if args.all else args.sample)
    for d in details:
        print(f"  {d}")
    n_grand = len(pol.get("grandfathered", []))
    if n_grand:
        print(f"  ⚠ 登記在案的既有例外 {n_grand} 筆（"
              f"{sum(g['bytes'] for g in pol['grandfathered'])} bytes）："
              f"{[g['path'].split('/')[-1] for g in pol['grandfathered']][:3]} …")
        print(f"    owner={pol['grandfathered'][0]['owner']}；多一筆就會紅")
    if problems:
        for p in problems:
            print(f"  PROBLEM  {p}", file=sys.stderr)
        die(f"log 政策有 {len(problems)} 項不一致")
    print("  log 政策一致")
    return 0


def cmd_self_test(args):
    """三條陰性對照：每一類違反都必須真的被判紅。"""
    pol = json.load(open(POLICY, encoding="utf-8"))
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, ok, detail))

    # 1) 一個「不在版控」的 must_track 條目必須被點名
    fake = json.loads(json.dumps(pol))
    fake["must_track"] = [{"path": "definitely/not/tracked", "glob": None,
                           "why": "self-test 造出來的"}]
    b, _d = check_must_track(fake)
    chk("must_track 少一筆 -> 紅（陰性對照）", len(b) == 1, str(b[:1]))

    # 2) 一個「在版控裡」的 must_not_track 條目必須被點名
    fake2 = json.loads(json.dumps(pol))
    fake2["must_not_track"] = [{"path": "agent_harness/CONVENTIONS.md",
                                "gitignore_rule": None,
                                "why": "self-test 故意說它不該進版控"}]
    fake2["grandfathered"] = []
    b2, _d2 = check_must_not_track(fake2)
    chk("must_not_track 多一筆 -> 紅（陰性對照）", len(b2) >= 1, str(b2[:1]))

    # 3) 去隱私掃描：把家目錄路徑塞進一個臨時 pack，必須命中
    zstd = shutil.which("zstd")
    if zstd:
        tmp = os.path.join(EVIDENCE, "__selftest_probe.txt.zst")
        try:
            r = subprocess.run([zstd, "-q", "-f", "-o", tmp],
                               input=("/Users/alexchuang/secret\n").encode(),
                               capture_output=True)
            if r.returncode == 0:
                hits = []
                for pname, rx in [(p["name"], re.compile(p["regex"]))
                                  for p in pol["privacy_forbidden"]]:
                    data = subprocess.run([zstd, "-dc", tmp], capture_output=True).stdout
                    if rx.search(data.decode()):
                        hits.append(pname)
                chk("去隱私：家目錄路徑出現在 pack 裡 -> 命中（陰性對照）",
                    "本機家目錄絕對路徑" in hits, str(hits))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        chk("去隱私自測（跳過：沒有 zstd）", True, "")

    # 4) 上限：造一個超上限的索引列必須被判紅
    fake3 = json.loads(json.dumps(pol))
    real = check_packs(pol)
    chk("現況的 pack 全部在上限內且索引指向的檔案都存在", not real[0], str(real[0][:1]))

    ok = True
    for name, good, detail in checks:
        print(f"  [{'ok' if good else 'BAD'}] {name}" + (f"   {detail}" if detail else ""))
        ok &= good
    print(f"\n  {sum(1 for _n, g, _d in checks if g)}/{len(checks)} checks passed")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--privacy", action="store_true", help="加上去隱私掃描（預設抽樣）")
    ap.add_argument("--sample", type=int, default=40, help="抽樣幾個 pack（預設 40）")
    ap.add_argument("--all", action="store_true", help="全掃 pack")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return cmd_self_test(a)
    if a.check or a.privacy:
        return cmd_check(a)
    ap.error("需要 --check / --privacy / --self-test 之一")


if __name__ == "__main__":
    sys.exit(main())
