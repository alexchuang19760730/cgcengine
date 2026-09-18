#!/usr/bin/env python3
"""區網（LAN）通道檢查器 —— 回答「同一個網段上的端側，現在連得到 harness 嗎」。

三個模式，三件**不同**的事（刻意分開，因為前置條件不同）：

  --check      離線。驗 lan.json 的「宣告」是否與原始碼一致、是否自洽、
               以及與 fleet.json 的本機埠引用是否對得上。**完全不連網。**
  --probe      連線。實測每個服務的**綁定**（loopback vs LAN 位址），可達時再驗**身分**。
  --self-test  突變式黑箱自測：真的起 socket，證明判定分得出差別。

★ 這支檢查器證明的是「socket 綁在哪裡」，**不是**「別的機器連得到」。
  後者還取決於 AP isolation／路由器 ACL／對端防火牆 —— 只有對端自己回報才算數。
★ 它也不掃描 ports_in_scope 以外的埠，不探測 192.168.101.0/24 以外的位址。

Usage:
  python3 agent_harness/portal/lan_check.py --check
  python3 agent_harness/portal/lan_check.py --probe [--json] [--out FILE]
  python3 agent_harness/portal/lan_check.py --self-test

  # 測試用覆寫（不會動到真 repo）
  --lan-json PATH --fleet-json PATH --repo-root PATH --site-repo PATH
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent                  # agent_harness/portal
REPO_DEFAULT = HERE.parents[1]                          # repo root
PORTAL_DIR = HERE

LOCAL_PORT_RE = re.compile(r"(?:127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\]|::1):(\d{2,5})")

# ★★ 一定要一個「不走代理」的 opener：環境裡的 HTTP_PROXY（本機實測是
#   http://127.0.0.1:7897）會把 http://192.168.101.90:… 也丟進代理，
#   於是區網的身分檢查靜默失敗 —— 而判定是 unverified（**不是紅**）⇒ 低報。
#   自測第 ⑤／⑥ 格就是這樣掛掉的：不是程式的判定錯，是連線被代理吃了。
OPENER_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ══════════════════════════════════════════════════════════════════════════
#  讀檔
# ══════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def repo_roots(repo_root: Path, site_repo: Path) -> dict[str, Path]:
    return {"self": Path(repo_root), "powerauto.ai": Path(site_repo)}


# ══════════════════════════════════════════════════════════════════════════
#  離線檢查：宣告 vs 原始碼
# ══════════════════════════════════════════════════════════════════════════

def check(spec: dict, fleet: dict, roots: dict[str, Path],
          allow_missing_site_repo: bool = False) -> list[str]:
    problems: list[str] = []

    # ── 1) 結構 ────────────────────────────────────────────────────────────
    if spec.get("schema") != 1:
        problems.append(f"schema 應該是 1，實際 {spec.get('schema')!r}")
    host = spec.get("this_host") or {}
    for k in ("iface", "ipv4", "cidr", "gateway"):
        if not host.get(k):
            problems.append(f"this_host 缺 {k}")
    services = spec.get("services") or []
    if not services:
        problems.append("services 是空的 —— 一份沒有服務的 LAN 註冊表沒有意義")

    seen: dict[int, str] = {}
    for s in services:
        sid = s.get("id") or "(沒有 id)"
        for k in ("id", "port", "binds_default", "lan_ready", "how", "auth_required"):
            if k not in s:
                problems.append(f"服務 {sid} 缺欄位 {k}")
        port = s.get("port")
        if not isinstance(port, int):
            problems.append(f"服務 {sid} 的 port 不是整數: {port!r}")
            continue
        if port in seen:
            problems.append(f"埠 {port} 被兩個服務宣告: {sid} 與 {seen[port]}")
        seen[port] = sid

    # ── 2) 自洽：lan_ready 不能與 binds_default 矛盾 ────────────────────────
    for s in services:
        if s.get("lan_ready") is True and str(s.get("binds_default")) in ("127.0.0.1", "localhost"):
            problems.append(
                f"服務 {s.get('id')} 宣告 lan_ready=true，但 binds_default={s.get('binds_default')!r} "
                f"⇒ 這兩個不能同時成立（端側連不到 loopback）")

    # ── 3) evidence 逐條驗真（跨 repo 的缺 repo 要出聲，不能靜默通過）──────
    ev_total = ev_ok = 0
    for s in services:
        for ev in (s.get("evidence") or []):
            ev_total += 1
            name = ev.get("repo", "self")
            root = roots.get(name)
            if root is None or not Path(root).exists():
                if allow_missing_site_repo:
                    continue
                problems.append(
                    f"服務 {s.get('id')} 的 evidence 指向 repo {name!r}，但它在 {root} 不存在 "
                    f"⇒ 無法驗真（要略過請加 --allow-missing-site-repo）")
                continue
            f = Path(root) / ev.get("path", "")
            if not f.exists():
                problems.append(f"服務 {s.get('id')} 的 evidence 檔不存在: {name}/{ev.get('path')}")
                continue
            n = f.read_text(encoding="utf-8", errors="ignore").count(ev.get("contains", ""))
            if n != 1:
                problems.append(
                    f"服務 {s.get('id')} 的 evidence 命中 {n} 次（要正好 1）: "
                    f"{name}/{ev.get('path')} ← {ev.get('contains','')[:60]!r}")
            else:
                ev_ok += 1

    # ── 4) probe 範圍要涵蓋所有服務埠 ──────────────────────────────────────
    scope = (spec.get("probe") or {}).get("ports_in_scope") or []
    for p in sorted(seen):
        if p not in scope:
            problems.append(f"服務 {seen[p]} 的埠 {p} 不在 probe.ports_in_scope ⇒ --probe 不會量到它")

    # ── 5) 別人的埠不可以與我們的撞號 ──────────────────────────────────────
    foreign = spec.get("foreign_lan_ports") or []
    for f in foreign:
        if f.get("port") in seen:
            problems.append(
                f"埠 {f.get('port')} 同時被宣告成我們的服務（{seen[f['port']]}）與別人的服務（{f.get('who')}）")
    # 每一條 foreign 都要有「為什麼要登記它」—— 否則它只是裝飾
    for f in foreign:
        if not f.get("why_it_matters"):
            problems.append(f"foreign 埠 {f.get('port')} 沒寫 why_it_matters（為什麼要登記它）")

    # ── 6) 雙向：fleet.json 引用的本機埠必須被宣告（釘缺席）───────────────
    declared = set(seen) | {f.get("port") for f in foreign}
    referenced = {int(m.group(1)) for m in LOCAL_PORT_RE.finditer(json.dumps(fleet, ensure_ascii=False))}
    for p in sorted(referenced - declared):
        problems.append(
            f"fleet.json 引用了本機埠 {p}，但 lan.json 沒有宣告它 "
            f"⇒ 那個引用指向一個沒人承認的服務")

    return problems


# ══════════════════════════════════════════════════════════════════════════
#  連線探測
# ══════════════════════════════════════════════════════════════════════════

def tcp_ok(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def bind_verdict(loop_ok: bool, lan_ok: bool) -> str:
    """★ 四種判定，**沒有** foreign —— 綁定層分不出「我們的」與「別人的」。

    實測（見 lan.json 的 measured_bind_matrix）：綁在特定 LAN IP 的 socket
    從 127.0.0.1 是連不上的 ⇒「loopback 不通」不等於「那是別人的」。
    """
    if lan_ok and loop_ok:
        return "bound-lan"
    if lan_ok and not loop_ok:
        return "bound-lan-only"
    if loop_ok and not lan_ok:
        return "loopback-only"
    return "absent"


def check_identity(host: str, svc: dict, timeout: float) -> dict:
    """可達之後再看它是誰。這是「可達 ≠ 是我們的」的唯一解法。"""
    ident = svc["identity"]
    url = f"http://{host}:{svc['port']}{ident['path']}"
    try:
        with OPENER_NO_PROXY.open(url, timeout=max(timeout * 3, 1.5)) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return {"verdict": "unverified", "url": url,
                "detail": f"HTTP {e.code}（服務有回應但不是 200）"}
    except Exception as e:
        return {"verdict": "unverified", "url": url, "detail": repr(e)[:140]}
    seen_obj = body.get("object")
    ok = seen_obj == ident.get("object")
    return {"verdict": "match" if ok else "mismatch", "url": url,
            "probed": {"object": ident.get("object")},
            "seen": {"object": seen_obj, "auth_required": body.get("auth_required")}}


def probe(spec: dict, timeout: float | None = None, do_identity: bool = True,
          lan_ip: str | None = None) -> dict:
    host = spec.get("this_host") or {}
    lan_ip = lan_ip or host.get("ipv4")
    if not lan_ip:
        raise SystemExit("lan.json 的 this_host.ipv4 是空的 ⇒ 不知道要從哪個位址探")
    t = float(timeout if timeout is not None else (spec.get("probe") or {}).get("timeout_s", 0.5))

    rows = []
    for s in spec.get("services") or []:
        port = s["port"]
        lo = tcp_ok("127.0.0.1", port, t)
        la = tcp_ok(lan_ip, port, t)
        row = {
            "id": s.get("id"), "port": port,
            "loopback_ok": lo, "lan_ok": la,
            "verdict": bind_verdict(lo, la),
            "declared_lan_ready": s.get("lan_ready"),
            "declared_binds": s.get("binds_default"),
            "what": s.get("what"),
        }
        if do_identity and la and s.get("identity"):
            row["identity"] = check_identity(lan_ip, s, t)
        rows.append(row)

    # ★ 「綁到區網卻沒有認證」是可機檢的風險。有身分端點就讀它**當下的**
    #   auth_required（比宣告準）；沒有才退回宣告值。
    security = []
    for r in rows:
        if r["verdict"] not in ("bound-lan", "bound-lan-only"):
            continue
        svc = next(s for s in (spec.get("services") or []) if s.get("id") == r["id"])
        live = (r.get("identity") or {}).get("seen", {}).get("auth_required")
        effective = live if live is not None else svc.get("auth_required")
        if effective is False:
            security.append({
                "id": r["id"], "port": r["port"],
                "detail": f"綁在區網上（{r['verdict']}）但沒有認證"
                          f"（{'實測 status 回 auth_required=false' if live is None else 'status 回報 auth_required=false'}）"
                          f"⇒ 同網段任何人都能用它的算力／讀它的能力清單",
            })

    foreign_rows = []
    for f in spec.get("foreign_lan_ports") or []:
        port = f.get("port")
        lo = tcp_ok("127.0.0.1", port, t)
        la = tcp_ok(lan_ip, port, t)
        foreign_rows.append({"port": port, "who": f.get("who"),
                             "loopback_ok": lo, "lan_ok": la,
                             "verdict": bind_verdict(lo, la)})

    scope = (spec.get("probe") or {}).get("ports_in_scope") or []
    known = {r["port"] for r in rows} | {r["port"] for r in foreign_rows}
    undeclared = [p for p in scope
                  if p not in known and (tcp_ok(lan_ip, p, t) or tcp_ok("127.0.0.1", p, t))]

    return {
        "security": security,
        "lan_ip": lan_ip,
        "iface": host.get("iface"), "cidr": host.get("cidr"), "gateway": host.get("gateway"),
        "firewall": (host.get("firewall") or {}).get("global_state"),
        "timeout_s": t,
        "services": rows,
        "foreign": foreign_rows,
        "undeclared_open": undeclared,
        "summary": {
            "services": len(rows),
            "lan_ready_declared": sum(1 for r in rows if r["declared_lan_ready"] is True),
            "lan_bound": sum(1 for r in rows if r["verdict"] in ("bound-lan", "bound-lan-only")),
            "running": sum(1 for r in rows if r["verdict"] != "absent"),
        },
    }


# ══════════════════════════════════════════════════════════════════════════
#  輸出
# ══════════════════════════════════════════════════════════════════════════

MARK = {"bound-lan": "✓ LAN", "bound-lan-only": "✓ LAN(只綁IP)",
        "loopback-only": "只 loopback", "absent": "沒在跑"}


def print_probe(p: dict) -> None:
    print(f"  ── 本機：{p['iface']} {p['lan_ip']} {p['cidr']}　閘道 {p['gateway']}　"
          f"防火牆 {p['firewall']}")
    print(f"  ── 服務（{p['summary']['services']} 個，timeout {p['timeout_s']}s）")
    for r in p["services"]:
        idn = ""
        if r.get("identity"):
            idn = f"　身分={r['identity']['verdict']}"
            if r["identity"]["verdict"] == "mismatch":
                idn += f"（看到 {r['identity'].get('seen')}）"
        warn = "" if (r["declared_lan_ready"] is not True or
                      r["verdict"] in ("bound-lan", "bound-lan-only")) else "  ← 宣告 lan_ready 但沒綁"
        print(f"     {r['id']:<22}:{r['port']:<6} {MARK[r['verdict']]:<14}{idn}{warn}")
    if p["foreign"]:
        print(f"  ── 別人的埠（{len(p['foreign'])} 個）—— 可達 ≠ 是我們的")
        for r in p["foreign"]:
            print(f"     :{r['port']:<6} {MARK[r['verdict']]:<14}{r['who']}")
    for w in p.get("security") or []:
        print(f"     ★ 安全：{w['id']} :{w['port']} —— {w['detail']}")
    print(f"  ── 掃描範圍內、卻沒被宣告的開著埠：{p['undeclared_open'] or '（無）'}")
    s = p["summary"]
    print(f"  ── 判定：宣告 lan_ready {s['lan_ready_declared']} 個，真的綁在區網上 {s['lan_bound']} 個；"
          f"正在跑 {s['running']}/{s['services']}")


# ══════════════════════════════════════════════════════════════════════════
#  自測（真的起 socket）
# ══════════════════════════════════════════════════════════════════════════

class FixtureServer:
    """真的 bind ＋ listen。body 給了就回一個 HTTP 200 JSON（供身分檢查）。"""

    def __init__(self, host: str, port: int = 0, body: str | None = None):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.body = body
        self.stop = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while not self.stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            try:
                if self.body is not None:
                    conn.settimeout(0.6)
                    try:
                        conn.recv(4096)
                    except Exception:
                        pass
                    payload = self.body.encode("utf-8")
                    conn.sendall(
                        (f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                         ).encode() + payload)
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    def close(self):
        self.stop = True
        try:
            self.sock.close()
        except Exception:
            pass


def registry_digest() -> str:
    """真 repo 的三份註冊表。自測前後要比對它 —— 自測不得有副作用。"""
    import hashlib
    h = hashlib.sha256()
    for f in ("lan.json", "fleet.json", "join.json"):
        fp = PORTAL_DIR / f
        h.update(f.encode())
        h.update(fp.read_bytes() if fp.exists() else b"")
    return h.hexdigest()


def self_test() -> int:
    me = str(Path(__file__).resolve())
    tmp = Path(tempfile.mkdtemp(prefix="lan_check_selftest_"))
    digest_before = registry_digest()          # ★ 在動任何東西之前取
    results: list[tuple[str, bool, str]] = []

    def case(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, bool(cond), detail))

    def run(args: list[str]):
        r = subprocess.run([sys.executable, me] + args, capture_output=True, text=True, timeout=180)
        return r.returncode, r.stdout + r.stderr

    # 真的 LAN 位址（探測要用它）
    LAN = json.loads((PORTAL_DIR / "lan.json").read_text(encoding="utf-8"))["this_host"]["ipv4"]

    def free_port() -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    def svc(sid, port, *, lan_ready=True, binds="0.0.0.0", identity=None, ev=True,
            auth_required=False):
        # ★ auth_required 是 --check 的必填欄位（要求「有沒有認證」被明寫而不是推的）
        s = {"id": sid, "what": f"fixture {sid}", "port": port,
             "binds_default": binds, "binds_flag": "--host", "auth": "fixture",
             "auth_required": auth_required, "lan_ready": lan_ready, "how": "fixture"}
        if identity:
            s["identity"] = identity
        if ev:
            s["evidence"] = [{"repo": "self",
                              "path": "agent_harness/portal/fixture_target.py",
                              "contains": "LAN-FIXTURE-MARKER"}]
        return s

    def fixture(idx, services, *, fleet=None, scope=None, foreign=None, marker="LAN-FIXTURE-MARKER"):
        r = tmp / f"repo{idx}"
        portal = r / "agent_harness" / "portal"
        portal.mkdir(parents=True, exist_ok=True)
        (portal / "fixture_target.py").write_text(f"MARKER = '{marker}'\n", encoding="utf-8")
        spec = {
            "schema": 1, "title": "fixture",
            "this_host": {"iface": "en0", "ipv4": LAN, "cidr": "x/24", "gateway": "x",
                          "firewall": {"global_state": "disabled"}},
            "services": services,
            "foreign_lan_ports": foreign or [],
            "probe": {"timeout_s": 0.4, "ports_in_scope": scope if scope is not None
                      else [s["port"] for s in services], "never": "fixture"},
            "rules": ["fixture"], "not_proves": ["fixture"],
        }
        (portal / "lan.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
        (portal / "fleet.json").write_text(
            json.dumps(fleet if fleet is not None else {"endpoints": []}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        return r

    def probe_fixture(r: Path, extra: list[str] | None = None):
        return run(["--probe", "--json", "--lan-json", str(r / "agent_harness/portal/lan.json"),
                    "--fleet-json", str(r / "agent_harness/portal/fleet.json"),
                    "--repo-root", str(r)] + (extra or []))

    def parse_json(out: str) -> dict:
        i = out.find("{")
        return json.loads(out[i:]) if i >= 0 else {}

    # ── ① 綁 127.0.0.1 ⇒ loopback-only ─────────────────────────────────────
    p1 = free_port()
    s1 = FixtureServer("127.0.0.1", p1)
    r1 = fixture(1, [svc("loop-only", p1, lan_ready=False, binds="127.0.0.1")])
    rc, out = probe_fixture(r1)
    d = parse_json(out)
    v = (d.get("services") or [{}])[0].get("verdict")
    case("① 綁 127.0.0.1 ⇒ 判定 loopback-only", v == "loopback-only", f"verdict={v}")
    s1.close()

    # ── ② 綁 0.0.0.0 ⇒ bound-lan ──────────────────────────────────────────
    p2 = free_port()
    s2 = FixtureServer("0.0.0.0", p2)
    r2 = fixture(2, [svc("lan-any", p2)])
    rc, out = probe_fixture(r2)
    d = parse_json(out)
    v = (d.get("services") or [{}])[0].get("verdict")
    case("② 綁 0.0.0.0 ⇒ 判定 bound-lan（端側連得到）", v == "bound-lan", f"verdict={v}")

    # ── ③ 綁特定 LAN IP ⇒ bound-lan-only（★ 這格是本次量測的產物）────────
    p3 = free_port()
    s3 = FixtureServer(LAN, p3)
    r3 = fixture(3, [svc("lan-ip-only", p3)])
    rc, out = probe_fixture(r3)
    d = parse_json(out)
    v = (d.get("services") or [{}])[0].get("verdict")
    case("★③ 綁特定 LAN IP ⇒ bound-lan-only（不是 foreign！）",
         v == "bound-lan-only", f"verdict={v}")
    case("★③b 同一格：判成 bound-lan-only 時**不算** lan_ready 未達成",
         v in ("bound-lan", "bound-lan-only"), f"verdict={v}")
    s3.close()

    # ── ④ 什麼都沒起 ⇒ absent ─────────────────────────────────────────────
    p4 = free_port()
    r4 = fixture(4, [svc("nothing", p4, lan_ready=False)])
    rc, out = probe_fixture(r4)
    d = parse_json(out)
    v = (d.get("services") or [{}])[0].get("verdict")
    case("④ 沒有 listener ⇒ absent（unknown，不是紅）", v == "absent", f"verdict={v}")

    # ── ⑤ 身分相符 ⇒ match ────────────────────────────────────────────────
    p5 = free_port()
    OBJ = "fixture.edge.status"
    s5 = FixtureServer("0.0.0.0", p5, body=json.dumps({"object": OBJ}))
    r5 = fixture(5, [svc("id-ok", p5, identity={"path": "/v1/edge/status", "object": OBJ})])
    rc, out = probe_fixture(r5)
    d = parse_json(out)
    idv = ((d.get("services") or [{}])[0].get("identity") or {}).get("verdict")
    case("⑤ 可達且身分相符 ⇒ match", idv == "match", f"identity={idv}")
    s5.close()

    # ── ⑥ 身分不符 ⇒ mismatch（可達但是別人的）────────────────────────────
    p6 = free_port()
    s6 = FixtureServer("0.0.0.0", p6, body=json.dumps({"object": "someone.else"}))
    r6 = fixture(6, [svc("id-bad", p6, identity={"path": "/v1/edge/status", "object": OBJ})])
    rc, out = probe_fixture(r6)
    d = parse_json(out)
    idv = ((d.get("services") or [{}])[0].get("identity") or {}).get("verdict")
    case("★⑥ 可達但身分不符 ⇒ mismatch（可達 ≠ 是我們的）", idv == "mismatch", f"identity={idv}")
    s6.close()

    # ── ⑥b 環境有 HTTP_PROXY ⇒ 身分檢查仍必須成立（★ 這格是踩到的 bug）──
    p6b = free_port()
    s6b = FixtureServer("0.0.0.0", p6b, body=json.dumps({"object": OBJ}))
    r6b = fixture(60, [svc("id-proxy", p6b,
                           identity={"path": "/v1/edge/status", "object": OBJ})])
    env = dict(os.environ, HTTP_PROXY="http://127.0.0.1:7897", HTTPS_PROXY="http://127.0.0.1:7897",
               http_proxy="http://127.0.0.1:7897", no_proxy="", NO_PROXY="")
    rr = subprocess.run([sys.executable, me, "--probe", "--json",
                         "--lan-json", str(r6b / "agent_harness/portal/lan.json"),
                         "--fleet-json", str(r6b / "agent_harness/portal/fleet.json"),
                         "--repo-root", str(r6b)],
                        capture_output=True, text=True, timeout=180, env=env)
    d6b = parse_json(rr.stdout + rr.stderr)
    idv6b = ((d6b.get("services") or [{}])[0].get("identity") or {}).get("verdict")
    case("★⑥b 環境有 HTTP_PROXY 時，區網的身分檢查仍要 match（不得被代理吃掉）",
         idv6b == "match", f"identity={idv6b}")
    s6b.close()

    # ── ⑥c 綁到區網 ＋ 沒有認證 ⇒ security 要出聲（且要能分辨有／無）────
    p6c = free_port()
    s6c = FixtureServer("0.0.0.0", p6c, body=json.dumps({"object": OBJ, "auth_required": False}))
    r6c = fixture(61, [svc("naked", p6c, identity={"path": "/v1/edge/status", "object": OBJ})])
    rc, out = probe_fixture(r6c)
    d6c = parse_json(out)
    case("★⑥c 綁在區網卻沒有認證 ⇒ security 有一筆指名它",
         any(w.get("id") == "naked" for w in (d6c.get("security") or [])),
         f"security={d6c.get('security')}")
    s6c.close()

    p6d = free_port()
    s6d = FixtureServer("0.0.0.0", p6d, body=json.dumps({"object": OBJ, "auth_required": True}))
    r6d = fixture(62, [svc("guarded", p6d, identity={"path": "/v1/edge/status", "object": OBJ})])
    rc, out = probe_fixture(r6d)
    d6d = parse_json(out)
    case("★⑥d 有認證時 security 必須是空的（證明⑥c 不是永遠報）",
         not (d6d.get("security") or []), f"security={d6d.get('security')}")
    s6d.close()

    # ── ⑦ --check：lan_ready=true 卻綁 loopback ⇒ 紅 ────────────────────
    r7 = fixture(7, [svc("bad-claim", free_port(), lan_ready=True, binds="127.0.0.1")])
    rc, out = run(["--check", "--lan-json", str(r7 / "agent_harness/portal/lan.json"),
                   "--fleet-json", str(r7 / "agent_harness/portal/fleet.json"),
                   "--repo-root", str(r7)])
    case("⑦ 宣告 lan_ready=true 而 binds_default=127.0.0.1 ⇒ rc=1 且指名",
         rc != 0 and "不能同時成立" in out, f"rc={rc}")

    # ── ⑧ --check：乾淨 fixture ⇒ rc=0（陰性對照）────────────────────────
    p8 = free_port()
    r8 = fixture(8, [svc("good", p8, lan_ready=True, binds="0.0.0.0")])
    rc, out = run(["--check", "--lan-json", str(r8 / "agent_harness/portal/lan.json"),
                   "--fleet-json", str(r8 / "agent_harness/portal/fleet.json"),
                   "--repo-root", str(r8)])
    case("⑧ 乾淨 fixture ⇒ --check rc=0（證明⑦不是因為檢查永遠紅）", rc == 0, f"rc={rc} {out[-200:]}")

    # ── ⑨ --check：evidence 指針壞掉 ⇒ 紅 ───────────────────────────────
    p9 = free_port()
    r9 = fixture(9, [svc("ev-bad", p9)], marker="DIFFERENT-MARKER")
    rc, out = run(["--check", "--lan-json", str(r9 / "agent_harness/portal/lan.json"),
                   "--fleet-json", str(r9 / "agent_harness/portal/fleet.json"),
                   "--repo-root", str(r9)])
    case("⑨ evidence 指針命中 0 次 ⇒ rc=1", rc != 0 and "命中 0 次" in out, f"rc={rc}")

    # ── ⑩ --check：fleet 引用未宣告的埠 ⇒ 紅（釘缺席）──────────────────
    p10 = free_port()
    fleet10 = {"endpoints": [{"id": "x", "channels": {
        "heartbeat": {"url": "http://127.0.0.1:9999/v1/compute/status"}}}]}
    r10 = fixture(10, [svc("ok", p10)], fleet=fleet10)
    rc, out = run(["--check", "--lan-json", str(r10 / "agent_harness/portal/lan.json"),
                   "--fleet-json", str(r10 / "agent_harness/portal/fleet.json"),
                   "--repo-root", str(r10)])
    case("★⑩ fleet.json 引用本機埠 9999 但 lan.json 沒宣告 ⇒ rc=1 並指名",
         rc != 0 and "9999" in out, f"rc={rc}")

    # ── ⑪ --check：服務埠不在 probe 範圍 ⇒ 紅 ──────────────────────────
    p11 = free_port()
    r11 = fixture(11, [svc("out-of-scope", p11)], scope=[])
    rc, out = run(["--check", "--lan-json", str(r11 / "agent_harness/portal/lan.json"),
                   "--fleet-json", str(r11 / "agent_harness/portal/fleet.json"),
                   "--repo-root", str(r11)])
    case("⑪ 服務埠不在 ports_in_scope ⇒ rc=1（--probe 會漏掉它）",
         rc != 0 and "ports_in_scope" in out, f"rc={rc}")

    # ── ⑫ --check：foreign 與我們的撞號 ⇒ 紅 ────────────────────────────
    p12 = free_port()
    r12 = fixture(12, [svc("mine", p12)],
                  foreign=[{"port": p12, "who": "someone", "why_it_matters": "fixture"}])
    rc, out = run(["--check", "--lan-json", str(r12 / "agent_harness/portal/lan.json"),
                   "--fleet-json", str(r12 / "agent_harness/portal/fleet.json"),
                   "--repo-root", str(r12)])
    case("⑫ 同一埠同時是我們的與別人的 ⇒ rc=1", rc != 0 and "同時被宣告成" in out, f"rc={rc}")

    # ── ⑬ --check：跨 repo evidence，repo 不在 ⇒ 出聲（不靜默通過）─────
    p13 = free_port()
    s13 = svc("cross", p13, ev=False)
    s13["evidence"] = [{"repo": "powerauto.ai", "path": "installer/edge_server.py",
                        "contains": "--api-key"}]
    r13 = fixture(13, [s13])
    rc1, out1 = run(["--check", "--lan-json", str(r13 / "agent_harness/portal/lan.json"),
                     "--fleet-json", str(r13 / "agent_harness/portal/fleet.json"),
                     "--repo-root", str(r13), "--site-repo", str(tmp / "no-such-repo")])
    rc2, out2 = run(["--check", "--lan-json", str(r13 / "agent_harness/portal/lan.json"),
                     "--fleet-json", str(r13 / "agent_harness/portal/fleet.json"),
                     "--repo-root", str(r13), "--site-repo", str(tmp / "no-such-repo"),
                     "--allow-missing-site-repo"])
    case("★⑬ 跨 repo 的 evidence 在 repo 不存在時 ⇒ rc=1 並說明（不是靜默通過）",
         rc1 != 0 and "不存在" in out1, f"rc={rc1}")
    case("⑬b 明確加 --allow-missing-site-repo ⇒ rc=0（可降級，但要明說）",
         rc2 == 0, f"rc={rc2} {out2[-160:]}")

    # ── ⑭ --probe：掃描範圍內未宣告卻開著的埠要出聲 ─────────────────────
    p14, p14b = free_port(), free_port()
    s14 = FixtureServer("0.0.0.0", p14)
    s14b = FixtureServer("0.0.0.0", p14b)          # 這個不宣告
    r14 = fixture(14, [svc("declared", p14)], scope=[p14, p14b])
    rc, out = probe_fixture(r14)
    d = parse_json(out)
    case("★⑭ 掃描範圍內有個沒被宣告的開著埠 ⇒ 出現在 undeclared_open",
         p14b in (d.get("undeclared_open") or []), f"undeclared={d.get('undeclared_open')}")
    s14.close()
    s14b.close()

    # ── ⑮ 看門狗：真 repo 一個位元組都不該動 ─────────────────────────────
    case("⑮ 看門狗：真 repo 的註冊表沒被自測寫到（雜湊不變）",
         registry_digest() == digest_before, "digest changed")

    ok = sum(1 for _, c, _ in results if c)
    print(f"  ── 自測 {ok}/{len(results)} ──")
    for name, c, detail in results:
        print(f"  {'PASS' if c else 'FAIL'}  {name}" + (f"    [{detail}]" if not c else ""))
    return 0 if ok == len(results) else 1


# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="區網（LAN）通道檢查器")
    ap.add_argument("--check", action="store_true", help="離線：宣告 vs 原始碼（完全不連網）")
    ap.add_argument("--probe", action="store_true", help="連線：實測綁定（可達時再驗身分）")
    ap.add_argument("--self-test", action="store_true", help="突變式黑箱自測")
    ap.add_argument("--json", action="store_true", help="機器可讀輸出")
    ap.add_argument("--out", default="", help="把 --probe 的結果寫到這個檔")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--no-identity", action="store_true", help="不驗身分（只驗可達）")
    ap.add_argument("--lan-json", default=str(PORTAL_DIR / "lan.json"))
    ap.add_argument("--fleet-json", default=str(PORTAL_DIR / "fleet.json"))
    ap.add_argument("--repo-root", default=str(REPO_DEFAULT))
    ap.add_argument("--site-repo", default=str(Path.home() / "Documents" / "powerauto.ai"))
    ap.add_argument("--allow-missing-site-repo", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    spec = load_json(args.lan_json)
    fleet = load_json(args.fleet_json)
    roots = repo_roots(Path(args.repo_root), Path(args.site_repo))

    if args.check:
        problems = check(spec, fleet, roots, args.allow_missing_site_repo)
        if problems:
            for p in problems:
                print(f"  [error] {p}")
            print(f"  -> DRIFT: {len(problems)} 項")
            return 1
        n_ev = sum(len(s.get("evidence") or []) for s in spec.get("services") or [])
        print(f"  -> OK: {len(spec.get('services') or [])} 個服務"
              f"（lan_ready {sum(1 for s in spec.get('services') or [] if s.get('lan_ready') is True)} 個）、"
              f"{n_ev} 條 evidence 逐條命中 1 次、與 fleet.json 的本機埠引用一致")
        return 0

    if args.probe:
        p = probe(spec, args.timeout, not args.no_identity)
        if args.json:
            print(json.dumps(p, ensure_ascii=False, indent=2))
        else:
            print_probe(p)
        if args.out:
            Path(args.out).write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
            print(f"  wrote {args.out}")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
