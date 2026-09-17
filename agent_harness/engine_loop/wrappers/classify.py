#!/usr/bin/env python3
"""classify.py — 把 `scripts/check/` 的每一個腳本按**能力**分層，而不移動它們。

為什麼**不移動**（E4 item 3 的「分層」）
----------------------------------------
`scripts/check/` 底下的每一個可執行腳本都被引用了 **562 次**（去重後的引用關係）：
`docs/*.html` 134、`Backup/` 101、`scripts/` 62、`.workbuddy/memory/` 41、`agent_harness/` 220。

其中兩類是**不可改寫**的：
  * `docs/*.html` 是已定稿的白皮書。把它們的引用改成新路徑，就是讓一份已發表的文件說假話 ——
    而本 repo 對這件事的立場是「標註而不是改寫」（見 `docs/AGENT_HARNESS_E1_RESTRUCTURE_*.html`
    對舊白皮書的處理方式）。
  * `.workbuddy/memory/*.md` 是 append-only 的日誌，描述的是**當時**的狀態。

所以「分層」若做成 `git mv`，代價是 175 條引用懸空（134 ＋ 41），而收益只是目錄好看一點。
PLAN §3 的紅線本身也反對這件事的**動機**：那裡的立場是「`scripts/check/*` 是生產腳本」。

改成什麼
--------
分層做成**資料**：這一支把每個腳本指派到一個能力類（`classes.tsv`），而**類別是可機檢的** ——
`--check` 斷言每個腳本剛好落一類。這一支就是「40+ 檔分層」的實作，只是分層的座標是「它做什麼」
而不是「它在哪個資料夾」。

為什麼需要它（不是為了整齊）—— MANIFEST 的 `role` 欄在這裡是退化的
------------------------------------------------------------------
`agent_harness/engine_loop/MANIFEST.jsonl` 也有一個分類欄，實測：

    role 分佈（實際數字現場量；這一輪是 probe 34、gate 4、measure 3、compare 2、arms 1）：

**其中 34 個是 `probe`** ⇒ 對這個目錄來說那個標籤幾乎不帶資訊（而且全部可執行，
所以「可不可執行」也不是區分軸）。能力類是**第二個座標**，它的用處是回答
「我要做 X，該跑哪一支」—— 那個問題在 34 個 probe 之間是答不出來的。

★ 一個與計畫不符的地方，如實記下
--------------------------------
PLAN §3 只列了 5 個 wrapper（`sweep` / `ab` / `bench` / `gate` / `triage`），而實測的分割需要
**6 類**：`check_env.sh`、`check_torch.sh`、`check_server.sh`、`check_server_profiles.py`
是**環境／服務健康檢查**，不是掃描、不是 A/B、不是 benchmark、不是判對錯的閘門、
也不是對某個警報做鑑識。把它們硬塞進那 5 類會讓其中一類變成雜物桶。
PLAN 沒替它們安排位置，所以這裡新增 `env` 一類並把偏差寫出來 —— 而不是假裝 5 類剛好夠。

    classify.py            重新產生 classes.tsv
    classify.py --check    驗證現有 classes.tsv 對得上磁碟；不一致則 exit 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CHECK_DIR = os.path.join(REPO, "scripts", "check")
OUT = os.path.join(HERE, "classes.tsv")
MANIFEST = os.path.abspath(os.path.join(HERE, "..", "MANIFEST.jsonl"))

# ---------------------------------------------------------------------------
# 能力類：這是人手判斷的部分（腳本無法供給），其餘全部推導
# ---------------------------------------------------------------------------
CLASSES = {
    "sweep": "對一個旋鈕做參數掃描，逐格重啟伺服器並記錄一列",
    "ab": "對兩份設定／兩份輸出做對照，得出「有沒有差」",
    "bench": "產生吞吐量或品質的基準讀數",
    "gate": "判對錯：合格／不合格、可比較／不可比較",
    "triage": "對一個既有警報或讀數做鑑識，找出它的來源",
    "env": "環境與服務的健康檢查（不產生量測讀數）",
}

# 每個腳本一類。清單必須覆蓋 scripts/check/ 底下所有 .py 與 .sh —— 沒覆蓋到的會被 --check 點名。
ASSIGNMENT = {
    "sweep": [
        "decode_sweep.py", "pool_curve.py", "prefill_idle_sweep.py", "knifeedge_matrix.py",
    ],
    "ab": [
        "ab_interleave.py", "flag_ab.py", "mtp_accept_ab.py", "cgc_logits_oracle_compare.py",
        "ids_capture_diff.py", "replay_bench_compare.py", "replay_ngl_comparison.py",
        "replay_server_vs_cli_comparison.py", "batch_retest_commits.sh",
    ],
    "bench": [
        "decode_bench.py", "llama_bench_matrix.py", "replay_server_profile.py",
        # 2026-09-17 本線新增。與 llama_bench_matrix.py 同類（驅動 llama-bench 產出數字），
        # 差別是它把「profile × 標準 cell」與回報規則固定下來，並且是唯一的生產級入口。
        "prod_matrix.py",
        "replay_bench_database.py", "bare_48.py", "replay_client_regime.py", "flip_rate.py",
        "probe_skip0.py", "session_baseline.py",
    ],
    "gate": [
        "m123_oracle_gate.py", "precommit_e2e_gate.sh", "precommit_replay_gate.sh",
        "feasibility_gate_selftest.py", "oracle_truth_gate_selftest.py", "mtp_head_identity.py",
        "mtp_long_sequence_verify.py", "pool_feasibility.py", "mmid_pool_vs_gguf.py",
        "check_server_profiles.py",
        # 2026-09-17 由另一條線新增。本表必須覆蓋磁碟上的**全部** —— `--check` 就是這樣變紅的，
        # 而它紅得對：一個沒被分類的腳本就是一個沒有入口的腳本。
        "ids_capture_width_selftest.py",
    ],
    "triage": [
        "mmid_zero_row_triage.py", "gguf_dead_expert_census.py", "mmid_geometry_probe.sh",
        "thermal_pressure.py", "ioreport_gpu_pstate_probe.py", "powermetrics_gpu_freq.sh",
        "powermetrics_gpu_freq_parse.py", "prefill_certifiability.py", "prefill_gputime_report.py",
        # 2026-09-17 由另一條線新增：從 GGUF 推導 pool 幾何（per-slot bytes、BINDING layer、
        # slot 容量）。它回答的是「這個容量為什麼是這個數字」⇒ 與 gguf_dead_expert_census.py 同類。
        "gguf_pool_geometry.py",
    ],
    "env": [
        "check_env.sh", "check_torch.sh", "check_server.sh",
    ],
}

# 明確**排除**的：這裡沒有排除項，因為磁碟上每一支 .py/.sh 都是可執行的入口
# （2026-09-17 實測 44 支，同日另一條線加到 46 支 —— 這一行不寫數字，數字會過期）。
# 留著這個常數是為了讓「有沒有東西被排除」在檔案裡看得見（空清單是**一個答案**，
# 而不是「忘了想這件事」）。
EXCLUDED: dict[str, str] = {}


def _doc_line(path: str) -> str:
    """第一句說明。用來讓人核對分類，不是用來分類（分類是上面那張表）。"""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""
    lines = text.splitlines()[:40]
    if path.endswith(".py"):
        m = re.search(r'"""(.*?)(?:"""|\Z)', "\n".join(lines), re.S)
        if m:
            for candidate in m.group(1).strip().splitlines():
                if candidate.strip():
                    return candidate.strip()
        return ""
    for line in lines[1:14]:
        s = line.lstrip("# ").strip()
        if s and not s.startswith("!") and not s.startswith("/"):
            return s
    return ""


def manifest_roles() -> dict[str, str]:
    roles: dict[str, str] = {}
    if not os.path.exists(MANIFEST):
        return roles
    with open(MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                roles[row["path"]] = row.get("role", "?")
    return roles


def rows() -> list[tuple[str, str, str, str]]:
    """(class, script, manifest_role, doc_line) for every .py/.sh in scripts/check."""
    roles = manifest_roles()
    seen: dict[str, str] = {}
    out: list[tuple[str, str, str, str]] = []
    for cls, names in ASSIGNMENT.items():
        for name in names:
            if name in seen:
                raise SystemExit(f"error: {name} 被指派了兩次（{seen[name]} 與 {cls}）")
            seen[name] = cls
    for name, why in EXCLUDED.items():
        if name in seen:
            raise SystemExit(f"error: {name} 同時在 ASSIGNMENT 與 EXCLUDED")
        seen[name] = "(excluded)"

    on_disk = sorted(
        n for n in os.listdir(CHECK_DIR)
        if os.path.isfile(os.path.join(CHECK_DIR, n)) and n.endswith((".py", ".sh"))
    )
    missing_from_table = [n for n in on_disk if n not in seen]
    gone = [n for n in seen if n not in on_disk]
    if missing_from_table or gone:
        problems = []
        if missing_from_table:
            problems.append(f"磁碟上有但沒被分類：{missing_from_table}")
        if gone:
            problems.append(f"被分類了但磁碟上沒有：{gone}")
        raise SystemExit("error: scripts/check/ 與分類表不一致 —— " + "；".join(problems))

    for name in on_disk:
        cls = seen[name]
        out.append((cls, name, roles.get(f"scripts/check/{name}", "-"),
                    _doc_line(os.path.join(CHECK_DIR, name))))
    return sorted(out, key=lambda r: (list(CLASSES) + ["(excluded)"]).index(r[0]) if r[0] in CLASSES else 99)


def render(rs: list[tuple[str, str, str, str]]) -> str:
    lines = [
        "# classes.tsv — scripts/check/ 的能力分層。由 classify.py 產生，不要手改。",
        "#",
        "# 欄位：class <TAB> 腳本 <TAB> MANIFEST 的 role <TAB> 腳本自己的一句話說明",
        "#",
        "# 為什麼是這 6 類而不是 PLAN §3 的 5 類：見 classify.py 的 docstring（env 是新增的那一類）。",
    ]
    for cls, desc in CLASSES.items():
        lines.append(f"# [{cls}] {desc}")
    lines.append("#")
    for cls, name, role, doc in rs:
        lines.append(f"{cls}\t{name}\t{role}\t{doc}")
    return "\n".join(lines) + "\n"


def build() -> tuple[str, dict, list[tuple[str, str, str, str]]]:
    rs = rows()
    summary: dict[str, int] = {}
    for cls, *_ in rs:
        summary[cls] = summary.get(cls, 0) + 1
    return render(rs), summary, rs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify classes.tsv against disk; exit 1 on drift")
    ap.add_argument("--stats", action="store_true", help="print the class sizes and the degenerate-role finding")
    args = ap.parse_args()

    text, summary, rs = build()

    if args.stats:
        print("  class         n   說明")
        for cls, desc in CLASSES.items():
            print(f"  {cls:12s} {summary.get(cls, 0):3d}   {desc}")
        roles: dict[str, int] = {}
        for _, _, role, _ in rs:
            roles[role] = roles.get(role, 0) + 1
        n = sum(roles.values())
        print()
        print(f"  MANIFEST 的 role（{n} 個腳本）: " + ", ".join(f"{k}={v}" for k, v in sorted(roles.items(), key=lambda kv: -kv[1])))
        dominant = max(roles.items(), key=lambda kv: kv[1])
        print(f"  ⇒ 最大的一類是 {dominant[0]}={dominant[1]}／{n}"
              f"（{dominant[1] / n * 100:.0f}%）⇒ 那個標籤對這個目錄不具區分力，"
              f"這才是需要第二個座標的理由。")
        return 0

    if args.check:
        if not os.path.exists(OUT):
            print(f"  error: {OUT} 不存在（跑 classify.py 產生）", file=sys.stderr)
            return 1
        have = open(OUT, encoding="utf-8").read()
        if have != text:
            print("  DRIFT: classes.tsv 與 scripts/check/ 或分類表不一致", file=sys.stderr)
            import difflib
            for line in list(difflib.unified_diff(have.splitlines(), text.splitlines(),
                                                  "on-disk", "derived", lineterm=""))[:30]:
                print("    " + line, file=sys.stderr)
            return 1
        print(f"  OK: {len(rs)} 個腳本、{len(CLASSES)} 類、無漂移、無未分類項")
        return 0

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  wrote {os.path.relpath(OUT, REPO)}")
    for cls, desc in CLASSES.items():
        print(f"    {cls:12s} {summary.get(cls, 0):3d}   {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
