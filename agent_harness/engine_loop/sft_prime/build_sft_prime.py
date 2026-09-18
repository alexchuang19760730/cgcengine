#!/usr/bin/env python3
"""sft_prime — the prime-agent projection: (evidence -> lesson) and (state -> next arm).

WHY THESE TWO SHAPES AND NOT "ALL THE TRACES"
--------------------------------------------
The loop's output is not a conversation, it is a set of judgements. A judgement has two halves
worth training on, and the plan names both (§6.2):

  evidence -> lesson   the generalisation step. Input is the observation that COST something
                       (`because`, with its numbers), output is the reusable rule. This is what
                       makes a later session cheaper.
  state -> next arm    the choice step. Input is a decision's question plus the readings it
                       leaned on PLUS what has already been ruled out, output is `action`.
                       Including `ruled_out` is the point: the schema itself calls the negative
                       knowledge "the single most reused thing in this loop", and a model that
                       only sees the winning hypothesis re-walks dead ends.

TWO THINGS ARE DELIBERATELY NOT TRAINED HERE
--------------------------------------------
* `judgement == "refuted"` decisions are EXCLUDED from the positive set. The decision schema
  says this field is "the anti-training-poison gate". A refuted hypothesis that becomes a
  positive example teaches the model to re-walk the dead end the loop already paid for.
* lessons whose `superseded_by` is set are excluded from `evidence -> lesson`, for the same
  reason. `--include-superseded` exists only to make the exclusion VISIBLE (it prints how many
  were held back); it does not make them usable.

Usage:
    python3 agent_harness/engine_loop/sft_prime/build_sft_prime.py [--out-dir DIR] [--val-frac F]
        [--system full|none]
    python3 agent_harness/engine_loop/sft_prime/build_sft_prime.py --check
    python3 agent_harness/engine_loop/sft_prime/build_sft_prime.py --self-test

WHY A BARE RUN IS NOT `--system full`, AND WHAT --check IS FOR
--------------------------------------------------------------
The committed artifact was built with `--system none` (see `PROVENANCE.json`), and every row here
already carries the observation it is meant to generalise, so repeating a 90,757 B charter in each
of 166 rows would add ~15 MB and no information. That is a judgement, not a fact -- so it is a flag,
it is recorded in `PROVENANCE.json` beside the data, and a bare rebuild reproduces the recorded
invocation rather than silently switching to `full`.

`--check` exists because nothing detected that this artifact had gone stale. Regenerating it today
yields +47,478 B in `train.jsonl` alone (lessons 112 -> 132, of which 130 are live), and every
existing signal stayed green while that was true: `index_assets.py --check` reports on the assets it
knows about, and these four files are not among them. **A green check says the entries it covers
agree; it does not say the artifact is covered.**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sft_common as C  # noqa: E402

BUILDER = "sft_prime/build_sft_prime.py"
# 真目錄，永遠由 __file__ 推導、不受 SFT_ROOT 影響：自測的看門狗要拿它當量尺，
# 若用它被重導後的值，看門狗量的就是暫時樹，永遠通過。
REAL_OUT_DIR = Path(__file__).resolve().parent


def evidence_to_lesson(lessons: list[dict], sys_text: str) -> list[dict]:
    rows = []
    for l in lessons:
        if l.get("superseded_by"):
            continue  # a replaced rule must not be taught as current
        user = [
            "以下是一條引擎調校迴圈裡「付過代價」的觀測。請把它一般化成**一條可重用的規訓**——",
            "換一個問題也適用，而且讀者手上沒有這筆 episode 也必須能照著做。",
            "",
            f"觀測（because）：\n{l['because']}",
        ]
        ce = l.get("counterexample_observed")
        if ce:
            user += ["", f"它實際被違反的紀錄（counterexample_observed）：\n{ce}"]
        if l.get("applies_to"):
            user += ["", "它管轄的檔案：" + ", ".join(f"`{p}`" for p in l["applies_to"])]
        rows.append({
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": "\n".join(user)},
                {"role": "assistant", "content": l["rule"]},
            ],
            "kind": "evidence_to_lesson",
            "source_id": l["lesson_id"],
            "class": l["class"],
            "label": "positive",
        })
    return rows


def state_to_next_arm(decisions: list[dict], by_ep: dict[str, dict], sys_text: str) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for d in decisions:
        if d.get("judgement") == "refuted":
            skipped += 1
            continue  # anti-training-poison gate (decision.schema.json)
        user = [
            "以下是引擎調校迴圈此刻的狀態。決定**下一個該做什麼**：一個具體的動作，",
            "不要複述狀態。已經被排除的路線不要再走一次。",
            "",
            C.render_state_for_decision(d, by_ep),
        ]
        rows.append({
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": "\n".join(user)},
                {"role": "assistant", "content": d["action"]},
            ],
            "kind": "state_to_next_arm",
            "source_id": d["decision_id"],
            "judgement": d["judgement"],
            "confidence": d.get("confidence"),
            "ruled_out_n": len(d.get("ruled_out") or []),
            "label": "positive",
        })
    return rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--check", action="store_true",
                    help="re-derive from the current traces and exit 1 on any drift")
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="黑箱自測（暫時根 + 陰性對照）")
    # None means "not given" -- see sft_common.reconcile for why that distinction is load-bearing.
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--system", choices=("full", "none"), default=None,
                    help="full = CONVENTIONS.md byte-identical; none = omit it. The committed "
                         "artifact used none: every row already carries its own observation")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rec = C.load_provenance(args.out_dir)
    eff, notes = C.reconcile(rec, builder=BUILDER, mode=args.system, head_chars=None,
                             val_frac=args.val_frac, check=args.check, default_head_chars=0)
    for n in notes:
        print(f"  {n}")

    full, charter_sha = C.charter()
    sys_text = C.system_text(full, eff["system"], eff["head_chars"])
    sys_bytes = len(sys_text.encode("utf-8"))

    lessons = C.load(C.LESSONS)
    decisions = C.load(C.DECISIONS)
    episodes = C.load(C.EPISODES)
    by_ep = {e["episode_id"]: e for e in episodes}

    rows = evidence_to_lesson(lessons, sys_text)
    n_l = len(rows)
    st_rows, skipped = state_to_next_arm(decisions, by_ep, sys_text)
    rows += st_rows

    train, valid = C.split(rows, eff["val_frac"])

    if args.check:
        problems, chk_notes = C.check_artifacts(args.out_dir, train, valid, rec, BUILDER, sys_text)
        for n in chk_notes:
            print(f"  {n}")
        if problems:
            print(f"  [error] {len(problems)} drift(s) in sft_prime/ against the current traces:")
            for p in problems:
                print(f"    {p}")
            print("    regenerate with: python3 agent_harness/engine_loop/sft_prime/build_sft_prime.py")
            return 1
        print(f"  sft_prime OK: train.jsonl + valid.jsonl re-derive to the same bytes; the invocation "
              f"and input hashes agree with {C.PROVENANCE_NAME}")
        return 0

    C.write_jsonl(args.out_dir / "train.jsonl", train)
    C.write_jsonl(args.out_dir / "valid.jsonl", valid)
    C.write_provenance(args.out_dir, {
        "builder": BUILDER,
        "invocation": eff,
        "split_seed": C.SPLIT_SEED,
        "system_chars": len(sys_text),
        "system_bytes": sys_bytes,
        "system_sha256_16": C.sha16(sys_text.encode("utf-8")),
        "charter_sha256_16": charter_sha,
        "input_sha256_16": C.inputs_fingerprint((C.LESSONS, C.DECISIONS, C.EPISODES)),
        "rows": {"train": len(train), "valid": len(valid)},
        "note": "reconstructed from traces/*.jsonl; this is NOT a recorded transcript",
    })

    C.report(args.out_dir, train, valid,
             f"evidence_to_lesson={n_l} state_to_next_arm={len(st_rows)} "
             f"| excluded: refuted_decisions={skipped} "
             f"superseded_lessons={sum(1 for l in lessons if l.get('superseded_by'))}")
    print(f"  system prompt: {eff['system']}, {len(sys_text)} chars / {sys_bytes} B"
          + (f", sha256[:16]={C.sha16(sys_text.encode('utf-8'))}" if sys_text else ""))
    print(f"  provenance: reconstructed from traces/*.jsonl; this is NOT a recorded transcript")
    if not rows:
        print("  [error] no rows -- nothing to train on", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------------------
# --self-test: black box. SFT_ROOT redirects C's path constants to a temp tree, and every case
# runs a real subprocess -- calling evidence_to_lesson() directly would still pass if the flag
# were never wired, or if a non-zero rc were swallowed.
#
# The generated artifacts are the only thing this file produces, and the failure it guards
# against is "two sets disagree" -- the same outside appearance as a broken pairing rule inside
# this file. So each case injects ONE cause and asserts the right one is named.
#
# ★ WATCHDOG: --out-dir defaults to this file's real parent, so a case that forgot to pass it
#   would overwrite the committed dataset with fixture data. Every run passes it explicitly, and
#   the real directory is hashed before and after the whole self-test. (Sibling
#   build_portal.py shipped exactly that bug once; a reader found it, not a test.)
# ---------------------------------------------------------------------------------------
def _dir_hash(d: Path) -> str:
    h = hashlib.sha256()
    if not d.exists():
        return "<missing>"
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def self_test() -> int:
    me = str(Path(__file__).resolve())
    real_before = _dir_hash(REAL_OUT_DIR)
    tmp = Path(tempfile.mkdtemp(prefix="sft_prime_selftest_"))
    # 佈局照抄真 repo：SFT_ROOT = engine_loop，CONVENTIONS.md 在它的上一層。
    root = tmp / "repo" / "agent_harness"
    el = root / "engine_loop"
    traces = el / "traces"
    out = el / "sft_prime"
    traces.mkdir(parents=True)
    out.mkdir(parents=True)
    (root / "CONVENTIONS.md").write_text("# fixture charter\n\nD1 some rule.\n", encoding="utf-8")

    episodes = [{"episode_id": "ep-fixture-0001", "profile": "p",
                 "arm": {"name": "a", "declared_purpose": "d"}, "goal": "g",
                 "build": {"stamp": "x"}, "usable_as_evidence": True, "caveats": []}]
    lessons = [
        {"lesson_id": "eng-fixture-0001", "rule": "rule one", "class": "measurement-hygiene",
         "because": "because one", "counterexample_observed": "ce one",
         "applies_to": ["agent_harness/example.py"], "superseded_by": None},
        {"lesson_id": "eng-fixture-0002", "rule": "rule two (superseded)", "class": "honest-bounds",
         "because": "because two", "counterexample_observed": None,
         "applies_to": ["agent_harness/other.py"], "superseded_by": "eng-fixture-0001"},
    ]
    decisions = [
        {"decision_id": "dec-fixture-0001", "question": "q1", "judgement": "sound",
         "evidence": [{"episode_id": "ep-fixture-0001", "reading": "r1"}],
         "ruled_out": [{"claim": "c", "why_false": "w"}], "action": "do the thing",
         "confidence": "high"},
        {"decision_id": "dec-fixture-0002", "question": "q2", "judgement": "refuted",
         "evidence": [{"artifact": "some/file.txt", "reading": "r2"}], "action": "do the OTHER thing",
         "confidence": "low"},
    ]

    def write_jsonl(p: Path, rows: list[dict]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def write_traces(les=None, dec=None, eps=None) -> None:
        write_jsonl(traces / "lessons.jsonl", les if les is not None else lessons)
        write_jsonl(traces / "decisions.jsonl", dec if dec is not None else decisions)
        write_jsonl(traces / "episodes.jsonl", eps if eps is not None else episodes)

    write_traces()

    env = dict(os.environ, SFT_ROOT=str(el))
    results: list[tuple[str, bool, str]] = []

    def run(*args: str) -> tuple[int, str]:
        p = subprocess.run([sys.executable, me, *args], capture_output=True, text=True, env=env)
        return p.returncode, p.stdout + p.stderr

    def rebuild() -> tuple[int, str]:
        """建置 fixture 產物。**一律顯式帶 --out-dir**（見 §WATCHDOG），並沿用已記錄的
        invocation 語意：第一次帶 --system none，之後不帶旗標就會沿用它。"""
        return run("--out-dir", str(out))

    def case(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    def rows_on_disk() -> list[dict]:
        return _read_jsonl(out / "train.jsonl") + _read_jsonl(out / "valid.jsonl")

    def tree_hash() -> str:
        h = hashlib.sha256()
        for p in sorted(root.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(root)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    # 1) 建置：兩個投影都出來，且列數 == live lessons + 非 refuted decisions
    rc, out_s = run("--out-dir", str(out), "--system", "none")
    n = len(rows_on_disk())
    case("build rc=0，寫出 train/valid/PROVENANCE",
         rc == 0 and (out / "train.jsonl").exists() and (out / "valid.jsonl").exists()
         and (out / "PROVENANCE.json").exists(), f"rc={rc}")
    case("列數 == live lessons(1) + 非 refuted decisions(1)",
         n == 2, f"n={n} out={out_s.strip()[:80]}")

    # 2) 反毒閘門：兩類排除各自發聲，而且真的不在正向集裡
    srcs = {r["source_id"] for r in rows_on_disk()}
    case("refuted decision 不入正向集（反毒閘門）",
         "dec-fixture-0002" not in srcs and "refuted_decisions=1" in out_s, f"srcs={sorted(srcs)}")
    case("superseded lesson 不入 evidence_to_lesson",
         "eng-fixture-0002" not in srcs and "superseded_lessons=1" in out_s, f"srcs={sorted(srcs)}")

    # 3) 陰性對照：乾淨狀態必須回 0
    rc, o = run("--check", "--out-dir", str(out))
    case("--check 乾淨 ⇒ rc=0（不能是永遠紅的檢查）", rc == 0, f"rc={rc} out={o.strip()[:90]}")

    # 4) --check 不寫檔
    before = tree_hash()
    run("--check", "--out-dir", str(out))
    case("--check 不寫任何檔", tree_hash() == before, "")

    # 5) 輸入前進 ⇒ 這是 staleness，不是「換一個 build」
    write_traces(les=lessons + [{"lesson_id": "eng-fixture-0003", "rule": "rule three",
                                 "class": "smoke", "because": "because three",
                                 "counterexample_observed": None,
                                 "applies_to": ["agent_harness/third.py"], "superseded_by": None}])
    rc, o = run("--check", "--out-dir", str(out))
    case("lessons 前進 ⇒ rc=1 且訊息是 INPUTS ADVANCED（staleness，不是不同 build）",
         rc == 1 and "lessons.jsonl: INPUTS ADVANCED" in o, f"rc={rc}")
    write_traces()

    # 6) 產物被手改 ⇒ 另一個原因
    rebuild()
    t = out / "train.jsonl"
    good = t.read_text(encoding="utf-8")
    t.write_text(good + '{"messages":[],"kind":"hand-edited"}\n', encoding="utf-8")
    rc, o = run("--check", "--out-dir", str(out))
    case("train.jsonl 被手改 ⇒ rc=1 且訊息是 content differs",
         rc == 1 and "train.jsonl: content differs from a re-derivation" in o, f"rc={rc}")

    # 7) PROVENANCE 不見了 ⇒ 追不回 invocation 與輸入版本
    (out / "PROVENANCE.json").unlink()
    rc, o = run("--check", "--out-dir", str(out))
    case("PROVENANCE.json 不見 ⇒ rc=1 且訊息是 cannot be traced",
         rc == 1 and "cannot be traced to an invocation" in o, f"rc={rc}")

    # 8) PROVENANCE 記的是別人寫的 ⇒ 兩個投影不能互寫彼此的目錄
    rebuild()
    prov = json.loads((out / "PROVENANCE.json").read_text(encoding="utf-8"))
    prov["builder"] = "sft_pi/build_sft_pi.py"
    (out / "PROVENANCE.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    rc, o = run("--check", "--out-dir", str(out))
    case("PROVENANCE 的 builder 是另一個投影 ⇒ rc!=0 且說明兩者不能互寫目錄",
         rc != 0 and "must not write into each other's directory" in o, f"rc={rc}")

    # 9) --check 驗的是「建置當時的樣子」，不能配矛盾的旗標。
    #    ★ 上一格把那筆 PROVENANCE 弄成了壞的，而 reconcile 對「builder 不對」是**無條件**擋的
    #      （--check 與重建都擋）⇒ 不先清掉它，這一格之後每一格都會紅，而且看起來像
    #      「--check 壞了」。第一版就是這樣：11/16，失敗集中在後面三格，實際只有一格有病。
    (out / "PROVENANCE.json").unlink()
    rc, o = run("--out-dir", str(out), "--system", "none")
    rc2, o2 = run("--check", "--out-dir", str(out), "--system", "full")
    case("--check 搭配矛盾的 --system ⇒ 拒跑（rc!=0）且說明原因",
         rc == 0 and rc2 != 0 and "cannot be combined with flags that" in o2, f"rc={rc} rc2={rc2}")
    rc, o = run("--check", "--out-dir", str(out))
    case("把上面的旗標拿掉 ⇒ rc=0（第三個陰性對照）", rc == 0, f"rc={rc}")

    # 10) 無旗標重建沿用已記錄的 invocation —— 不會靜默切成 full
    rc, o = rebuild()
    sysmode = json.loads((out / "PROVENANCE.json").read_text(encoding="utf-8"))["invocation"]["system"]
    case("無旗標重建沿用 recorded invocation（system 仍是 none，不是靜默變 full）",
         rc == 0 and sysmode == "none" and "using the recorded invocation" in o, f"rc={rc} system={sysmode}")

    # 11) 決定性：同樣的輸入重建兩次，位元必須相同（否則「valid 變了」與「資料變了」分不出來）
    rebuild()
    h1 = hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
    rebuild()
    h2 = hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
    case("同樣輸入重建兩次 ⇒ train.jsonl 位元相同（split 種子固定）", h1 == h2, f"{h1[:12]} vs {h2[:12]}")

    # 12) 空來源要拒跑（缺席不可靜默）
    write_traces(les=[])
    rc, o = run("--out-dir", str(out))
    case("空 lessons.jsonl ⇒ 拒跑（rc!=0）",
         rc != 0 and "refusing to build a dataset from nothing" in o, f"rc={rc}")
    write_traces()

    # 13) 看門狗：整個自測過程中，真 repo 的 sft_prime/ 一個位元組都不該動
    case("看門狗：真 repo 的 sft_prime/ 沒被自測寫到",
         _dir_hash(REAL_OUT_DIR) == real_before, f"{real_before[:12]} -> {_dir_hash(REAL_OUT_DIR)[:12]}")

    shutil.rmtree(tmp, ignore_errors=True)

    ok = sum(1 for _, o_, _ in results if o_)
    for name, o_, detail in results:
        print(f"  {'PASS' if o_ else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not o_ else ""))
    print(f"  {ok}/{len(results)}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
