#!/usr/bin/env python3
"""sft_pi — the pi-coding-agent projection: tool-call trajectories.

WHAT ONE SAMPLE IS (PLAN §6.2)
------------------------------
    system    : CONVENTIONS.md, byte-identical to the file (see sft_common.charter)
    user      : the state — the question, the evidence on hand, what is already ruled out
    assistant : a tool_call that runs ONE arm
    tool      : that arm's observation, rendered by the same renderer both projections use
    assistant : the conclusion and the action

WHY THE TOOL CALL IS RECONSTRUCTED, NOT REPLAYED -- AND WHY THAT IS STATED IN EVERY SAMPLE
-----------------------------------------------------------------------------------------
The traces do not contain tool calls. `episode` records what an arm produced, and its `arm.env`
is enough to REGENERATE the command that would produce it, but "the command that would produce
this" is not "the command that was run". Every sample therefore carries
`_provenance.reconstructed = true` and the episode_id it was derived from, so a consumer can
always go back to the authoritative record. A dataset that silently presents a reconstruction as
a transcript is the same class of error as a number quoted without its build fingerprint.

WHY THE TRAJECTORY IS PAIRED WITH A DECISION RATHER THAN AN EPISODE
------------------------------------------------------------------
An episode alone has no question and no answer; a decision alone has no observation block. The
pair is what makes a `(state -> tool call -> observation -> conclusion)` turn. Episodes that no
decision cites are counted and reported as unpaired rather than dropped silently -- they are
still training material for a different projection, and "0 unpaired" vs "we did not look" must
not look the same.

Usage:
    python3 agent_harness/engine_loop/sft_pi/build_sft_pi.py [--out-dir DIR] [--val-frac F]
        [--system full|head|none] [--head-chars N]
    python3 agent_harness/engine_loop/sft_pi/build_sft_pi.py --check

WHICH SYSTEM MODE, AND WHY A BARE RUN IS NOT `--system full`
------------------------------------------------------------
PLAN §6.2 contradicts itself here: its table says `system=CONVENTIONS.md 摘要`, its prose says the
system prompt "就是 `CONVENTIONS.md`" and byte-identical. The committed artifact was built with
`head:8000` -- the table's reading -- and `--system full` would put 90,757 B of charter into each
of the 14 trajectories instead of 8,000 characters. Neither reading is picked for you: the choice
is a flag, it is recorded in `PROVENANCE.json` beside the data, and `--check` reads that record.
A bare rebuild reproduces the recorded invocation rather than silently switching to `full`.

WHAT --check IS FOR
-------------------
Nothing detected that this projection had gone stale: it is cheaper to rebuild than to notice, so
nobody noticed. `--check` re-derives from the current traces and compares, and it separates three
states that otherwise look the same from the outside -- stale artifact, different build, and a
charter that moved under an unchanged dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sft_common as C  # noqa: E402

BUILDER = "sft_pi/build_sft_pi.py"
HEAD_CHARS = 12000  # a workable default when the charter is too long for the training context


def build_state(dec: dict, by_ep: dict[str, dict]) -> str:
    return "\n".join([
        "你是引擎調校迴圈的操作者。以下是此刻的狀態。",
        "下一步只做一件事：跑一個臂，然後讀它的觀測。不要先下結論。",
        "",
        C.render_state_for_decision(dec, by_ep),
    ])


def trajectories(decisions: list[dict], by_ep: dict[str, dict], sys_text: str) -> tuple[list[dict], set, int]:
    rows, used, unpaired_cited = [], set(), 0
    for d in decisions:
        if d.get("judgement") == "refuted":
            continue  # a refuted line of reasoning is not a demonstration (decision.schema.json)
        cited = [ev["episode_id"] for ev in d["evidence"] if ev.get("episode_id")]
        if not cited:
            unpaired_cited += 1
            continue
        # One trajectory per decision, walking its cited episodes in order.
        msgs = [{"role": "system", "content": sys_text},
                {"role": "user", "content": build_state(d, by_ep)}]
        for i, eid in enumerate(cited):
            ep = by_ep.get(eid)
            if ep is None:
                continue  # validate.py already rejects dangling evidence; belt and braces
            used.add(eid)
            call_id = f"call_{len(rows)}_{i}"
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "bash",
                                 "arguments": json.dumps({"command": C.arm_invocation(ep)}, ensure_ascii=False)},
                }],
            })
            msgs.append({"role": "tool", "tool_call_id": call_id, "name": "bash",
                         "content": C.render_obs(ep)})
        msgs.append({"role": "assistant",
                     "content": f"{d['conclusion']}\n\n下一步：{d['action']}"})
        rows.append({
            "messages": msgs,
            "kind": "arm_trajectory",
            "source_id": d["decision_id"],
            "episode_ids": cited,
            "judgement": d["judgement"],
            "n_tool_calls": len(cited),
            "_provenance": {
                "reconstructed": True,
                "from": d["decision_id"],
                "episodes": cited,
                "note": "the tool call is regenerated from the episode's arm.env; it is not a recorded command",
            },
        })
    return rows, used, unpaired_cited


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--check", action="store_true",
                    help="re-derive from the current traces and exit 1 on any drift")
    # None means "not given". Keeping that distinguishable from "given the default value" is what
    # lets a bare run reproduce the recorded invocation instead of silently flipping system mode.
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--system", choices=("full", "head", "none"), default=None)
    ap.add_argument("--head-chars", type=int, default=None, dest="head_chars",
                    help="with --system head: how many CHARACTERS of the charter to include")
    ap.add_argument("--head-bytes", type=int, default=None, dest="head_bytes",
                    help="deprecated alias for --head-chars (the value never counted bytes)")
    args = ap.parse_args()

    if args.head_bytes is not None:
        print("  [warn] --head-bytes is now --head-chars: the value has always counted CHARACTERS",
              file=sys.stderr)
        if args.head_chars is None:
            args.head_chars = args.head_bytes

    rec = C.load_provenance(args.out_dir)
    eff, notes = C.reconcile(rec, builder=BUILDER, mode=args.system, head_chars=args.head_chars,
                             val_frac=args.val_frac, check=args.check, default_head_chars=HEAD_CHARS)
    for n in notes:
        print(f"  {n}")

    full, charter_sha = C.charter()
    sys_text = C.system_text(full, eff["system"], eff["head_chars"])
    sys_bytes = len(sys_text.encode("utf-8"))

    decisions = C.load(C.DECISIONS)
    episodes = C.load(C.EPISODES)
    by_ep = {e["episode_id"]: e for e in episodes}

    rows, used, n_unpaired = trajectories(decisions, by_ep, sys_text)
    if not rows:
        print("  [error] no decision cites a real episode -- nothing to build. "
              "This is not a bug: it means the traces are still all-artifact evidence.",
              file=sys.stderr)
        return 1

    train, valid = C.split(rows, eff["val_frac"])

    if args.check:
        problems, chk_notes = C.check_artifacts(args.out_dir, train, valid, rec, BUILDER, sys_text)
        for n in chk_notes:
            print(f"  {n}")
        if problems:
            print(f"  [error] {len(problems)} drift(s) in sft_pi/ against the current traces:")
            for p in problems:
                print(f"    {p}")
            print("    regenerate with: python3 agent_harness/engine_loop/sft_pi/build_sft_pi.py")
            return 1
        print(f"  sft_pi OK: train.jsonl + valid.jsonl re-derive to the same bytes; the invocation "
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
        "input_sha256_16": C.inputs_fingerprint((C.EPISODES, C.DECISIONS)),
        "rows": {"train": len(train), "valid": len(valid)},
        "note": "reconstructed from traces/*.jsonl; the tool call is regenerated from each "
                "episode's arm.env and is not a recorded command",
    })

    n_calls = sum(r["n_tool_calls"] for r in rows)
    C.report(args.out_dir, train, valid,
             f"trajectories={len(rows)} tool_calls={n_calls} "
             f"| episodes used={len(used)}/{len(episodes)} "
             f"decisions with no episode evidence={n_unpaired}")
    print(f"  system prompt: {eff['system']}, {len(sys_text)} chars / {sys_bytes} B"
          + (f", sha256[:16]={C.sha16(sys_text.encode('utf-8'))}" if sys_text else ""))
    print(f"  every sample carries _provenance.reconstructed=true + the episode_ids it came from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
