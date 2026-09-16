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
        [--system full|head|none] [--head-bytes N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sft_common as C  # noqa: E402

HEAD_BYTES = 12000  # a workable default when the charter is too long for the training context


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
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--system", choices=("full", "head", "none"), default="full",
                    help="full = byte-identical CONVENTIONS.md (the plan's requirement); "
                         "head = first --head-bytes; none = omitted")
    ap.add_argument("--head-bytes", type=int, default=HEAD_BYTES)
    args = ap.parse_args()

    full, sys_sha = C.charter()
    if args.system == "full":
        sys_text = full
    elif args.system == "head":
        sys_text = full[: args.head_bytes]
    else:
        sys_text = ""

    decisions = C.load(C.DECISIONS)
    episodes = C.load(C.EPISODES)
    by_ep = {e["episode_id"]: e for e in episodes}

    rows, used, n_unpaired = trajectories(decisions, by_ep, sys_text)
    if not rows:
        print("  [error] no decision cites a real episode -- nothing to build. "
              "This is not a bug: it means the traces are still all-artifact evidence.",
              file=sys.stderr)
        return 1

    train, valid = C.split(rows, args.val_frac)
    C.write_jsonl(args.out_dir / "train.jsonl", train)
    C.write_jsonl(args.out_dir / "valid.jsonl", valid)

    n_calls = sum(r["n_tool_calls"] for r in rows)
    C.report(args.out_dir, train, valid,
             f"trajectories={len(rows)} tool_calls={n_calls} "
             f"| episodes used={len(used)}/{len(episodes)} "
             f"decisions with no episode evidence={n_unpaired}")
    print(f"  system prompt: {args.system}, {len(sys_text)} B"
          + (f", sha256[:16]={sys_sha}" if sys_text else ""))
    print(f"  every sample carries _provenance.reconstructed=true + the episode_ids it came from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
