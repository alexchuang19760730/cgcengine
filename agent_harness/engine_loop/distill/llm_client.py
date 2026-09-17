#!/usr/bin/env python3
"""Send one prompt VERBATIM to a local llama.cpp server; print the completion.

WHY RAW COMPLETION AND NOT CHAT
-------------------------------
`closed_loop.py` records `sha256(prompt)` in every answer row and in `manifest.json`. That hash is
only evidence if the bytes it hashes are the bytes the model saw. A chat endpoint rewrites the
prompt -- it wraps it in the model's chat template, inserting role markers that never appear in the
hash. So this client posts to `/completion` (the raw endpoint) and sends the prompt untouched:
**the hash covers exactly what was evaluated.**

That distinction matters here specifically. The whole point of the comparison is that the four arms
differ by a known, measured amount of text. If the transport silently added a fixed wrapper, the
arms would still differ by the same amount, every artefact would still look right, and nothing
would report a problem. The failure would be invisible in everything we keep.

DETERMINISM IS AN EXPERIMENTAL REQUIREMENT, NOT A PREFERENCE
------------------------------------------------------------
PLAN §9's acceptance sentence is "improved AND REPRODUCIBLE". `temperature=0` plus a fixed `seed`
is what makes "the same question asked twice gets the same answer" a property of the COMPARISON
instead of a property of the model's mood. `--reps 3` exists to verify it: if the reps disagree,
`compare.json` marks that question not-comparable rather than reporting noise as an effect.

Note for whoever reads this next: `scripts/run_server.sh` deliberately does NOT force temp=0 by
default, because greedy decoding on this model has been observed to enter echo loops
(`CGC_FORCE_TEMP0` defaults to 0 for that reason). That is a real observation and it applies here
too -- but it is an observation we will SEE, because every answer is stored. Pinning the
temperature in the request (rather than relying on the server flag) means the server's setting
cannot quietly change the experiment.

WHAT GOES TO STDERR
-------------------
llama.cpp returns a `timings` block with prompt/predicted token counts and per-second rates. Those
go to stderr, and `closed_loop.py` keeps the tail of stderr in every answer row -- so each call's
cost is recorded for free, and a run that quietly got 10x slower per call is visible afterwards.

    CLOSED_LOOP_MODEL_CMD='python3 agent_harness/engine_loop/distill/llm_client.py' \\
      python3 agent_harness/engine_loop/distill/closed_loop.py --memories-scope first:8

WHAT THIS CLIENT DOES NOT SOLVE (measured 2026-09-17)
-----------------------------------------------------
Transport works. ANSWER QUALITY DOES NOT, yet. Measured against this repo's own server
(`scripts/run_server.sh`, Nail-Qwen3.6-MTP, MTP draft acceptance 0.89):

  * `/v1/chat/completions` with a bare user message -> the model enters its thinking mode and
    emits `<think>...` until the token budget runs out (`finish_reason: length`). The repo's own
    template comments describe exactly this failure and the fix (a COMPLETED think block AND an
    anchor -- "neither alone works"); the shipped `Qwen3-nothink-ChatML.jinja` does not apply the
    completed block on its no-prefill path.
  * raw `/completion` with hand-written ChatML -> pure echo: the question
    `15+27 等於多少？請只輸出答案` came straight back. Expected, in hindsight -- the server's
    think-seed injection only runs on the chat path.

So a comparison run today would produce an artefact full of degenerate answers, and `compare.json`
would report "the arms differ" quite correctly -- they would. **Getting the scaffolding right is a
prerequisite for the comparison, not a detail of it.** Fixing it means building the generation
prompt the template comments describe and verifying it against a known question (`15+27` -> `42`)
BEFORE any 96-call run.

AND THE COST, MEASURED IN THE SAME SESSION
------------------------------------------
The prompt these four arms generate is ~30,594 tokens. The server reports `total_slots: 1`, so
there is one KV cache; its log then printed

    prompt processing, n_tokens = 400, progress = 0.01, t = 59.34 s / 6.74 tokens per second

i.e. **76 minutes per call**, against ~250 t/s for this repo's own `prefill250` profile -- a 37x
gap. The cause was not the model: `vm.swapusage` read 13,674 of 14,336 MB used and `Pages free` was
58 MB, with the server's RSS at 9.8 GB on a 16 GB machine.

**Every number produced in that state would be a number about swap, not about the model -- and
nothing in the output would say so.** That is the same failure shape as `eng-mh-0003`, one level
up: a measurement whose machine state is not pinned cannot be quoted, and the more honest-looking
detail it carries (token counts, per-second rates) the less likely anyone is to ask.

Env: LLM_BASE_URL (default http://127.0.0.1:8080), LLM_MAX_TOKENS (default 512),
     LLM_SEED (default 42), LLM_TIMEOUT (default 1800), LLM_TEMPERATURE (default 0).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
SEED = int(os.environ.get("LLM_SEED", "42"))
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "1800"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))


def _rate(v: object) -> float:
    return float(v) if isinstance(v, (int, float)) else 0.0


def main() -> int:
    prompt = sys.stdin.read()
    if not prompt.strip():
        print("llm_client: empty prompt on stdin", file=sys.stderr)
        return 2

    body = json.dumps({
        "prompt": prompt,
        "n_predict": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "cache_prompt": True,   # same-prefix reuse: the charter is shared by every arm and question
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(f"{BASE}/completion", data=body,
                                headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        print(f"llm_client: HTTP {e.code} from {BASE}/completion: {detail}", file=sys.stderr)
        return 3
    except Exception as e:                                    # noqa: BLE001 -- report, never swallow
        print(f"llm_client: {type(e).__name__} talking to {BASE}/completion: {e}", file=sys.stderr)
        return 4
    elapsed = time.time() - t0

    text = payload.get("content")
    if text is None:
        print(f"llm_client: response had no 'content': {json.dumps(payload)[:300]}", file=sys.stderr)
        return 5

    t = payload.get("timings") or {}
    print(f"llm_client[elapsed={elapsed:.1f}s prompt_tok={t.get('prompt_n', '?')} "
          f"gen_tok={t.get('predicted_n', '?')} "
          f"prefill={_rate(t.get('prompt_per_second')):.0f}t/s "
          f"decode={_rate(t.get('predicted_per_second')):.0f}t/s "
          f"cached={payload.get('tokens_cached', '?')}]", file=sys.stderr)

    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
