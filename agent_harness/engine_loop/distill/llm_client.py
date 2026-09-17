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

WHAT THIS CLIENT DOES NOT SOLVE YET
-----------------------------------
Transport works; the SCAFFOLD IS NOW IMPLEMENTED **AND VERIFIED** (2026-09-17 17:2x -- see
"THE SMOKE THAT CLOSED THIS" below). Measured 2026-09-17 against this
repo's own server (`scripts/run_server.sh`, Nail-Qwen3.6-MTP, MTP draft acceptance 0.89):

  * `/v1/chat/completions` with a bare user message -> the model enters its thinking mode and
    emits `<think>...` until the token budget runs out (`finish_reason: length`).
  * raw `/completion` with hand-written ChatML and no generation scaffold -> pure echo: the
    question `15+27 等於多少？請只輸出答案` came straight back.

Both have the same cause, described in the template's own comments: the generation prompt needs a
COMPLETED think block AND an anchor, because "neither alone works" -- the block says thinking is
over, the anchor says start answering. `closed_loop.build_prompt()` now renders exactly that and
posts it verbatim here.

THE SMOKE THAT CLOSED THIS (2026-09-17 17:2x, two calls, ~20 s)
--------------------------------------------------------------
The `15+27` -> `42` check ran against `scripts/run_server.sh` (Nail-Qwen3.6-MTP, `CGC_SERVER_CTX=40960`).
The prompt bytes were produced by `closed_loop.build_prompt()` itself -- never transcribed by hand:

  * POSITIVE (with the scaffold) -> `42` (gen_tok=3, elapsed 11.5 s).
  * NEGATIVE (same prompt, generation segment reduced to a bare `<|im_start|>assistant\n`)
    -> the model enters its thinking mode, emits `<think>Here's a thinking process: ...` and
       spends the whole 64-token budget without answering (gen_tok=64, elapsed 8.2 s).

★ The negative control is the half that makes this evidence: without it, "42" could simply mean the
question was easy. The negative reproduces exactly the failure mode recorded above, which is why the
scaffold (completed think block + anchor) is the thing being tested.
★ Those timings are COLD -- the model had just been mapped and the machine was swapping. They are
**not** performance figures; do not quote them.
`closed_loop_selftest.py` group H still checks the string's STRUCTURE -- markers, order, anchor --
and **structure is not behaviour**. The smoke is what covered behaviour, and for one question only.

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
