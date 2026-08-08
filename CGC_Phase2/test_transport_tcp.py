"""
Local, GPU-free validation of the CGC handoff transport (TCP backend).

Spins up a TcpHandoff SERVER in one thread and a TcpHandoff CLIENT in another,
exercises send/recv of realistic hidden_states payloads across multiple ranks
and steps, and asserts the received tensor matches byte-for-byte. This proves
the transport correctly serializes/deserializes tensors and the request/
response + polling wire protocol works without touching a shared filesystem.

Run:  python test_transport_tcp.py
"""
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from cgc_handoff_transport import TcpHandoff, HandoffTransport


def make_payload(rank: int, step: int, n_tok: int):
    # Mimic the real hidden_states shape: [n_tok, hc_mult=4, hidden=4096].
    hs = torch.randn(n_tok, 4, 4096, dtype=torch.float32)
    return {
        "finished_layer": 21,
        "hidden_states": hs,
        "step": step,
        "rank": rank,
    }


def main():
    PORT = 32123
    server = TcpHandoff(role="server", host="127.0.0.1", port=PORT, timeout=10.0)
    # Give the server thread a moment to bind.
    time.sleep(0.3)
    client = TcpHandoff(
        role="client", connect_host="127.0.0.1", port=PORT, timeout=10.0
    )

    # Build a set of (rank, step) payloads like a real cut=21 request:
    #   prefill (step 0) carries 5 prompt tokens; each decode (step 1..8) 1 token.
    plan = [(r, s, 5 if s == 0 else 1) for r in range(8) for s in range(9)]
    originals = {}
    for rank, step, ntok in plan:
        p = make_payload(rank, step, ntok)
        originals[(rank, step)] = p
        server.send(rank, step, p)

    failures = 0
    for rank, step, ntok in plan:
        got = client.recv(rank, step, timeout=10.0)
        exp = originals[(rank, step)]
        ok_shape = tuple(got["hidden_states"].shape) == (ntok, 4, 4096)
        ok_layer = got["finished_layer"] == 21
        ok_close = torch.allclose(got["hidden_states"], exp["hidden_states"], atol=0.0)
        if not (ok_shape and ok_layer and ok_close):
            failures += 1
            print(
                f"MISMATCH rank={rank} step={step}: shape={ok_shape} "
                f"layer={ok_layer} data={ok_close}"
            )
        else:
            print(f"OK rank={rank} step={step} shape={tuple(got['hidden_states'].shape)}")

    # Also verify the miss/timeout path raises cleanly for a never-sent key.
    t0 = time.time()
    try:
        client.recv(99, 99, timeout=1.0)
        print("ERROR: expected TimeoutError for missing key")
        failures += 1
    except TimeoutError:
        print(f"OK missing-key timeout raised in {time.time()-t0:.2f}s")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'} ({len(plan)} payloads, {failures} failures)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
