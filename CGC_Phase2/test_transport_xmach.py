"""Cross-machine TCP transport smoke test (M1v2 de-risk).

Server (cloud side, Host2): pre-populates (rank, step) hidden_states into the
in-memory TCP store for a couple of ranks, then holds the ports open so the
client can fetch. Shapes mimic a real cut=21 request: step 0 = 5 tokens,
steps 1..8 = 1 token, dtype bfloat16, shape [n, 4, 4096].

Client (edge side, Host1): connects to the cloud's VPC IP and fetches each
(rank, step), asserting shape + byte-identical data.

Run:
  server:  python test_transport_xmach.py --role server --ranks 0 1 --port 31000 --steps 9
  client:  python test_transport_xmach.py --role client --ranks 0 1 --host 172.30.132.117 --port 31000 --steps 9
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgc_handoff_transport import HandoffTransport  # noqa: E402
import torch  # noqa: E402


def make_tensor(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 4, 4096, generator=g, dtype=torch.bfloat16)


def run_server(ranks, base_port, steps):
    ts = {
        r: HandoffTransport.make(
            "tcp", role="server", host="0.0.0.0",
            port=base_port + r, connect_host="127.0.0.1",
        )
        for r in ranks
    }
    for r in ranks:
        for s in range(steps):
            n = 5 if s == 0 else 1
            hs = make_tensor(n, r * 1000 + s)
            ts[r].send(r, s, {"finished_layer": 21, "hidden_states": hs, "step": s})
            print(f"[server] rank={r} step={s} shape={tuple(hs.shape)}", flush=True)
    print("[server] all sent; holding ports 40s for client", flush=True)
    time.sleep(40)
    print("[server] exit", flush=True)


def run_client(ranks, host, base_port, steps):
    ts = {
        r: HandoffTransport.make(
            "tcp", role="client", host="127.0.0.1",
            port=base_port + r, connect_host=host,
        )
        for r in ranks
    }
    for r in ranks:
        for s in range(steps):
            n = 5 if s == 0 else 1
            payload = ts[r].recv(r, s, timeout=25.0)
            hs = payload["hidden_states"]
            exp = make_tensor(n, r * 1000 + s)
            ok = torch.allclose(hs.float(), exp.float(), atol=1e-3)
            assert tuple(hs.shape) == (n, 4, 4096), f"shape {tuple(hs.shape)}"
            assert ok, f"data mismatch r={r} s={s}"
            print(
                f"[client] rank={r} step={s} shape={tuple(hs.shape)} match={ok}",
                flush=True,
            )
    print("[client] ALL OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["server", "client"], required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--steps", type=int, default=9)
    a = ap.parse_args()
    if a.role == "server":
        run_server(a.ranks, a.port, a.steps)
    else:
        run_client(a.ranks, a.host, a.port, a.steps)
