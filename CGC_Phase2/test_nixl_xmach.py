#!/usr/bin/env python3
"""
Standalone cross-machine NIXL zero-copy VRAM transfer test (M1v2).

Adapted from NIXL's official basic_two_peers.py but:
  * uses CUDA tensors (real VRAM -> VRAM path, no host bounce),
  * verifies correctness via a checksum shipped in the notification
    (sum + sum-of-squares of the source tensor), so the initiator can
    confirm the transferred bytes without the source sending the payload
    over the control channel,
  * targets the CGC topology: `target` runs on the cloud (Host2, listens on
    its VPC IP), `initiator` runs on the edge (Host1) and READs the cloud's
    hidden-states-like tensor into its own VRAM.

Run:
  # on Host2 (cloud, VPC ip 172.30.132.117):
  python test_nixl_xmach.py --mode target --port 32000 --use_cuda \
      --shape 8,4,4096 --seed 1234
  # on Host1 (edge):
  python test_nixl_xmach.py --mode initiator --ip 172.30.132.117 --port 32000 \
      --use_cuda --shape 8,4,4096 --seed 1234

Exit 0 = byte-identical transfer; non-zero = failure (with reason printed).
"""
import argparse
import os
import struct
import sys
import time

import torch

# No RDMA on RTX PRO 5000 -> force UCX over TCP with cuda_copy for the
# device->host bounce. Harmless if RDMA were present.
os.environ.setdefault("UCX_TLS", "tcp,cuda_copy,cuda_ipc")

from nixl import nixl_agent, nixl_agent_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["target", "initiator"])
    p.add_argument("--ip", type=str, default="127.0.0.1",
                   help="target (cloud) VPC IP for initiator to connect to")
    p.add_argument("--port", type=int, default=32000)
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--shape", type=str, default="8,4,4096",
                   help="comma separated tensor shape (e.g. n,4,4096)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()


def main():
    a = parse_args()
    shape = tuple(int(x) for x in a.shape.split(","))

    if a.use_cuda:
        torch.set_default_device("cuda:0")
    else:
        torch.set_default_device("cpu")

    # Target listens on the given port; initiator uses an ephemeral port.
    listen_port = a.port if a.mode == "target" else 0
    # nixl_agent_config defaults backends=['UCX']; the agent auto-creates the
    # UCX backend at construction, so do NOT call create_backend again
    # (that raises NIXL_ERR_INVALID_PARAM "backend already created").
    cfg = nixl_agent_config(True, True, listen_port)
    agent = nixl_agent(a.mode, cfg)

    print(f"[{a.mode}] agent '{a.mode}' up on port {listen_port} "
          f"(cuda={a.use_cuda}, shape={shape})", flush=True)

    if a.mode == "target":
        # ---- cloud: owns the data ----
        torch.manual_seed(a.seed)
        data = torch.randn(shape, dtype=torch.float32)
        reg = agent.register_memory(data)
        if not reg:
            print("[target] register_memory failed", flush=True)
            sys.exit(1)
        xdescs = agent.get_xfer_descs([data])
        desc_str = agent.get_serialized_descs(xdescs)
        raw = data.detach().cpu().numpy().tobytes()

        # Wait until initiator's metadata has arrived, then hand over descs.
        t0 = time.time()
        while not agent.check_remote_metadata("initiator"):
            if time.time() - t0 > a.timeout:
                print("[target] timeout waiting for initiator metadata", flush=True)
                sys.exit(2)
            time.sleep(0.02)
        # Notif #1: serialized transfer descriptors. Notif #2: raw source bytes
        # (prefixed b"VERIFY:") for a rigorous byte-for-byte check on the edge.
        agent.send_notif("initiator", desc_str)
        agent.send_notif("initiator", b"VERIFY:" + raw)
        print(f"[target] sent descs + {len(raw)} verification bytes", flush=True)

        # Wait for the initiator to confirm it finished reading (count Dones
        # so multi-iter runs don't deregister memory while READs are in flight).
        remaining = a.iters
        t0 = time.time()
        while remaining > 0:
            notifs = agent.get_new_notifs()
            for n in notifs.get("initiator", []):
                if b"Done" in n:
                    remaining -= 1
                    break
            if time.time() - t0 > a.timeout:
                print(f"[target] timeout waiting for {remaining} Done", flush=True)
                sys.exit(3)
            time.sleep(0.02)
        print("[target] initiator confirmed all Done. OK", flush=True)
        agent.deregister_memory(reg)
        sys.exit(0)

    else:
        # ---- edge: pulls the data into its own VRAM ----
        buf = torch.zeros(shape, dtype=torch.float32)
        reg = agent.register_memory(buf)
        if not reg:
            print("[initiator] register_memory failed", flush=True)
            sys.exit(1)

        # Exchange metadata with the target over NIXL's own TCP control chan.
        agent.fetch_remote_metadata("target", a.ip, a.port)
        agent.send_local_metadata(a.ip, a.port)
        print(f"[initiator] fetched target metadata from {a.ip}:{a.port}", flush=True)

        # Collect two notifs from the target: the transfer descriptors, and the
        # raw source bytes (prefixed b"VERIFY:") for a byte-exact verification.
        t0 = time.time()
        desc_str = None
        raw_expected = None
        while desc_str is None or raw_expected is None:
            notifs = agent.get_new_notifs()
            for n in notifs.get("target", []):
                if n.startswith(b"VERIFY:"):
                    raw_expected = n[len(b"VERIFY:"):]
                else:
                    desc_str = n
            if time.time() - t0 > a.timeout:
                print("[initiator] timeout waiting for target descs/verify", flush=True)
                sys.exit(4)
            time.sleep(0.02)
        target_descs = agent.deserialize_descs(desc_str)

        local_descs = agent.get_xfer_descs([buf])

        t0 = time.time()
        while not agent.check_remote_metadata("target"):
            if time.time() - t0 > a.timeout:
                print("[initiator] timeout waiting for target metadata ready", flush=True)
                sys.exit(5)
            time.sleep(0.02)

        ok = True
        for it in range(a.iters):
            # refresh buf to zeros so a no-op transfer would be caught
            buf.zero_()
            handle = agent.initialize_xfer(
                "READ", local_descs, target_descs, "target", "Done")
            state = agent.transfer(handle)
            if state == "ERR":
                print(f"[initiator] iter {it}: transfer post failed", flush=True)
                ok = False
                break
            t1 = time.time()
            while True:
                state = agent.check_xfer_state(handle)
                if state == "ERR":
                    print(f"[initiator] iter {it}: transfer error", flush=True)
                    ok = False
                    break
                if state == "DONE":
                    break
                if time.time() - t1 > a.timeout:
                    print(f"[initiator] iter {it}: transfer timeout", flush=True)
                    ok = False
                    break
                time.sleep(0.005)
            if not ok:
                break
            got = buf.detach().cpu().numpy().tobytes()
            match = (got == raw_expected)
            print(f"[initiator] iter {it}: {len(got)} bytes "
                  f"{'MATCH' if match else 'MISMATCH'}", flush=True)
            if not match:
                ok = False
                break
            # tell the target we finished this read
            agent.send_notif("target", b"Done")

        if ok:
            print("[initiator] ALL ITERS MATCH. ZERO-COPY VRAM TRANSFER OK", flush=True)
            sys.exit(0)
        else:
            print("[initiator] TRANSFER FAILED", flush=True)
            sys.exit(6)


if __name__ == "__main__":
    main()
