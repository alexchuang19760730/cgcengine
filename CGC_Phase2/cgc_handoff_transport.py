"""
CGC handoff transport (M1v2)
==========================

Moves the layer-`cut` hidden_states (and, later, partial KV) from the cloud
*emit* side to the edge *resume* side, **per forward step**.

Why per-step streaming?
-----------------------
In a true layer-split (cut < last_layer) the edge runs layers cut+1..end. For
each decoded token the edge needs the cloud's residual stream AFTER layers
0..cut for THAT token, which the edge cannot reproduce by embedding the token
locally (that would skip 0..cut). So the cloud must emit the cut-layer
hidden_states on *every* forward (prefill = step 0, each decode = step k) and
the edge must load the matching step before each of its own forwards. For
greedy decoding the two step sequences line up by induction.

Backends
--------
  * "file"  -- torch `.pt` files keyed by (rank, step). Trivial, proven, but
               pollutes a shared filesystem and is NOT request-scoped (a second
               request resets the global step counter). Used for offline proofs.
  * "tcp"   -- host-side in-memory channel. The cloud runs a tiny request/
               response server; the edge pulls (rank, step) on demand.
               Request-scoped via a unique request id, so it is immune to other
               traffic on the machine. This is the production fallback.
  * "nixl"  -- zero-copy GPUDirect/UCX via `nixl.nixlAgent`. VRAM buffers are
               registered; descriptors are exchanged over a TCP side-channel
               (the self-written handshake) and the transfer is a NIXL_WRITE
               from cloud VRAM to edge VRAM. Falls back to "tcp" if NIXL init
               or the transfer fails.

Interface
---------
    transport = HandoffTransport.make(mode, role="server"|"client", ...)
    transport.send(rank, step, payload_dict)            # cloud / server side
    payload   = transport.recv(rank, step, timeout=...) # edge / client side

payload_dict = {"finished_layer": int, "hidden_states": Tensor, "step": int}
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
import traceback
from typing import Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Wire protocol for the TCP / NIXL-control channel
# ---------------------------------------------------------------------------
# Requests are 1 byte: b"G" = GET (rank, step); b"P" = PUT (notify stored).
# GET frame:  b"G" + rank(int32 BE) + step(int32 BE)
# Response:   b"OK" + nbytes(int64 BE) + payload_bytes (torch.save stream)
# On miss the server returns b"WT" and the client retries (polling) up to timeout.
_MAGIC_OK = b"OK"
_MAGIC_WAIT = b"WT"
_MAGIC_ERR = b"ER"


def _pack_get(rank: int, step: int) -> bytes:
    return b"G" + struct.pack(">ii", int(rank), int(step))


def _read_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("transport channel closed")
        buf += chunk
    return buf


class HandoffTransport:
    """Abstract base. Subclasses implement send/recv."""

    def send(self, rank: int, step: int, payload: dict) -> None:
        raise NotImplementedError

    def recv(self, rank: int, step: int, timeout: float = 30.0) -> dict:
        raise NotImplementedError

    @staticmethod
    def make(mode: str = "file", **kwargs) -> "HandoffTransport":
        mode = (mode or os.environ.get("CGC_TRANSPORT", "file")).lower()
        if mode == "file":
            return FileHandoff(
                base_path=kwargs.get("base_path")
                or os.environ.get("CGC_HANDOFF_PATH", "/data/cgc_handoff.pt")
            )
        if mode == "tcp":
            role = kwargs.pop("role", "server")
            return TcpHandoff(role=role, **kwargs)
        if mode == "nixl":
            role = kwargs.pop("role", "server")
            try:
                return NixlHandoff(role=role, **kwargs)
            except Exception as e:  # pragma: no cover - depends on IB fabric
                print(
                    f"[CGC_TRANSPORT] nixl backend unavailable ({e!r}); "
                    f"falling back to tcp",
                    flush=True,
                )
                return TcpHandoff(role=role, **kwargs)
        if mode == "mac_emit":
            # Mac→cloud reverse path: Mac is the emitter (TCP client, PUT push),
            # cloud is the receiver (TCP server, store + cloud forward pulls
            # via recv()). Symmetric reverse of NixlHandoff's cloud→edge flow.
            role = kwargs.pop("role", "emitter")
            return MacEmitHandoff(role=role, **kwargs)
        raise ValueError(f"unknown CGC_TRANSPORT mode: {mode}")


# ---------------------------------------------------------------------------
# File backend (proven, offline-proof only)
# ---------------------------------------------------------------------------
class FileHandoff(HandoffTransport):
    def __init__(self, base_path: str = "/data/cgc_handoff.pt"):
        self.base_path = base_path

    def _path(self, rank: int, step: int) -> str:
        return f"{self.base_path}.rank{rank}.step{step}"

    def send(self, rank: int, step: int, payload: dict) -> None:
        torch.save(payload, self._path(rank, step))

    def recv(self, rank: int, step: int, timeout: float = 30.0) -> dict:
        p = self._path(rank, step)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return torch.load(p, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# TCP backend (host-side fallback, request-scoped)
# ---------------------------------------------------------------------------
class _TcpStore:
    """Server-side in-memory store keyed by (rank, step), with optional seq_len matching.

    seq_len: int or None — the hidden_states seq length Mac emitted. When set,
    pop(expected_seq=N) will only return blobs whose seq_len == N. This prevents
    a 1-token warmup forward from consuming a 7-token Mac payload (step competition).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[Tuple[int, int], Tuple[bytes, Optional[int]]] = {}

    def put(self, rank: int, step: int, blob: bytes, seq_len: Optional[int] = None) -> None:
        with self._lock:
            self._data[(rank, step)] = (blob, seq_len)

    def get(self, rank: int, step: int) -> Optional[bytes]:
        with self._lock:
            item = self._data.get((rank, step))
            return item[0] if item is not None else None

    def pop(self, rank: int, step: int, expected_seq: Optional[int] = None) -> Optional[bytes]:
        """Atomically retrieve and remove (rank, step). If expected_seq is set,
        only pop if the stored seq_len matches (prevents step competition)."""
        with self._lock:
            item = self._data.get((rank, step))
            if item is None:
                return None
            blob, seq_len = item
            if expected_seq is not None and seq_len is not None and seq_len != expected_seq:
                return None  # seq mismatch — leave in store for the matching forward
            del self._data[(rank, step)]
            return blob


class TcpHandoff(HandoffTransport):
    """A request/response channel.

    role="server" (cloud/emit): runs a background thread serving a `_TcpStore`.
    role="client" (edge/resume): connects per recv() and GETs (rank, step).
    """

    def __init__(
        self,
        role: str = "server",
        host: str = "0.0.0.0",
        port: int = int(os.environ.get("CGC_TRANSPORT_TCP_PORT", "31000")),
        connect_host: str = os.environ.get("CGC_TRANSPORT_TCP_HOST", "127.0.0.1"),
        timeout: float = 60.0,
    ):
        self.role = role
        self.port = port
        self.connect_host = connect_host if role == "client" else host
        self.timeout = timeout
        self._store = _TcpStore() if role == "server" else None
        self._server = None
        self._sock = None
        if role == "server":
            self._start_server(host, port)

    # -- server side -------------------------------------------------------
    def _start_server(self, host: str, port: int) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(64)
        self._sock = srv

        def _loop():
            while True:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    break
                threading.Thread(
                    target=self._handle, args=(conn,), daemon=True
                ).start()

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        self._server = t
        print(f"[CGC_TRANSPORT] tcp server listening on {host}:{port}", flush=True)

    def _handle(self, conn: socket.socket) -> None:
        rank = step = None
        try:
            hdr = _read_exact(conn, 9)
            if hdr[:1] != b"G":
                conn.sendall(_MAGIC_ERR)
                return
            rank, step = struct.unpack(">ii", hdr[1:])
            deadline = time.time() + self.timeout
            while True:
                blob = self._store.get(rank, step)
                if blob is not None:
                    conn.sendall(_MAGIC_OK + struct.pack(">q", len(blob)) + blob)
                    # Do NOT call shutdown(SHUT_WR) here: sending a FIN through
                    # an SSH local-forward (direct-tcpip) channel makes OpenSSH
                    # tear the channel down immediately and RST the local client
                    # before the full payload is delivered. Instead we just wait
                    # for the client to consume the payload and close its side,
                    # which guarantees the whole blob is flushed through any
                    # tunnel before we close.
                    try:
                        conn.settimeout(10.0)
                        conn.recv(1)
                    except Exception:
                        pass
                    return
                if time.time() > deadline:
                    conn.sendall(_MAGIC_ERR)
                    return
                time.sleep(0.002)
        except Exception as e:
            print(
                f"[CGC_TRANSPORT][_handle] EXCEPTION rank={rank} step={step}: "
                f"{e!r}",
                flush=True,
            )
            traceback.print_exc()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def send(self, rank: int, step: int, payload: dict) -> None:
        if self.role != "server" or self._store is None:
            raise RuntimeError("tcp send() requires role='server'")
        blob = _torch_dump(payload)
        self._store.put(rank, step, blob)

    # -- client side -------------------------------------------------------
    def recv(self, rank: int, step: int, timeout: float = 30.0) -> dict:
        if self.role != "client":
            raise RuntimeError("tcp recv() requires role='client'")
        deadline = time.time() + timeout
        last_err = None
        while time.time() <= deadline:
            try:
                with socket.create_connection(
                    (self.connect_host, self.port), timeout=min(5.0, timeout)
                ) as sock:
                    sock.sendall(_pack_get(rank, step))
                    magic = _read_exact(sock, 2)
                    if magic == _MAGIC_OK:
                        (nbytes,) = struct.unpack(">q", _read_exact(sock, 8))
                        blob = _read_exact(sock, nbytes)
                        return _torch_load(blob)
                    if magic == _MAGIC_WAIT:
                        time.sleep(0.005)
                        continue
                    raise RuntimeError(f"transport error magic={magic!r}")
            except Exception as e:  # connection refused / race -> retry
                last_err = e
                time.sleep(0.02)
        raise TimeoutError(
            f"tcp recv rank={rank} step={step} timed out ({last_err!r})"
        )


def _torch_dump(payload: dict) -> bytes:
    """Serialize a payload dict (with a tensor) to bytes without touching disk."""
    import io

    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _torch_load(blob: bytes) -> dict:
    import io

    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# NIXL backend (zero-copy VRAM, real API proven byte-for-byte in
# test_nixl_xmach.py: register_memory / get_xfer_descs / get_serialized_descs /
# deserialize_descs / fetch_remote_metadata / send_local_metadata /
# initialize_xfer / transfer / check_xfer_state, acked via NIXL notifs).
# ---------------------------------------------------------------------------
_DTYPE_TO_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2,
                  torch.float64: 3}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


class NixlHandoff(HandoffTransport):
    """Zero-copy VRAM hidden_states transfer via NIXL (real, proven API).

    Model (mirrors the isolated test_nixl_xmach.py that passed byte-for-byte):
      * Cloud (role="server"/emit): one nixl_agent named "cgc_cloud" per rank,
        listening on CGC_NIXL_PORT+rank. Per send() it registers the current
        hidden_states tensor, serializes its xfer descriptor, and ships it to
        the edge inside a NIXL notification (along with rank/step/shape/dtype).
        It publishes and returns immediately (non-blocking); a background
        watcher deregisters the buffer once the edge acks the READ, so the
        buffer stays alive for the transfer without stalling the cloud.
      * Edge (role="client"/resume): one nixl_agent named "cgc_edge" per rank
        (ephemeral listen port). At init it fetches the cloud's metadata and
        publishes its own. Per recv() it waits for the cloud's desc notif for
        (rank, step), allocates+registers a same-shape recv buffer (pooled by
        shape), issues a NIXL READ into it, acks the cloud, and returns the
        buffer as hidden_states. No host bounce, no disk.

    Falls back to TcpHandoff if NIXL is unavailable at construction time
    (handled by HandoffTransport.make).
    """

    _CLOUD_NAME = "cgc_cloud"
    _EDGE_NAME = "cgc_edge"

    def __init__(
        self,
        role: str = "server",
        host: str = "0.0.0.0",
        port: int = int(os.environ.get("CGC_TRANSPORT_TCP_PORT", "31000")),
        connect_host: str = os.environ.get("CGC_NIXL_CLOUD_HOST", "172.30.132.117"),
        timeout: float = 60.0,
        rank: int = 0,
        backend: str = "UCX",
    ):
        self.role = role
        self.timeout = timeout
        self.rank = int(rank)
        self.backend = backend
        # UCX over TCP (no RDMA on RTX PRO 5000); cuda_copy moves device memory.
        os.environ.setdefault("UCX_TLS", "tcp,cuda_copy,cuda_ipc")

        import nixl  # noqa: F401  (raises ImportError early if absent)
        from nixl import nixl_agent, nixl_agent_config

        # `_cgc_get_transport` already passes port = base + rank, so use it
        # directly (do NOT add rank again). Server listens on `port`; the edge
        # connects to the cloud's VPC IP on the same `port`.
        if role == "server":
            listen_port = int(port)
            self._peer_ip = None
            self._peer_port = None
        else:
            listen_port = 0  # edge initiates; ephemeral listen port
            self._peer_ip = connect_host
            self._peer_port = int(port)

        cfg = nixl_agent_config(True, True, listen_port)
        self._agent = nixl_agent(
            self._CLOUD_NAME if role == "server" else self._EDGE_NAME, cfg
        )
        self._reg = None          # cloud: current registration handle
        self._pool = {}           # edge: shape -> (tensor, reg_handle)
        self._tcp = None          # lazy TCP fallback for non-cuda payloads
        self._pending = {}        # cloud: (rank, step) -> (reg, header)
        self._ready_set = set()   # cloud: (rank, step) edge has asked for
        self._stop = threading.Event()
        self._ack_thread = None

        if role == "client":
            # edge: exchange metadata with the cloud (edge-initiated handshake).
            # Retry until the cloud is listening (the edge may start first).
            t0 = time.time()
            while True:
                try:
                    self._agent.fetch_remote_metadata(
                        self._CLOUD_NAME, self._peer_ip, self._peer_port)
                    self._agent.send_local_metadata(
                        self._peer_ip, self._peer_port)
                    break
                except Exception as e:
                    if time.time() - t0 > timeout:
                        raise RuntimeError(
                            f"NIXL edge metadata exchange failed: {e!r}")
                    time.sleep(0.05)
            print(f"[CGC_TRANSPORT][nixl] edge rank {self.rank} connected to "
                  f"cloud {self._peer_ip}:{self._peer_port}", flush=True)
        else:
            # cloud: do NOT block on the edge. Publish descriptors on demand and
            # deregister each buffer only after the edge acks (see _ack_watcher),
            # so a request that arrives before the edge is up cannot deadlock
            # server startup. The edge connects (fetches our metadata) at its own
            # init, after which send_notif() reaches it.
            self._ack_thread = threading.Thread(
                target=self._ack_watcher, daemon=True)
            self._ack_thread.start()
            print(f"[CGC_TRANSPORT][nixl] cloud rank {self.rank} ready "
                  f"(listening {listen_port})", flush=True)

    def _tcp_fallback(self):
        if self._tcp is None:
            self._tcp = TcpHandoff(
                role=self.role,
                host="0.0.0.0",
                port=int(os.environ.get("CGC_TRANSPORT_TCP_PORT", "31000")) + self.rank,
                connect_host=os.environ.get("CGC_TRANSPORT_TCP_HOST", "127.0.0.1"),
                timeout=self.timeout,
            )
        return self._tcp

    def send(self, rank: int, step: int, payload: dict) -> None:
        hs = payload["hidden_states"]
        if not (isinstance(hs, torch.Tensor) and hs.is_cuda):
            # CPU payload -> fall back to the host-side TCP channel
            self._tcp_fallback().send(rank, step, payload)
            return
        # Register the buffer and hold it until the edge acks the READ. The
        # descriptor is delivered only when the edge signals READY (see
        # _ack_watcher), so we never drop a notif on an edge that isn't
        # connected yet at emit time.
        reg = self._agent.register_memory(hs)
        descs = self._agent.get_xfer_descs([hs])
        desc_blob = self._agent.get_serialized_descs(descs)
        # Diagnostic sum is gated behind CGC_NIXL_DIAG (default off): the
        # .sum().item() does a device->host sync, which is illegal inside
        # cuda-graph capture (the emit callback fires during the captured
        # forward). Conditional expr is lazy -> .item() not evaluated when off.
        _s = (float(hs.detach().float().sum().item())
              if os.environ.get("CGC_NIXL_DIAG") else 0.0)
        if os.environ.get("CGC_NIXL_DIAG"):
            print(f"[CGC_NIXL][send] r={rank} s={step} shape={list(hs.shape)} "
                  f"dtype={hs.dtype} sum={_s:.4f} descblob={len(desc_blob)}", flush=True)
        shape = list(hs.shape)
        header = (
            struct.pack(">iiii", int(rank), int(step),
                        int(payload.get("finished_layer", -1)),
                        int(_DTYPE_TO_CODE.get(hs.dtype, 0)))
            + struct.pack(">i", len(shape))
            + struct.pack(">%di" % len(shape), *shape)
            + b"\nDESC=" + desc_blob
        )
        # Keep `hs` alive (referenced) until the edge acks the READ, so the
        # registered CUDA buffer is never freed/reused mid-transfer. Dropping
        # this reference was the root cause of garbled live transfers: the
        # isolated test kept the tensor alive for the whole function, but here
        # send() returns and the clone would be GC'd / its VRAM reused.
        self._pending[(rank, step)] = (hs, reg, header)
        # If the edge already asked for this step, deliver now.
        if (rank, step) in self._ready_set:
            try:
                self._agent.send_notif(self._EDGE_NAME, header)
                print(f"[CGC_NIXL][send] notif sent r={rank} s={step}", flush=True)
            except Exception as _e:
                print(f"[CGC_NIXL][send] notif FAILED r={rank} s={step}: {_e!r}", flush=True)

    def _ack_watcher(self):
        """Background: deliver descriptors on READY, deregister on ACK.

        pending entries are (hs, reg, header): hs is kept referenced so its
        registered CUDA memory stays valid until the edge finishes the READ.
        """
        while not self._stop.is_set():
            try:
                notifs = self._agent.get_new_notifs()
                for n in notifs.get(self._EDGE_NAME, []):
                    if n.startswith(b"READY") and len(n) >= 13:
                        r, s = struct.unpack(">ii", n[5:13])
                        self._ready_set.add((r, s))
                        entry = self._pending.get((r, s))
                        if entry is not None:
                            try:
                                self._agent.send_notif(self._EDGE_NAME, entry[2])
                                print(f"[CGC_NIXL][watcher] sent desc r={r} s={s}", flush=True)
                            except Exception as _e:
                                print(f"[CGC_NIXL][watcher] desc send FAILED r={r} s={s}: {_e!r}", flush=True)
                    elif len(n) >= 8:
                        r, s = struct.unpack(">ii", n[:8])
                        entry = self._pending.pop((r, s), None)
                        self._ready_set.discard((r, s))
                        if entry is not None:
                            try:
                                self._agent.deregister_memory(entry[1])
                            except Exception:
                                pass
            except Exception:
                pass
            time.sleep(0.005)

    def recv(self, rank: int, step: int, timeout: float = 30.0) -> dict:
        # Tell the cloud we are ready to pull this step's hidden_states; the
        # cloud delivers the descriptor only after this signal (so it never
        # drops a notif on an edge that connects after emit time). Re-assert
        # READY periodically in case the cloud missed the first one.
        def _send_ready():
            try:
                self._agent.send_notif(
                    self._CLOUD_NAME, b"READY" + struct.pack(">ii", rank, step))
            except Exception:
                pass

        _send_ready()
        deadline = time.time() + timeout
        desc_blob = None
        shape = None
        finished_layer = -1
        dtype_code = 0
        _poll = 0
        while desc_blob is None:
            notifs = self._agent.get_new_notifs()
            for n in notifs.get(self._CLOUD_NAME, []):
                if len(n) < 16:
                    continue
                print(f"[CGC_NIXL][recv] got notif r={rank} s={step} len={len(n)}", flush=True)
                r, s, finished_layer, dtype_code = struct.unpack(">iiii", n[:16])
                if r != rank or s != step:
                    continue
                off = 16
                ndim = struct.unpack(">i", n[off:off + 4])[0]
                off += 4
                dims = struct.unpack(">%di" % ndim, n[off:off + 4 * ndim])
                off += 4 * ndim
                shape = tuple(dims)
                _, _, db = n[off:].partition(b"\nDESC=")
                desc_blob = db
                break
            if desc_blob is not None:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"NIXL edge recv rank={rank} step={step} timeout")
            _poll += 1
            if _poll % 20 == 0:
                _send_ready()
            time.sleep(0.005)

        # The cloud registers a FRESH VRAM region on EVERY emit (after this
        # edge's one-time init handshake), so the edge's cached remote metadata
        # is stale for this step. Re-fetch the cloud's current metadata so the
        # target descriptor's memory region is visible to our READ below.
        # Without this, initialize_xfer raises NIXL_ERR_NOT_FOUND ("NOT_FOUND").
        try:
            self._agent.fetch_remote_metadata(
                self._CLOUD_NAME, self._peer_ip, self._peer_port)
        except Exception as _e:
            print(f"[CGC_NIXL][recv] re-fetch metadata FAILED "
                  f"r={rank} s={step}: {_e!r}", flush=True)
        _m0 = time.time()
        while not self._agent.check_remote_metadata(self._CLOUD_NAME):
            if time.time() - _m0 > self.timeout:
                break
            time.sleep(0.005)

        # allocate + register a recv buffer of the same shape/dtype (pooled)
        if shape not in self._pool:
            buf = torch.empty(shape, dtype=_CODE_TO_DTYPE.get(dtype_code, torch.float32),
                              device="cuda")
            reg = self._agent.register_memory(buf)
            self._pool[shape] = (buf, reg)
        buf, reg = self._pool[shape]

        target_descs = self._agent.deserialize_descs(desc_blob)
        local_descs = self._agent.get_xfer_descs([buf])
        handle = self._agent.initialize_xfer(
            "READ", local_descs, target_descs, self._CLOUD_NAME, "xfer")
        state = self._agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError(f"NIXL READ post failed rank={rank} step={step}")
        t1 = time.time()
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "ERR":
                raise RuntimeError(f"NIXL READ error rank={rank} step={step}")
            if state == "DONE":
                break
            if time.time() - t1 > timeout:
                raise TimeoutError(f"NIXL READ timeout rank={rank} step={step}")
            time.sleep(0.005)
        # ack the cloud (8-byte rank,step) so it can deregister.
        # Diagnostic sum gated behind CGC_NIXL_DIAG (default off) -> no
        # device->host sync during cuda-graph capture on the edge decode path.
        _bs = (float(buf.detach().float().sum().item())
               if os.environ.get("CGC_NIXL_DIAG") else 0.0)
        if os.environ.get("CGC_NIXL_DIAG"):
            print(f"[CGC_NIXL][recv] DONE r={rank} s={step} shape={list(buf.shape)} "
                  f"dtype={buf.dtype} sum={_bs:.4f}", flush=True)
        self._agent.send_notif(self._CLOUD_NAME, struct.pack(">ii", rank, step))
        return {"finished_layer": finished_layer, "hidden_states": buf}


# ---------------------------------------------------------------------------
# Mac→cloud reverse handoff (layer-split: Mac runs first P layers, emits
# hidden_P; cloud resumes from layer P). Symmetric reverse of NixlHandoff.
#
# Why a separate transport?
#   * NixlHandoff.send() requires hs.is_cuda and the nixl package — Mac has
#     neither (Metal/MLX, unified memory, no CUDA, no nixl wheel). The NIXL
#     zero-copy path is cloud→edge only (CUDA→CUDA).
#   * TcpHandoff's wire protocol is GET-only (client=edge pulls from
#     server=cloud), so it cannot express "Mac pushes to cloud".
#   * MacEmitHandoff adds a PUT frame so the Mac (TCP client) can push
#     hidden_P to the cloud (TCP server), and the cloud's own forward process
#     reads from the in-memory store via recv() (no wire GET needed — same
#     process). This mirrors NixlHandoff's "emit returns immediately, receiver
#     pulls" semantics but over TCP and with the direction reversed.
#
# Topology:
#   Mac (role="emitter")  ──PUT (rank,step,payload)──►  Cloud (role="receiver")
#   client, connects to cloud:port                        server, listens on port,
#                                                          stores payload, cloud
#                                                          forward calls recv()
# ---------------------------------------------------------------------------
class MacEmitHandoff(HandoffTransport):
    """Mac→cloud reverse hidden_states transport (TCP, PUT-based).

    role="emitter"  (Mac):   TCP client. send(rank, step, payload) opens a
                             connection to the cloud receiver, writes a PUT
                             frame, and waits for b"OK". Retries on transient
                             connection errors up to `timeout`.
    role="receiver" (cloud): TCP server with a `_TcpStore`. On PUT it stores
                             the blob keyed by (rank, step). The cloud's resume
                             forward calls recv(rank, step) in-process, which
                             pops the blob and deserializes it. No wire GET.

    Wire frame (PUT):
        b"P" + rank(int32 BE) + step(int32 BE) + nbytes(int64 BE) + payload
    Response: b"OK" (stored) | b"ER" (error)

    The payload is the same dict shape as NixlHandoff:
        {"finished_layer": int, "hidden_states": Tensor, "step": int, ...}
    serialized via torch.save into a BytesIO (host memory on Mac, copied to
    host on cloud — no zero-copy since Mac is not a CUDA peer).
    """

    _MAGIC_PUT = b"P"

    def __init__(
        self,
        role: str = "emitter",
        host: str = "0.0.0.0",
        port: int = int(os.environ.get("CGC_MAC_EMIT_PORT", "31010")),
        connect_host: str = os.environ.get(
            "CGC_MAC_EMIT_CLOUD_HOST", "127.0.0.1"
        ),
        timeout: float = 60.0,
    ):
        self.role = role
        self.port = int(port)
        self.connect_host = connect_host
        self.timeout = timeout
        # Only the cloud (receiver) keeps a store; the Mac (emitter) is
        # stateless beyond the socket it opens per send().
        self._store: Optional[_TcpStore] = _TcpStore() if role == "receiver" else None
        self._server = None
        self._sock = None
        if role == "receiver":
            self._start_server(host, self.port)

    # -- cloud / receiver side --------------------------------------------
    def _start_server(self, host: str, port: int) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(64)
        self._sock = srv

        def _loop():
            while True:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    break
                threading.Thread(
                    target=self._handle, args=(conn,), daemon=True
                ).start()

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        self._server = t
        print(
            f"[CGC_TRANSPORT][mac_emit] cloud receiver listening on {host}:{port}",
            flush=True,
        )

    def _handle(self, conn: socket.socket) -> None:
        rank = step = None
        try:
            hdr = _read_exact(conn, 13)  # b"P" + rank(int32) + step(int32) + seq_len(int32)
            if hdr[:1] != self._MAGIC_PUT:
                conn.sendall(_MAGIC_ERR)
                return
            rank, step, seq_len = struct.unpack(">iii", hdr[1:13])
            (nbytes,) = struct.unpack(">q", _read_exact(conn, 8))
            blob = _read_exact(conn, nbytes)
            self._store.put(rank, step, blob, seq_len if seq_len >= 0 else None)
            conn.sendall(_MAGIC_OK)
            # Wait for the client to close so the full ACK is flushed through
            # any SSH/tunnel layer (same lesson as TcpHandoff._handle).
            try:
                conn.settimeout(5.0)
                conn.recv(1)
            except Exception:
                pass
        except Exception as e:
            print(
                f"[CGC_TRANSPORT][mac_emit] _handle EXCEPTION rank={rank} "
                f"step={step}: {e!r}",
                flush=True,
            )
            try:
                conn.sendall(_MAGIC_ERR)
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def recv(self, rank: int, step: int, timeout: float = 30.0, expected_seq: Optional[int] = None) -> dict:
        """Cloud-side: pop the payload the Mac pushed for (rank, step).

        If expected_seq is set, only pop payloads whose hidden_states seq_len
        matches — prevents a 1-token warmup forward from consuming a 7-token
        Mac payload (step competition). The mismatched payload stays in store
        for the matching forward to pop.

        Blocks until match arrives or `timeout` elapses.
        """
        if self.role != "receiver" or self._store is None:
            raise RuntimeError("mac_emit recv() requires role='receiver'")
        deadline = time.time() + timeout
        while time.time() <= deadline:
            blob = self._store.pop(rank, step, expected_seq=expected_seq)
            if blob is not None:
                return _torch_load(blob)
            time.sleep(0.002)
        raise TimeoutError(
            f"mac_emit recv rank={rank} step={step} expected_seq={expected_seq} "
            f"timed out (Mac did not push within {timeout}s)"
        )

    # -- Mac / emitter side -----------------------------------------------
    def send(self, rank: int, step: int, payload: dict) -> None:
        """Mac-side: PUT the payload to the cloud receiver. Non-blocking from
        the cloud's perspective — send() returns once the cloud ACKs storage,
        and the cloud's forward pulls via recv() whenever it is ready."""
        if self.role != "emitter":
            raise RuntimeError("mac_emit send() requires role='emitter'")
        blob = _torch_dump(payload)
        # 提取 seq_len (hidden_states 的 seq 维度) 供 cloud store seq-aware pop
        # 防止 1-token warmup forward 消费 7-token Mac payload (step 竞争)
        # hidden_states 可能是 3D [batch, seq, hidden] 或 2D [seq, hidden]
        seq_len = -1
        hs = payload.get("hidden_states")
        if hs is not None and hasattr(hs, "shape"):
            seq_len = int(hs.shape[-2]) if hs.dim() >= 2 else int(hs.shape[0])
        deadline = time.time() + self.timeout
        last_err: Optional[Exception] = None
        while time.time() <= deadline:
            try:
                with socket.create_connection(
                    (self.connect_host, self.port), timeout=min(5.0, self.timeout)
                ) as sock:
                    frame = (
                        self._MAGIC_PUT
                        + struct.pack(">iii", int(rank), int(step), seq_len)
                        + struct.pack(">q", len(blob))
                        + blob
                    )
                    sock.sendall(frame)
                    magic = _read_exact(sock, 2)
                    if magic == _MAGIC_OK:
                        print(
                            f"[CGC_MAC_EMIT][send] r={rank} s={step} "
                            f"nbytes={len(blob)} seq_len={seq_len} OK", flush=True,
                        )
                        return
                    if magic == _MAGIC_ERR:
                        raise RuntimeError(f"cloud receiver rejected PUT (ER)")
                    raise RuntimeError(f"mac_emit unexpected magic={magic!r}")
            except Exception as e:
                last_err = e
                # Cloud receiver may not be up yet (Mac prefill can finish
                # before the cloud resume endpoint issues recv). Retry — the
                # store on the cloud side is durable for the connection's
                # lifetime, and send() only needs the listener to be up.
                time.sleep(0.02)
        raise TimeoutError(
            f"mac_emit send rank={rank} step={step} timed out ({last_err!r})"
        )
