"""rdma_cm_exchange.py — host1↔host2 RDMA out-of-band 元数据交换协议

NFSoRDMA 的 RC QP 建立需要 4 个对端元数据：
  - QPN (Queue Pair Number)
  - GID (Global Identifier, 16 bytes)
  - PSN (Packet Sequence Number)
  - LID (若 IB 而非 RoCE) / gid_index + port_num

本模块通过 TCP 实现 out-of-band 交换（生产环境也可用 RDMA CM 的
rdma_connect/rdma_accept，但 TCP 更通用，便于跨网络调试）。

协议：
  1. 双方各自创建 QP（INIT 状态），获得 local_qpn
  2. 双方各自查询本地 GID table，获得 local_gid + gid_index
  3. 双方各自生成 local_psn（随机 24-bit）
  4. 通过 TCP 交换 EndpointPair JSON:
       { "qpn": int, "gid": "xx:xx:...:xx", "psn": int,
         "gid_index": int, "port_num": int, "lid": int }
  5. 双方用对端元数据 modify QP 到 RTR + RTS

使用方式：
  # 云侧（作为 exchange server，被动等待端侧连接）
  server = RDMAEndpointExchangeServer(host="0.0.0.0", port=18515)
  cloud_pair = server.wait_for_endpoint()

  # 端侧（作为 exchange client，主动连云侧）
  client = RDMAEndpointExchangeClient()
  edge_pair = client.exchange_with_peer(
      local_endpoint=local_endpoint, peer_addr=("cloud_host1", 18515))

对应能力：nfsordma_rdma_cm_oob_exchange（host1↔host2 RDMA CM）
"""

from __future__ import annotations

import json
import os
import random
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# 协议魔数 + 版本，避免误连
_PROTO_MAGIC = "CGC_RDMA_OOB_V1"


@dataclass
class RDMAEndpoint:
    """单个 RDMA 端点的本地元数据

    用于 QP RTR 修改所需的对端信息（QPN/GID/PSN/gid_index/port_num/lid）。
    """
    qpn: int                     # local Queue Pair Number
    gid: str                     # 16-byte GID，格式 "xx:xx:...:xx" (16 段)
    gid_index: int               # GID table index
    port_num: int = 1            # 物理端口
    psn: int = 0                 # Packet Sequence Number（24-bit）
    lid: int = 0                 # IB LID（RoCEv2 用 0）
    mtu: int = 4096              # MTU（1024 / 2048 / 4096）
    device: str = ""             # 设备名（如 "rocep0s2"）
    cloud_node: str = ""         # 逻辑节点标识（端侧/云侧命名空间）

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "RDMAEndpoint":
        return cls(**json.loads(s))


@dataclass
class EndpointPair:
    """一对端点：本地 + 远端

    用于 modify QP 到 RTR/RTS 时同时引用本地和对端信息。
    """
    local: RDMAEndpoint
    remote: RDMAEndpoint


def generate_psn() -> int:
    """随机生成 24-bit PSN"""
    return random.randint(0, (1 << 24) - 1)


def query_local_gid(ctx: Any, port_num: int = 1, gid_index: Optional[int] = None) -> Tuple[str, int]:
    """查询本地 GID table

    Args:
        ctx: pyverbs Context
        port_num: 物理端口
        gid_index: 指定 GID index；None 时自动选第一个非零 GID

    Returns:
        (gid_str, gid_index)
    """
    try:
        from pyverbs.device import Context  # type: ignore
        # pyverbs 查询 GID: ctx.query_gid(port, index) -> GID 对象
        # API: ctx.query_gid(port_num, gid_index) 返回 GID，GID.__str__() 返回 "xx:xx:..."
        if gid_index is None:
            # 试 0..31，找第一个非零 GID
            for idx in range(32):
                try:
                    g = ctx.query_gid(port_num, idx)
                    gs = str(g)
                    if gs and not gs.startswith("00:00:"):
                        return gs, idx
                except Exception:
                    continue
            # 全 0，返回 index=0
            return "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00", 0
        else:
            g = ctx.query_gid(port_num, gid_index)
            return str(g), gid_index
    except Exception:
        return "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00", 0


def build_local_endpoint_from_qp(
    qp: Any,
    ctx: Any,
    *,
    port_num: int = 1,
    gid_index: Optional[int] = None,
    device: str = "",
    cloud_node: str = "",
) -> RDMAEndpoint:
    """从已创建的 QP + Context 构建本地 RDMAEndpoint

    Args:
        qp: pyverbs QP 对象（已创建，QP num 可从 qp.qp_num 获取）
        ctx: pyverbs Context
        port_num: 物理端口
        gid_index: GID index；None 时自动选
        device: 设备名
        cloud_node: 逻辑节点标识
    """
    qpn = getattr(qp, "qp_num", 0)
    gid_str, gidx = query_local_gid(ctx, port_num=port_num, gid_index=gid_index)
    return RDMAEndpoint(
        qpn=qpn,
        gid=gid_str,
        gid_index=gidx,
        port_num=port_num,
        psn=generate_psn(),
        lid=0,  # RoCEv2 默认 0
        mtu=4096,
        device=device,
        cloud_node=cloud_node,
    )


# ---------------------------------------------------------------------------
# TCP out-of-band 交换
# ---------------------------------------------------------------------------

class _OOBStream:
    """简单的 framed TCP 流：4-byte length + payload"""

    @staticmethod
    def send(sock: socket.socket, data: bytes) -> None:
        n = len(data)
        sock.sendall(n.to_bytes(4, "big") + data)

    @staticmethod
    def recv(sock: socket.socket, *, max_bytes: int = 64 * 1024) -> bytes:
        hdr = _recv_exact(sock, 4)
        if hdr is None:
            raise ConnectionError("connection closed during recv header")
        n = int.from_bytes(hdr, "big")
        if n > max_bytes:
            raise ValueError(f"frame too large: {n} > {max_bytes}")
        payload = _recv_exact(sock, n)
        if payload is None:
            raise ConnectionError("connection closed during recv payload")
        return payload


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (ConnectionResetError, OSError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _exchange_one_side(sock: socket.socket, local: RDMAEndpoint) -> RDMAEndpoint:
    """在已建立的 TCP 连接上交换一次 endpoint

    协议：先发 _PROTO_MAGIC + local.to_json()，再收对端相同结构。
    """
    # 发送：MAGIC + local endpoint JSON
    payload = json.dumps({"magic": _PROTO_MAGIC, "endpoint": asdict(local)}).encode("utf-8")
    _OOBStream.send(sock, payload)

    # 接收：对端 endpoint
    resp = _OOBStream.recv(sock)
    msg = json.loads(resp.decode("utf-8"))
    if msg.get("magic") != _PROTO_MAGIC:
        raise ValueError(f"protocol magic mismatch: got {msg.get('magic')!r}")
    return RDMAEndpoint(**msg["endpoint"])


class RDMAEndpointExchangeClient:
    """端侧：主动连接云侧 exchange server，交换 endpoint

    使用方式：
        client = RDMAEndpointExchangeClient()
        remote = client.exchange_with_peer(
            local_endpoint=local, peer_addr=("cloud_host1", 18515))
    """

    def __init__(self, *, timeout_s: float = 10.0):
        self.timeout_s = timeout_s

    def exchange_with_peer(
        self,
        *,
        local_endpoint: RDMAEndpoint,
        peer_addr: Tuple[str, int],
    ) -> RDMAEndpoint:
        """与对端交换 endpoint

        Args:
            local_endpoint: 本地 RDMAEndpoint
            peer_addr: (host, port) 对端 TCP 地址

        Returns:
            对端 RDMAEndpoint
        """
        sock = socket.create_connection(peer_addr, timeout=self.timeout_s)
        try:
            return _exchange_one_side(sock, local_endpoint)
        finally:
            sock.close()


class RDMAEndpointExchangeServer:
    """云侧：监听 TCP 端口，被动接受端侧连接，交换 endpoint

    使用方式：
        server = RDMAEndpointExchangeServer(host="0.0.0.0", port=18515)
        # 阻塞等待一个端侧连接
        remote = server.wait_for_endpoint(local_endpoint=local)
        # 或异步
        server.start(local_endpoint=local)
        remote = server.wait_for_endpoint(timeout_s=60)
        server.stop()
    """

    def __init__(self, *, host: str = "0.0.0.0", port: int = 18515, backlog: int = 4):
        self.host = host
        self.port = port
        self.backlog = backlog
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._local: Optional[RDMAEndpoint] = None
        self._remote: Optional[RDMAEndpoint] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    def wait_for_endpoint(
        self,
        *,
        local_endpoint: RDMAEndpoint,
        timeout_s: float = 60.0,
    ) -> RDMAEndpoint:
        """阻塞等待一个端侧连接，返回对端 endpoint"""
        self._local = local_endpoint
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(self.backlog)
        self._sock.settimeout(timeout_s)

        try:
            conn, _addr = self._sock.accept()
        except socket.timeout as e:
            raise TimeoutError(f"no edge connection within {timeout_s}s") from e
        try:
            return _exchange_one_side(conn, local_endpoint)
        finally:
            conn.close()
            self._sock.close()
            self._sock = None

    def start(self, *, local_endpoint: RDMAEndpoint) -> None:
        """异步启动监听"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._local = local_endpoint
        self._error = None
        self._remote = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            remote = self.wait_for_endpoint(local_endpoint=self._local, timeout_s=3600)
            with self._lock:
                self._remote = remote
        except Exception as e:
            with self._lock:
                self._error = str(e)

    def get_remote(self, *, timeout_s: float = 60.0) -> RDMAEndpoint:
        """异步模式下获取对端 endpoint"""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._remote is not None:
                    return self._remote
                if self._error:
                    raise RuntimeError(self._error)
            time.sleep(0.05)
        raise TimeoutError(f"no remote endpoint within {timeout_s}s")

    def stop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# 双向交换便捷入口（client + server 一对一匹配）
# ---------------------------------------------------------------------------

def exchange_endpoint_pair(
    *,
    local_endpoint: RDMAEndpoint,
    peer_addr: Optional[Tuple[str, int]] = None,
    listen_host: str = "0.0.0.0",
    listen_port: int = 18515,
    is_server: bool = False,
    timeout_s: float = 60.0,
) -> EndpointPair:
    """一站式交换：返回本地 + 对端 EndpointPair

    Args:
        local_endpoint: 本地 RDMAEndpoint
        peer_addr: 对端 TCP 地址（client 模式必填）
        listen_host/listen_port: server 模式监听地址
        is_server: True=server 模式，False=client 模式
        timeout_s: 超时秒

    Returns:
        EndpointPair(local, remote)
    """
    if is_server:
        server = RDMAEndpointExchangeServer(host=listen_host, port=listen_port)
        remote = server.wait_for_endpoint(
            local_endpoint=local_endpoint, timeout_s=timeout_s)
        return EndpointPair(local=local_endpoint, remote=remote)
    else:
        if peer_addr is None:
            raise ValueError("peer_addr required for client mode")
        client = RDMAEndpointExchangeClient(timeout_s=timeout_s)
        remote = client.exchange_with_peer(
            local_endpoint=local_endpoint, peer_addr=peer_addr)
        return EndpointPair(local=local_endpoint, remote=remote)


# ---------------------------------------------------------------------------
# 缓存：避免重复交换（QP 复用场景）
# ---------------------------------------------------------------------------

class EndpointCache:
    """per-cloud_node endpoint 缓存

    QP pool 复用时，避免每次 send_handoff 都重新交换。
    缓存到磁盘文件，进程重启后可恢复。
    """

    def __init__(self, cache_dir: str = "/tmp/cgc_rdma_endpoints"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, cloud_node: str) -> Optional[RDMAEndpoint]:
        p = self.cache_dir / f"{cloud_node}.json"
        if not p.exists():
            return None
        try:
            return RDMAEndpoint.from_json(p.read_text())
        except Exception:
            return None

    def put(self, cloud_node: str, endpoint: RDMAEndpoint) -> None:
        p = self.cache_dir / f"{cloud_node}.json"
        p.write_text(endpoint.to_json())

    def invalidate(self, cloud_node: str) -> None:
        p = self.cache_dir / f"{cloud_node}.json"
        if p.exists():
            p.unlink()
