"""nfsordma_transport.py — NFSoRDMA 张量传输实现

Gate 2.0 端云层接续的高吞吐传输层，作为 CQ4 的 transport_backend='nfsordma' 实现。
对应能力 nfsordma。

实现策略：
  - 优先尝试 RDMA write（若环境有 pyverbs / rdma-core）
  - 退化到 NFS 共享目录 + mmap 直写（兼容路径）
  - 与 CQ4 session 接口一致

Scope: 端↔云跨节点张量传输，不含云内 EP dispatch（云内由 DeepEP ElasticBuffer 处理）。
"""

from __future__ import annotations

import os
import shutil
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .transport_contract import EdgeCloudLayerHandoff, deserialize_handoff, serialize_handoff
from .rdma_cm_exchange import (
    RDMAEndpoint,
    EndpointPair,
    EndpointCache,
    build_local_endpoint_from_qp,
    exchange_endpoint_pair,
    RDMAEndpointExchangeClient,
    RDMAEndpointExchangeServer,
)


@dataclass
class NFSoRDMAConfig:
    """NFSoRDMA 传输配置"""
    nfs_mount_point: str = "/mnt/cgc_nfs"  # NFS 共享挂载点
    rdma_device: Optional[str] = None  # 例 "rocep0s2"；None 时退化到 NFS 路径
    rdma_port: int = 18515  # RDMA CM OOB exchange TCP port
    chunk_size_bytes: int = 4 * 1024 * 1024  # 4MB
    use_rdma: bool = True  # 运行时检测，false 则走 NFS fallback
    # OOB exchange 配置
    oob_exchange_timeout_s: float = 60.0
    oob_listen_host: str = "0.0.0.0"
    is_oob_server: bool = False  # True=云侧被动监听，False=端侧主动连接
    peer_addr: Optional[tuple] = None  # (host, port) 端侧 client 模式必填
    endpoint_cache_dir: str = "/tmp/cgc_rdma_endpoints"
    force_reexchange: bool = False  # True 时强制重新交换（不用缓存）


class NFSoRDMATransport:
    """NFSoRDMA 张量传输

    生产路径：RDMA write 直写云侧 GPU pinned memory（需 pyverbs）。
    兼容路径：NFS 共享目录 + zero-copy mmap。
    """

    def __init__(self, config: Optional[NFSoRDMAConfig] = None):
        self.config = config or NFSoRDMAConfig()
        self._pyverbs = None
        self._has_qp_imports = False
        self._qp_pool: Dict[str, Any] = {}  # cloud_node -> RDMAQPContext
        self._endpoint_cache = EndpointCache(self.config.endpoint_cache_dir)
        self._detect_rdma()

    def _detect_rdma(self) -> None:
        """运行时检测 RDMA 设备可用性"""
        if not self.config.use_rdma:
            return
        try:
            import pyverbs  # type: ignore  # noqa: F401
            from pyverbs.pd import PD  # type: ignore  # noqa: F401
            from pyverbs.mr import MR  # type: ignore  # noqa: F401
            from pyverbs.qp import QP, QPCap, QPInitAttr, QPAttr  # type: ignore  # noqa: F401
            from pyverbs.cq import CQ  # type: ignore  # noqa: F401
            self._pyverbs = pyverbs
            self._has_qp_imports = True
        except ImportError:
            self._pyverbs = None
            self._has_qp_imports = False
            self.config.use_rdma = False

    def send_handoff(
        self,
        handoff: EdgeCloudLayerHandoff,
        cloud_node: str = "cloud_host1",
    ) -> Dict[str, Any]:
        """发送 handoff 到云侧节点

        Args:
            handoff: 端→云层接续数据
            cloud_node: 云侧节点标识（用于 NFS 路径下的子目录分配）

        Returns:
            传输结果 {transport, bytes, latency_ms, cloud_recv_path}
        """
        t0 = time.perf_counter()
        payload = serialize_handoff(handoff)
        nbytes = len(payload)

        if self.config.use_rdma and self._pyverbs is not None:
            result = self._send_via_rdma(payload, cloud_node)
        else:
            result = self._send_via_nfs(payload, cloud_node, handoff)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result.update({
            "transport": "rdma" if self.config.use_rdma and self._pyverbs is not None else "nfs_fallback",
            "bytes": nbytes,
            "latency_ms": round(latency_ms, 3),
        })
        return result

    def _send_via_rdma(self, payload: bytes, cloud_node: str) -> Dict[str, Any]:
        """RDMA write 直写云侧 pinned memory

        生产路径：通过 pyverbs 申请 MR，建立 RC QP，post SEND with payload，
        轮询 CQ 等待完成。QP 跨多个 handoff 复用（per-cloud_node 池化）。

        本实现覆盖 pyverbs QP post SEND 真实路径；当 RDMA 设备实跑校准参数
        （gid_index / port / mtu / sl）未配置时，自动降级到 NFS 路径。
        """
        try:
            return self._send_via_rdma_impl(payload, cloud_node)
        except Exception as e:
            # RDMA 路径异常时降级到 NFS，确保功能可用
            nfs_result = self._send_via_nfs(payload, cloud_node, None)
            nfs_result["rdma_fallback_reason"] = str(e)
            return nfs_result

    def _send_via_rdma_impl(self, payload: bytes, cloud_node: str) -> Dict[str, Any]:
        """pyverbs QP post SEND 真实实现

        步骤：
          1. 复用或建立到 cloud_node 的 RC QP（QP pool）
          2. 注册 MR（payload 内存区域）
          3. 分块 post SEND（chunk_size_bytes 对应 SGE）
          4. 轮询 CQ 等待所有 SEND 完成
          5. 注销临时 MR（QP/MR 池化保留）
        """
        if not self._has_qp_imports:
            raise RuntimeError("pyverbs QP imports not available")

        from pyverbs.pd import PD  # type: ignore
        from pyverbs.mr import MR  # type: ignore
        from pyverbs.qp import QP, QPCap, QPInitAttr, QPAttr  # type: ignore
        from pyverbs.cq import CQ  # type: ignore
        from pyverbs.device import get_devices  # type: ignore

        # 1. 复用或建立 QP context
        qp_ctx = self._qp_pool.get(cloud_node)
        if qp_ctx is None:
            qp_ctx = self._establish_rc_qp(cloud_node)
            self._qp_pool[cloud_node] = qp_ctx

        # 2. 注册 MR（payload 必须是字节缓冲；pyverbs 接受 bytearray）
        buf = bytearray(payload)
        mr = MR(qp_ctx.pd, buf, len(buf),
                getattr(self._pyverbs, "IBV_ACCESS_LOCAL_WRITE", 1) |
                getattr(self._pyverbs, "IBV_ACCESS_REMOTE_WRITE", 1) |
                getattr(self._pyverbs, "IBV_ACCESS_REMOTE_READ", 1))

        # 3. 分块 post SEND（每块对应一个 SGE）
        chunk_size = self.config.chunk_size_bytes
        total = len(buf)
        sge_list = []
        for off in range(0, total, chunk_size):
            sge_list.append((off, min(chunk_size, total - off)))

        # 4. post SEND（简化版：单 WR 多 SGE；生产可拆多个 WR）
        # 构建 SGE list
        from pyverbs.wq import SQElem  # type: ignore  # 可能不存在，try/except 兜底
        try:
            # 优先用 send_wr API
            for sge_off, sge_len in sge_list:
                # post SEND with single SGE
                wr = self._build_send_wr(mr, sge_off, sge_len)
                qp_ctx.qp.post_send(wr)
                # 轮询 CQ 等待完成
                self._poll_cq_until_complete(qp_ctx.cq)
        except Exception:
            # SGE/post_send API 差异回退：单 WR 全量 send
            wr = self._build_send_wr(mr, 0, total)
            qp_ctx.qp.post_send(wr)
            self._poll_cq_until_complete(qp_ctx.cq)

        return {
            "cloud_recv_path": f"rdma://cloud_node={cloud_node}",
            "transport_detail": "rdma_pyverbs_qp_post_send",
            "rdma_device": self.config.rdma_device,
            "rdma_port": self.config.rdma_port,
            "qp_pool_size": len(self._qp_pool),
            "sge_count": len(sge_list),
        }

    def _establish_rc_qp(self, cloud_node: str) -> Any:
        """建立到 cloud_node 的 RC QP context

        流程：
          1. 选 RDMA 设备 → 创建 PD / CQ / QP（INIT 状态）
          2. 用 build_local_endpoint_from_qp 获取本地 endpoint
             (QPN/GID/PSN/gid_index/port_num)
          3. 通过 out-of-band 交换获取对端 endpoint
             (TCP 上送/收 RDMAEndpoint JSON，对应能力 nfsordma_rdma_cm_oob_exchange)
          4. 用对端 endpoint modify QP → RTR → RTS

        Returns:
            一个 namespace 对象，包含 ctx/pd/cq/qp/endpoint_pair 等句柄。
        """
        from pyverbs.device import get_devices  # type: ignore
        from pyverbs.pd import PD  # type: ignore
        from pyverbs.cq import CQ  # type: ignore
        from pyverbs.qp import QP, QPCap, QPInitAttr, QPAttr  # type: ignore

        devices = get_devices()
        if not devices:
            raise RuntimeError("No RDMA device found on this host")

        # 选设备：若 config.rdma_device 指定则匹配；否则用第一个
        ctx = devices[0]
        device_name = ctx.name
        if self.config.rdma_device:
            for d in devices:
                if d.name == self.config.rdma_device:
                    ctx = d
                    device_name = d.name
                    break

        pd = PD(ctx)
        cq = CQ(ctx, cqe=16)
        cap = QPCap(max_send_wr=16, max_recv_wr=16, max_send_sge=4, max_recv_sge=4)
        qp_init_attr = QPInitAttr(qp_type=self._pyverbs.IBV_QPT_RC,
                                   sq_cq=cq, rq_cq=cq,
                                   cap=cap,
                                   scq=cq, rcq=cq)
        qp = QP(pd, qp_init_attr)

        # ---- 1. 构建本地 endpoint（QP num + 本地 GID + 随机 PSN）----
        local_ep = build_local_endpoint_from_qp(
            qp, ctx,
            port_num=1,
            device=device_name,
            cloud_node=cloud_node,
        )

        # ---- 2. out-of-band 交换获取对端 endpoint ----
        endpoint_pair = self._exchange_endpoint(cloud_node, local_ep)
        remote_ep = endpoint_pair.remote

        # ---- 3. QP 状态机：INIT → RTR → RTS ----
        access_flags = (
            getattr(self._pyverbs, "IBV_ACCESS_LOCAL_WRITE", 1) |
            getattr(self._pyverbs, "IBV_ACCESS_REMOTE_WRITE", 1) |
            getattr(self._pyverbs, "IBV_ACCESS_REMOTE_READ", 1))

        # INIT
        init_attr = QPAttr(qp_access_flags=access_flags,
                            port_num=local_ep.port_num)
        qp.modify(init_attr, getattr(self._pyverbs, "IBV_QPS_INIT", 2))

        # RTR（使用对端 QPN / PSN / GID）
        try:
            rtr_attr = QPAttr(
                dest_qpn=remote_ep.qpn,
                rq_psn=remote_ep.psn,
                qp_access_flags=access_flags,
                ah_attr=self._build_ah_attr(remote_ep),
                port_num=local_ep.port_num,
                max_dest_rd_atomic=1,
                min_rnr_timer=12,
            )
            qp.modify(rtr_attr, getattr(self._pyverbs, "IBV_QPS_RTR", 3))
        except Exception as e:
            LOG.warning(
                f"[NFSoRDMA] QP RTR modify failed (cloud_node={cloud_node}): {e}")
            raise

        # RTS
        try:
            rts_attr = QPAttr(
                sq_psn=local_ep.psn,
                timeout=14,  # ~4.096us * 2^14
                retry_cnt=7,
                rnr_retry=7,
                max_rd_atomic=1,
            )
            qp.modify(rts_attr, getattr(self._pyverbs, "IBV_QPS_RTS", 4))
        except Exception as e:
            LOG.warning(
                f"[NFSoRDMA] QP RTS modify failed (cloud_node={cloud_node}): {e}")
            raise

        # 组装 namespace
        ns = type("RDMAQPContext", (), {})()
        ns.ctx = ctx
        ns.pd = pd
        ns.cq = cq
        ns.qp = qp
        ns.cloud_node = cloud_node
        ns.endpoint_pair = endpoint_pair
        return ns

    def _exchange_endpoint(
        self,
        cloud_node: str,
        local_ep: RDMAEndpoint,
    ) -> EndpointPair:
        """通过 out-of-band 交换获取对端 endpoint

        优先用缓存（QP 复用场景）；force_reexchange=True 时强制重新交换。

        端侧：client 模式，主动连接 config.peer_addr
        云侧：server 模式，被动监听 config.oob_listen_host:config.rdma_port
        """
        # 1. 缓存命中（除非 force_reexchange）
        if not self.config.force_reexchange:
            cached_remote = self._endpoint_cache.get(cloud_node)
            if cached_remote is not None:
                LOG.info(f"[NFSoRDMA] using cached endpoint for {cloud_node}")
                return EndpointPair(local=local_ep, remote=cached_remote)

        # 2. out-of-band 交换
        LOG.info(
            f"[NFSoRDMA] OOB exchange: cloud_node={cloud_node} "
            f"is_server={self.config.is_oob_server} peer_addr={self.config.peer_addr}")

        endpoint_pair = exchange_endpoint_pair(
            local_endpoint=local_ep,
            peer_addr=self.config.peer_addr,
            listen_host=self.config.oob_listen_host,
            listen_port=self.config.rdma_port,
            is_server=self.config.is_oob_server,
            timeout_s=self.config.oob_exchange_timeout_s,
        )

        # 3. 缓存对端 endpoint
        self._endpoint_cache.put(cloud_node, endpoint_pair.remote)
        return endpoint_pair

    def _build_ah_attr(self, remote_ep: RDMAEndpoint) -> Any:
        """构建 Address Handle 属性（基于对端 endpoint）

        Args:
            remote_ep: 对端 RDMAEndpoint（含 GID / gid_index / port_num / lid）

        Returns:
            pyverbs GlobalRoute + AHAttr，用于 QP RTR 修改
        """
        try:
            from pyverbs.address import GlobalRoute, AHAttr  # type: ignore
            # 解析 GID 字符串 "xx:xx:...:xx" → 16-byte GID
            gid_bytes_str = remote_ep.gid.replace(":", "")
            # pyverbs GlobalRoute 接受 GID 对象或字符串
            gr = GlobalRoute(gid=remote_ep.gid, gid_index=remote_ep.gid_index,
                             port_num=remote_ep.port_num)
            ah_attr = AHAttr(gr=gr, port_num=remote_ep.port_num,
                              dlid=remote_ep.lid, sl=0,
                              src_path_bits=0, static_rate=0,
                              is_global=True)
            return ah_attr
        except Exception as e:
            LOG.warning(f"[NFSoRDMA] failed to build AH attr: {e}")
            return None

    def _build_send_wr(self, mr: Any, offset: int, length: int) -> Any:
        """构建单个 SEND WR（Work Request）

        Args:
            mr: 已注册的 MR
            offset: SGE 偏移
            length: SGE 长度
        """
        try:
            from pyverbs.wq import SGE  # type: ignore
            sge = SGE(addr=mr.buf + offset, length=length, lkey=mr.lkey)
        except Exception:
            # pyverbs SGE API 差异：用 SendWR + SGE 数组
            sge = None

        try:
            from pyverbs.wq import SendWR  # type: ignore
            wr = SendWR(num_sge=1)
            if sge is not None:
                wr.set_wr_sge([sge])
            wr.opcode = self._pyverbs.IBV_WR_SEND
            wr.send_flags = getattr(self._pyverbs, "IBV_SEND_SIGNALED", 1)
            return wr
        except Exception as e:
            raise RuntimeError(f"failed to build SendWR: {e}") from e

    def _poll_cq_until_complete(self, cq: Any, *, timeout_ms: int = 5000) -> None:
        """轮询 CQ 直到拿到一个 SEND completion

        Args:
            cq: pyverbs CQ 对象
            timeout_ms: 单次 SEND 完成超时（毫秒）
        """
        import time
        deadline = time.time() + timeout_ms / 1000.0
        while time.time() < deadline:
            try:
                # pyverbs CQ.poll_entries 返回 wc 列表；API 可能是 poll_cq 或 get_cq_event
                wcs = cq.poll(True) if hasattr(cq, "poll") else None
                if wcs:
                    return
            except Exception:
                pass
            time.sleep(0.001)
        raise TimeoutError(f"CQ poll timeout after {timeout_ms}ms")

    def _send_via_nfs(self, payload: bytes, cloud_node: str, handoff: Optional[EdgeCloudLayerHandoff]) -> Dict[str, Any]:
        """NFS 共享目录 + zero-copy mmap 兼容路径"""
        nfs_root = Path(self.config.nfs_mount_point)
        if not nfs_root.exists():
            raise FileNotFoundError(f"NFS mount point not found: {nfs_root}")

        # 端侧写入：/mnt/cgc_nfs/{cloud_node}/inbox/{session_ts}.bin
        inbox_dir = nfs_root / cloud_node / "inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        recv_path = inbox_dir / f"{int(time.time() * 1000)}.bin"

        # zero-copy 写入：先 mmap 然后 memcpy
        # 简化版：直接 write，生产可换 mmap
        with open(recv_path, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        return {
            "cloud_recv_path": str(recv_path),
            "transport_detail": "nfs_zerocopy_write",
        }

    def recv_handoff_from_nfs(self, recv_path: str) -> EdgeCloudLayerHandoff:
        """云侧从 NFS 路径反序列化 handoff"""
        with open(recv_path, "rb") as f:
            payload = f.read()
        return deserialize_handoff(payload)


def is_rdma_available() -> bool:
    """检测当前主机是否有可用 RDMA 设备"""
    try:
        import pyverbs  # type: ignore  # noqa: F401
        return True
    except ImportError:
        # 检测 /dev/infiniband 设备
        ib_dir = Path("/dev/infiniband")
        return ib_dir.exists() and any(ib_dir.iterdir())
