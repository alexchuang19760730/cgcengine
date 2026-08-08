"""分布式拓撲計算器

取代 pipeline.py 中 _maybe_wrap_colossalai 的硬編碼邏輯。
從環境變數 + config 自動推導 tp/ep/pp/dp，支援雙機 TP4EP4+DP2。

設計原則：
- 不硬編碼任何 tp/dp 值
- 優先使用 config 顯式設定
- config 缺失時從 world_size 自適應推導
- 驗證 tp*ep*pp*dp == world_size
- 支援雙機場景：每機 TP4EP4，跨機 DP2
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParallelTopology:
    """並行拓撲描述符"""
    tp_size: int  # 張量並行（機內 NVLink）
    ep_size: int  # 專家並行（機內，MoE 加速關鍵）
    pp_size: int  # 流水線並行
    dp_size: int  # 資料並行（跨機 IB）
    world_size: int
    num_nodes: int
    gpus_per_node: int

    def __post_init__(self) -> None:
        total = self.tp_size * self.ep_size * self.pp_size * self.dp_size
        if total != self.world_size and self.world_size > 0:
            raise ValueError(
                f"ParallelTopology invalid: tp*ep*pp*dp={total} != world_size={self.world_size} "
                f"(tp={self.tp_size}, ep={self.ep_size}, pp={self.pp_size}, dp={self.dp_size})"
            )

    @property
    def is_cross_node_dp(self) -> bool:
        """是否啟用跨機 DP（資料並行跨節點）"""
        return self.num_nodes > 1 and self.dp_size > 1

    @property
    def is_intra_node_ep(self) -> bool:
        """EP 是否限制在單機內（推薦配置，避免跨機 all-to-all）"""
        return self.ep_size <= self.gpus_per_node

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp_size": self.tp_size,
            "ep_size": self.ep_size,
            "pp_size": self.pp_size,
            "dp_size": self.dp_size,
            "world_size": self.world_size,
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "is_cross_node_dp": self.is_cross_node_dp,
            "is_intra_node_ep": self.is_intra_node_ep,
        }


def detect_num_nodes() -> int:
    """從環境變數偵測節點數"""
    # PyTorch 標準
    nnodes = os.environ.get("NNODES") or os.environ.get("CGC_NUM_NODES")
    if nnodes and nnodes.isdigit():
        return int(nnodes)
    # 從 WORLD_SIZE / LOCAL_WORLD_SIZE 推導
    world = os.environ.get("WORLD_SIZE", "")
    local = os.environ.get("LOCAL_WORLD_SIZE", "")
    if world.isdigit() and local.isdigit() and int(local) > 0:
        return max(1, int(world) // int(local))
    return 1


def detect_gpus_per_node() -> int:
    """偵測每節點 GPU 數"""
    local = os.environ.get("LOCAL_WORLD_SIZE")
    if local and local.isdigit():
        return int(local)
    if torch_cuda_available := _torch_cuda_available():
        import torch
        return max(1, torch.cuda.device_count())
    return 8  # 預設假設 8 卡


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def compute_parallel_topology(
    *,
    world_size: Optional[int] = None,
    num_nodes: Optional[int] = None,
    gpus_per_node: Optional[int] = None,
    tp_size: Optional[int] = None,
    ep_size: Optional[int] = None,
    pp_size: Optional[int] = None,
    dp_size: Optional[int] = None,
    prefer_intra_node_ep: bool = True,
) -> ParallelTopology:
    """計算並行拓撲

    優先順序：
    1. 顯式傳入的 tp/ep/pp/dp
    2. 環境變數 CGC_PARALLEL_TP / EP / PP / DP
    3. 自適應推導（雙機 TP4EP4+DP2）

    Args:
        prefer_intra_node_ep: 若 True，確保 EP 不跨節點（避免跨機 all-to-all）
    """
    # 1. 偵測基礎環境
    if world_size is None:
        ws_env = os.environ.get("WORLD_SIZE", "1")
        world_size = int(ws_env) if ws_env.isdigit() else 1
    if num_nodes is None:
        num_nodes = detect_num_nodes()
    if gpus_per_node is None:
        gpus_per_node = detect_gpus_per_node()

    # 2. 讀取顯式設定（參數 > 環境變數）
    if tp_size is None:
        tp_env = os.environ.get("CGC_PARALLEL_TP") or os.environ.get("CGC_PARALLEL_TP_SIZE")
        tp_size = int(tp_env) if tp_env and tp_env.isdigit() else None
    if ep_size is None:
        ep_env = os.environ.get("CGC_PARALLEL_EP") or os.environ.get("CGC_PARALLEL_EP_SIZE")
        ep_size = int(ep_env) if ep_env and ep_env.isdigit() else None
    if pp_size is None:
        pp_env = os.environ.get("CGC_PARALLEL_PP") or os.environ.get("CGC_PARALLEL_PP_SIZE")
        pp_size = int(pp_env) if pp_env and pp_env.isdigit() else None
    if dp_size is None:
        dp_env = os.environ.get("CGC_PARALLEL_DP") or os.environ.get("CGC_PARALLEL_DP_SIZE")
        dp_size = int(dp_env) if dp_env and dp_env.isdigit() else None

    # 3. 預設值
    pp_size = pp_size or 1

    # 4. 自適應推導（當 tp/ep/dp 部分缺失時）
    if tp_size is None or ep_size is None or dp_size is None:
        tp_size, ep_size, dp_size = _auto_infer_topology(
            world_size=world_size,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            pp_size=pp_size,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            prefer_intra_node_ep=prefer_intra_node_ep,
        )

    topology = ParallelTopology(
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        dp_size=dp_size,
        world_size=world_size,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
    )

    if not topology.is_intra_node_ep and prefer_intra_node_ep:
        logger.warning(
            f"[ParallelTopology] EP={topology.ep_size} 跨節點（gpus_per_node={gpus_per_node}），"
            f"跨機 all-to-all 可能成為瓶頸。建議 ep_size <= {gpus_per_node}"
        )

    logger.info(f"[ParallelTopology] {topology.to_dict()}")
    return topology


def _auto_infer_topology(
    *,
    world_size: int,
    num_nodes: int,
    gpus_per_node: int,
    pp_size: int,
    tp_size: Optional[int],
    ep_size: Optional[int],
    dp_size: Optional[int],
    prefer_intra_node_ep: bool,
) -> Tuple[int, int, int]:
    """自適應推導 tp/ep/dp

    策略：
    - 單機（num_nodes=1）：TP=world_size, EP=1, DP=1（或 TP=4, EP=4 if MoE）
    - 雙機（num_nodes=2）：每機 TP=4, EP=4, 跨機 DP=2（TP4EP4+DP2 黃金配置）
    - 多機：每機 TP=gpus_per_node/2, EP=gpus_per_node/2, 跨機 DP=num_nodes
    """
    available = world_size // pp_size

    if num_nodes == 1:
        # 單機
        if tp_size is not None and ep_size is None and dp_size is None:
            ep_size = 1
            dp_size = max(1, available // tp_size)
        elif ep_size is not None and tp_size is None and dp_size is None:
            tp_size = max(1, available // ep_size)
            dp_size = max(1, available // (tp_size * ep_size))
        elif dp_size is not None and tp_size is None and ep_size is None:
            tp_size = max(1, available // dp_size)
            ep_size = 1
        else:
            # 全部缺失，用單機預設：TP=world_size, EP=1, DP=1
            # 但若 world_size >= 4 且偵測到 MoE 偏好，用 TP=world/2, EP=world/2
            if os.environ.get("CGC_MOE_PREFER_EP", "0") in {"1", "true", "yes"} and available >= 4:
                half = available // 2
                tp_size = half
                ep_size = half
                dp_size = 1
            else:
                tp_size = available
                ep_size = 1
                dp_size = 1
    else:
        # 多機：推導 TP4EP4+DP{num_nodes}
        per_node = gpus_per_node
        if tp_size is None:
            tp_size = max(1, per_node // 2)
        if ep_size is None:
            ep_size = max(1, per_node // tp_size)
            if prefer_intra_node_ep:
                ep_size = min(ep_size, per_node)  # EP 不跨節點
        if dp_size is None:
            used_per_node = tp_size * ep_size
            dp_per_node = max(1, per_node // used_per_node)
            dp_size = dp_per_node * num_nodes

        # 驗證
        if tp_size * ep_size * dp_size > available:
            # fallback：縮小 EP
            ep_size = max(1, per_node // tp_size)
            dp_size = max(1, available // (tp_size * ep_size * pp_size))

    return tp_size, ep_size, dp_size


def init_distributed_for_training(
    *,
    backend: str = "nccl",
    topology: Optional[ParallelTopology] = None,
) -> dict[str, Any]:
    """初始化 PyTorch 分布式（用於訓練）

    取代 pipeline.py 中 _maybe_wrap_colossalai 的硬編碼 init_process_group。

    Args:
        backend: nccl / gloo / mpi
        topology: 預計算的拓撲，None 則自動偵測

    Returns:
        初始化結果與拓撲資訊
    """
    import torch.distributed as dist

    if not dist.is_available():
        return {"status": "SKIP", "reason": "torch.distributed not available"}

    if dist.is_initialized():
        if topology is None:
            topology = compute_parallel_topology(world_size=dist.get_world_size())
        return {
            "status": "PASS",
            "already_initialized": True,
            "rank": int(dist.get_rank()),
            "world_size": int(dist.get_world_size()),
            "topology": topology.to_dict() if topology else None,
        }

    # 從環境變數讀取（torchrun 會自動設定）
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")

    if not master_addr or not master_port or rank is None or world_size is None:
        return {
            "status": "SKIP",
            "reason": "missing MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE (use torchrun)",
        }

    if topology is None:
        topology = compute_parallel_topology(world_size=int(world_size))

    try:
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if _torch_cuda_available():
            import torch
            torch.cuda.set_device(local_rank)
        return {
            "status": "PASS",
            "already_initialized": False,
            "backend": backend,
            "rank": int(rank),
            "world_size": int(world_size),
            "local_rank": local_rank,
            "topology": topology.to_dict(),
        }
    except Exception as e:
        return {"status": "FAIL", "error": repr(e)}
