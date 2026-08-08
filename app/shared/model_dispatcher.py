"""模型分发 + MTP draft 同步 (十步流水线 Step 7.6 + 7.7).

Step 7.6 模型分发决策:
  根据路由模式 + 模型大小 + Mac 磁盘空间,决定:
    - 是否需要下载模型到端侧
    - 下载完整模型还是部分层 (layer-split)
    - 下载源 (HuggingFace / 云端缓存 / 已有本地)

Step 7.7 MTP draft model 同步:
  云端检查是否有该模型的 MTP draft
    - 有 → 转换 MLX 格式 → 传输端侧
    - 无 → 标记需要训练 (或跳过投机)
  端侧加载 MTP draft → 用于投机 decode
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DispatchAction(str, Enum):
    """模型分发动作."""
    NONE = "none"                    # 不需要下载 (全云)
    DOWNLOAD_FULL = "download_full"  # 下载完整模型
    DOWNLOAD_PARTIAL = "download_partial"  # 下载部分层 (layer-split)
    USE_LOCAL = "use_local"          # 已有本地模型
    UPLOAD_CLOUD = "upload_cloud"    # 端侧模型上传云 (用户自定义模型)


@dataclass
class DispatchDecision:
    """模型分发决策 (Step 7.6)."""
    action: str = "none"
    model_name: str = ""
    quantization: str = "4bit"
    download_size_gb: float = 0.0
    download_layers: int = 0         # 0 = 全部, >0 = 部分
    download_source: str = ""        # huggingface / cloud_cache / local
    local_path: str = ""
    reason: str = ""
    disk_sufficient: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MTPDraftStatus:
    """MTP draft model 状态 (Step 7.7)."""
    available: bool = False          # 端侧是否有可用的 MTP draft
    model_name: str = ""
    source: str = ""                 # cloud_sync / local / none
    params_m: float = 0.0            # 参数量 (M)
    expected_accept_rate: float = 0.28
    expected_decode_boost: float = 1.0  # decode 加速倍数
    mlx_path: str = ""
    needs_training: bool = False     # 云端是否需要训练
    sync_status: str = "pending"     # pending / syncing / ready / failed / not_needed / no_draft
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ModelDispatcher:
    """模型分发决策器 (Step 7.6).

    根据路由决策 + 模型信息 + Mac 磁盘,决定模型分发策略.
    """

    # 模型下载源映射
    MODEL_SOURCES = {
        "qwen3-vl-2b": {
            "hf_repo": "mlx-community/Qwen3-VL-2B-Instruct-4bit",
            "size_gb": 1.5,
            "layers": 28,
        },
        "qwen3-vl-2b-bf16": {
            "hf_repo": "mlx-community/Qwen3-VL-2B-Instruct-bf16",
            "size_gb": 4.26,
            "layers": 28,
        },
        "qwen3-vl-30b": {
            "hf_repo": "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",
            "size_gb": 15.0,
            "layers": 48,
        },
        "deepseek-v4-flash": {
            "hf_repo": "",  # 云端 only, 不下载
            "size_gb": 300.0,
            "layers": 61,
        },
    }

    def __init__(self, hardware_info=None, models_dir: str = ""):
        self.hw = hardware_info
        self.models_dir = models_dir or os.path.expanduser("~/models")

    def decide(self, route_decision, model_info) -> DispatchDecision:
        """根据路由决策 + 模型信息决定分发动作."""
        mode = route_decision.mode
        disk_avail = self.hw.disk_available_gb if self.hw else 0
        # 去掉量化后缀匹配 (qwen3-vl-2b-4bit → qwen3-vl-2b)
        model_key = model_info.name.lower().replace(" ", "-")
        for suffix in ["-4bit", "-bf16", "-8bit", "-fp8"]:
            if model_key.endswith(suffix):
                model_key = model_key[:-len(suffix)]
                break

        # 全云 → 不需要下载
        if mode == "cloud_only":
            return DispatchDecision(
                action=DispatchAction.NONE.value,
                model_name=model_info.name,
                reason=f"全云模式,不需要端侧模型",
            )

        # 检查本地是否已有模型
        local_path = self._find_local_model(model_info.name)
        if local_path:
            return DispatchDecision(
                action=DispatchAction.USE_LOCAL.value,
                model_name=model_info.name,
                quantization=model_info.quantization,
                local_path=local_path,
                download_size_gb=0,
                reason=f"本地已有: {local_path}",
            )

        # 需要下载 → 检查磁盘空间
        source = self.MODEL_SOURCES.get(model_key, {})
        if not source or not source.get("hf_repo"):
            return DispatchDecision(
                action=DispatchAction.UPLOAD_CLOUD.value,
                model_name=model_info.name,
                reason=f"模型 {model_info.name} 无下载源 → 全云",
            )

        download_size = source["size_gb"]
        total_layers = source["layers"]

        # Layer-split → 只下载部分层
        if mode == "layer_split":
            P = route_decision.P
            per_layer_gb = download_size / total_layers
            partial_size = P * per_layer_gb

            if disk_avail < partial_size:
                return DispatchDecision(
                    action=DispatchAction.NONE.value,
                    model_name=model_info.name,
                    disk_sufficient=False,
                    reason=f"磁盘不足: 需 {partial_size:.1f}GB, 可用 {disk_avail:.1f}GB → 全云",
                )

            return DispatchDecision(
                action=DispatchAction.DOWNLOAD_PARTIAL.value,
                model_name=model_info.name,
                quantization=model_info.quantization,
                download_size_gb=partial_size,
                download_layers=P,
                download_source="huggingface",
                local_path=os.path.join(self.models_dir, f"{model_info.name}-4bit-partial"),
                disk_sufficient=True,
                reason=f"Layer-split: 下载前 {P}/{total_layers} 层 ({partial_size:.1f}GB)",
            )

        # PD分离 / 本地 → 下载完整模型
        if disk_avail < download_size + 1:  # +1GB 余量
            return DispatchDecision(
                action=DispatchAction.NONE.value,
                model_name=model_info.name,
                disk_sufficient=False,
                reason=f"磁盘不足: 需 {download_size}GB, 可用 {disk_avail:.1f}GB → 全云",
            )

        return DispatchDecision(
            action=DispatchAction.DOWNLOAD_FULL.value,
            model_name=model_info.name,
            quantization=model_info.quantization,
            download_size_gb=download_size,
            download_layers=0,  # 0 = 全部
            download_source="huggingface",
            local_path=os.path.join(self.models_dir, f"{model_info.name}-4bit"),
            disk_sufficient=True,
            reason=f"下载完整模型 ({download_size}GB)",
        )

    def _find_local_model(self, model_name: str) -> Optional[str]:
        """检查本地是否已有模型."""
        # 搜索常见路径
        search_patterns = [
            model_name.replace(" ", "-"),
            model_name.lower().replace(" ", "-"),
        ]

        for pattern in search_patterns:
            path = os.path.join(self.models_dir, pattern)
            if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                return path

        # 搜索 HF cache
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(hf_cache):
            for d in os.listdir(hf_cache):
                if pattern.replace("-", "--") in d.lower():
                    snapshots = os.path.join(hf_cache, d, "snapshots")
                    if os.path.exists(snapshots):
                        for s in os.listdir(snapshots):
                            sp = os.path.join(snapshots, s)
                            if os.path.exists(os.path.join(sp, "config.json")):
                                return sp

        return None

    def execute_download(self, decision: DispatchDecision) -> bool:
        """执行下载 (调用 huggingface_hub)."""
        if decision.action not in (DispatchAction.DOWNLOAD_FULL.value,
                                    DispatchAction.DOWNLOAD_PARTIAL.value):
            return True  # 不需要下载

        try:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

            from huggingface_hub import snapshot_download, hf_hub_download

            model_key = decision.model_name.lower().replace(" ", "-")
            source = self.MODEL_SOURCES.get(model_key, {})
            repo_id = source.get("hf_repo", "")

            if not repo_id:
                logger.error(f"[dispatch] No HF repo for {decision.model_name}")
                return False

            if decision.action == DispatchAction.DOWNLOAD_FULL:
                logger.info(f"[dispatch] Downloading full model: {repo_id} ({decision.download_size_gb}GB)")
                snapshot_download(repo_id, local_dir=decision.local_path)
            else:
                # 部分下载 (layer-split)
                # 下载 config + tokenizer + 前 P 层的 safetensors
                logger.info(f"[dispatch] Downloading partial: {decision.download_layers} layers")
                # 实际实现需要根据 index.json 选择文件
                # 简化: 下载第一个 safetensors (通常包含前几层)
                allow_patterns = ["config.json", "tokenizer*", "*.json", "model-00001-of-*"]
                snapshot_download(repo_id, local_dir=decision.local_path,
                                  allow_patterns=allow_patterns)

            logger.info(f"[dispatch] Download complete: {decision.local_path}")
            return True

        except Exception as e:
            logger.error(f"[dispatch] Download failed: {e}")
            return False


class MTPDraftSyncer:
    """MTP draft model 云→端同步器 (Step 7.7).

    云端有训练好的 MTP draft model → 转换 MLX → 传端侧.
    """

    # MTP draft model 注册表 (云端维护)
    MTP_DRAFT_REGISTRY = {
        "qwen3-vl-2b": {
            "cloud_checkpoint": "/data/mtp_head_output/mtp_head_final.pt",
            "mlx_path": "~/models/Qwen3VL-2B-MTP-Head",
            "params_m": 59.77,
            "expected_accept": 0.6,
            "trained": False,  # 是否已训练
        },
        "qwen3-vl-30b": {
            "cloud_checkpoint": "/data/mtp_head_output/mtp_head_30b_final.pt",
            "mlx_path": "~/models/Qwen3VL-30B-MTP-Head",
            "params_m": 80.0,
            "expected_accept": 0.5,
            "trained": False,
        },
        "deepseek-v4-flash": {
            "cloud_checkpoint": "/data/models/DeepSeek-V4-Flash-UD-IQ2/mtp",
            "mlx_path": "~/models/DSV4-MTP-Head",
            "params_m": 200.0,
            "expected_accept": 0.28,
            "trained": True,  # V4-Flash 自带 MTP
        },
    }

    def __init__(self, cloud_endpoint: str = "", models_dir: str = ""):
        self.cloud_endpoint = cloud_endpoint
        self.models_dir = models_dir or os.path.expanduser("~/models")

    def check_and_sync(self, model_name: str, route_mode: str) -> MTPDraftStatus:
        """检查并同步 MTP draft model."""
        # 去掉量化后缀匹配
        model_key = model_name.lower().replace(" ", "-")
        for suffix in ["-4bit", "-bf16", "-8bit", "-fp8"]:
            if model_key.endswith(suffix):
                model_key = model_key[:-len(suffix)]
                break

        # 全云模式 → MTP draft 在云端用,不需要传端侧
        if route_mode == "cloud_only":
            return MTPDraftStatus(
                available=False,
                model_name=model_name,
                source="cloud_only",
                sync_status="not_needed",
                reason="全云模式,MTP在云端使用",
            )

        # 检查注册表 (精确匹配 + 模糊匹配)
        entry = self.MTP_DRAFT_REGISTRY.get(model_key)
        if not entry:
            # 模糊匹配: qwen3-vl-30b-a3b → qwen3-vl-30b
            for reg_key, reg_val in self.MTP_DRAFT_REGISTRY.items():
                if reg_key in model_key or model_key in reg_key:
                    entry = reg_val
                    break
        if not entry:
            return MTPDraftStatus(
                available=False,
                model_name=model_name,
                source="none",
                sync_status="no_draft",
                needs_training=True,
                reason=f"无 {model_name} 的 MTP draft 注册",
            )

        # 检查是否已训练
        if not entry["trained"]:
            return MTPDraftStatus(
                available=False,
                model_name=model_name,
                source="none",
                sync_status="needs_training",
                needs_training=True,
                expected_accept_rate=entry["expected_accept"],
                params_m=entry["params_m"],
                reason=f"MTP draft 未训练 (预期 accept={entry['expected_accept']})",
            )

        # 检查端侧是否已有
        mlx_path = os.path.expanduser(entry["mlx_path"])
        if os.path.exists(mlx_path) and os.path.exists(os.path.join(mlx_path, "config.json")):
            return MTPDraftStatus(
                available=True,
                model_name=model_name,
                source="local",
                mlx_path=mlx_path,
                params_m=entry["params_m"],
                expected_accept_rate=entry["expected_accept"],
                expected_decode_boost=self._estimate_boost(entry["expected_accept"]),
                sync_status="ready",
                reason=f"端侧已有 MTP draft: {mlx_path}",
            )

        # 需要从云端同步
        return self._sync_from_cloud(model_name, entry)

    def _sync_from_cloud(self, model_name: str, entry: dict) -> MTPDraftStatus:
        """从云端同步 MTP draft (转换 + 传输)."""
        mlx_path = os.path.expanduser(entry["mlx_path"])

        try:
            # 1. 请求云端转换 + 下载
            import requests

            logger.info(f"[mtp-sync] Requesting cloud to convert MTP draft for {model_name}")

            resp = requests.post(
                f"{self.cloud_endpoint}/v1/cgc/mtp/convert",
                json={
                    "model_name": model_name,
                    "checkpoint_path": entry["cloud_checkpoint"],
                    "target_format": "mlx",
                },
                timeout=30,
            )

            if resp.status_code != 200:
                return MTPDraftStatus(
                    available=False,
                    model_name=model_name,
                    source="cloud_sync",
                    sync_status="failed",
                    needs_training=True,
                    reason=f"云端转换失败: {resp.status_code}",
                )

            result = resp.json()
            download_url = result.get("download_url", "")

            if not download_url:
                return MTPDraftStatus(
                    available=False,
                    model_name=model_name,
                    source="cloud_sync",
                    sync_status="failed",
                    reason="云端未返回下载URL",
                )

            # 2. 下载 MLX 格式 MTP draft
            logger.info(f"[mtp-sync] Downloading MTP draft from {download_url}")
            os.makedirs(mlx_path, exist_ok=True)

            resp = requests.get(download_url, stream=True, timeout=120)
            if resp.status_code != 200:
                return MTPDraftStatus(
                    available=False,
                    model_name=model_name,
                    sync_status="failed",
                    reason=f"下载失败: {resp.status_code}",
                )

            # 保存
            weights_path = os.path.join(mlx_path, "mtp_head.safetensors")
            with open(weights_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 保存 config
            config = {
                "architectures": ["MTPHead"],
                "model_type": "mtp_head",
                "hidden_size": 2048,
                "vocab_size": 151936,
                "num_hidden_layers": 1,
                "base_model": model_name,
                "params_m": entry["params_m"],
            }
            with open(os.path.join(mlx_path, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"[mtp-sync] MTP draft synced to {mlx_path}")

            return MTPDraftStatus(
                available=True,
                model_name=model_name,
                source="cloud_sync",
                mlx_path=mlx_path,
                params_m=entry["params_m"],
                expected_accept_rate=entry["expected_accept"],
                expected_decode_boost=self._estimate_boost(entry["expected_accept"]),
                sync_status="ready",
                reason=f"云端同步完成: {mlx_path}",
            )

        except Exception as e:
            logger.error(f"[mtp-sync] Sync failed: {e}")
            return MTPDraftStatus(
                available=False,
                model_name=model_name,
                source="cloud_sync",
                sync_status="failed",
                reason=f"同步异常: {e}",
            )

    def _estimate_boost(self, accept_rate: float, N: int = 21) -> float:
        """估算 decode 加速倍数."""
        # 无投机: 1.0x
        # 有投机: accept_tokens / (draft_time + verify_time) vs 1/verify_time
        # 简化: boost ≈ 1 + accept_rate × N / (1 + N × draft_overhead)
        draft_overhead = 0.05  # MTP head 极轻
        boost = 1 + accept_rate * N * draft_overhead / (1 + N * draft_overhead)
        return round(boost, 2)


def run_step_76_77(hardware_info, route_decision, model_info, cloud_endpoint: str = ""):
    """十步流水线 Step 7.6 + 7.7: 模型分发 + MTP draft 同步.

    Returns:
        (dispatch_decision, mtp_status)
    """
    print(f"\n  [7.6/11] 模型分發決策...")

    dispatcher = ModelDispatcher(hardware_info)
    dispatch = dispatcher.decide(route_decision, model_info)

    print(f"           動作: {dispatch.action}")
    print(f"           模型: {dispatch.model_name} ({dispatch.quantization})")
    print(f"           大小: {dispatch.download_size_gb}GB")
    if dispatch.action == "download_partial":
        print(f"           層數: {dispatch.download_layers} 層 (layer-split)")
    print(f"           來源: {dispatch.download_source or 'N/A'}")
    print(f"           本地: {dispatch.local_path or 'N/A'}")
    if not dispatch.disk_sufficient:
        print(f"           ⚠️ 磁盤不足 → 降級全雲")
    print(f"           原因: {dispatch.reason}")

    print(f"\n  [7.7/11] MTP draft model 同步...")

    syncer = MTPDraftSyncer(cloud_endpoint)
    mtp = syncer.check_and_sync(model_info.name, route_decision.mode)

    print(f"           可用: {'✓' if mtp.available else '✗'}")
    print(f"           來源: {mtp.source}")
    print(f"           參數: {mtp.params_m}M")
    print(f"           預期 accept: {mtp.expected_accept_rate:.0%}")
    print(f"           預期加速: {mtp.expected_decode_boost}x")
    if mtp.needs_training:
        print(f"           ⚠️ 需要訓練 MTP draft")
    print(f"           狀態: {mtp.sync_status}")
    print(f"           原因: {mtp.reason}")

    return dispatch, mtp


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516")

    print("=" * 60)
    print("模型分发 + MTP draft 同步 (Step 7.6 + 7.7)")
    print("=" * 60)

    from app.shared.hardware_sensing import detect_all
    from app.shared.route_decision import MODEL_PRESETS, compute_route

    hw = detect_all()
    print(f"硬件: {hw.cpu_brand}, {hw.available_mem_gb}GB, 磁盘={hw.disk_available_gb}GB")

    for key in ["qwen3-vl-2b-4bit", "qwen3-vl-30b-4bit", "deepseek-v4-flash"]:
        model = MODEL_PRESETS[key]
        route = compute_route(hw, model)

        print(f"\n{'='*60}")
        print(f"模型: {model.name} → 路由: {route.mode} P={route.P}")
        print(f"{'='*60}")

        dispatch, mtp = run_step_76_77(hw, route, model, "http://47.95.250.55:30001")
