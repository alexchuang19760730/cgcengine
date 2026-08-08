#!/usr/bin/env python3
"""Draft Registry — 多 MTP Draft 模型动态加载与管理.

管理多个 MTP Draft 模型权重, 根据 cloud model 动态加载/卸载.
支持 Tier 1 策略: 一模型一 Draft, 同 Tokenizer 为前提.

用法:
    from app.shared.draft_registry import DraftRegistry

    registry = DraftRegistry()
    registry.register("gemma4", "/data/drafts/gemma4_mtp_v1")
    registry.register("dsv4", "/data/drafts/dsv4_mtp_v1")

    draft = registry.get_draft("gemma4")
    if draft:
        tokens = draft.generate(input_ids, num_tokens=5)

Tier 0 (原生 MTP): 不需要 draft, cloud 自带 NEXTN head
Tier 1 (同 Tokenizer 专属 Draft): 从此 registry 加载
Tier 2 (TLI 映射): 降级, 不加载 draft
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class DraftEntry:
    """单个 Draft 模型注册条目."""
    model_name: str           # 目标 cloud model name (e.g. "gemma4")
    draft_path: str           # Draft 权重路径 (MLX 或 PyTorch)
    tokenizer_path: str = ""  # Tokenizer 路径 (通常与 cloud model 共享)
    vocab_size: int = 0       # 词表大小 (必须与 cloud model 一致)
    hidden_size: int = 0      # Draft model hidden size
    backend: str = "mlx"      # "mlx" / "pytorch" / "llama_cpp"
    tier: int = 1             # 0=native, 1=same_tokenizer, 2=tli_mapping
    trained: bool = False     # 是否已训练 (False = 使用 pattern fallback)
    accept_rate: float = 0.0  # 最近 accept rate (滚动统计)
    loaded: bool = False      # 是否当前已加载到内存
    load_time: float = 0.0    # 加载耗时 (秒)
    last_used: float = 0.0    # 最后使用时间戳
    availability_reason: str = ""


class DraftRegistry:
    """多 Draft 模型注册表 + 动态加载器.

    线程安全. 支持运行时切换 cloud model 时自动加载对应 Draft.
    """

    def __init__(self, max_loaded: int = 2):
        """初始化.

        Args:
            max_loaded: 同时驻留内存的 Draft 模型上限 (节省内存)
        """
        self._entries: dict[str, DraftEntry] = {}
        self._loaded_models: dict[str, object] = {}  # model_name → loaded model object
        self._lock = threading.RLock()
        self._max_loaded = max_loaded
        self._active_model: str = ""

    def register(
        self,
        model_name: str,
        draft_path: str,
        tokenizer_path: str = "",
        vocab_size: int = 0,
        hidden_size: int = 0,
        backend: str = "mlx",
        tier: int = 1,
        trained: bool = False,
    ) -> DraftEntry:
        """注册一个 Draft 模型.

        Args:
            model_name: 目标 cloud model 名称
            draft_path: Draft 权重路径
            tokenizer_path: Tokenizer 路径 (空则用 model_registry 中的)
            vocab_size: 词表大小 (必须与 cloud model 一致)
            hidden_size: Draft model hidden size
            backend: "mlx" / "pytorch" / "llama_cpp"
            tier: 0=native, 1=same_tokenizer, 2=tli_mapping
            trained: 是否已训练

        Returns:
            DraftEntry
        """
        with self._lock:
            entry = DraftEntry(
                model_name=model_name,
                draft_path=draft_path,
                tokenizer_path=tokenizer_path,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                backend=backend,
                tier=tier,
                trained=trained,
            )
            self._entries[model_name] = entry
            logger.info(f"[draft-registry] Registered: {model_name} → {draft_path} (tier={tier}, trained={trained})")
            return entry

    def register_from_registry(self, model_name: str) -> Optional[DraftEntry]:
        """从 model_registry 自动注册 (如果该模型有 draft 配置).

        Args:
            model_name: model_registry 中的模型名

        Returns:
            DraftEntry or None (如果该模型用 native MTP, 不需要 draft)
        """
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from app.shared.model_registry import get_model_config
            cfg = get_model_config(model_name)

            # 检查是否有原生 MTP (Tier 0)
            if hasattr(cfg, "has_native_mtp") and cfg.has_native_mtp:
                logger.info(f"[draft-registry] {model_name} has native MTP (Tier 0), no draft needed")
                return None

            # 检查是否有 draft 路径
            draft_path = getattr(cfg, "draft_model_path", "")
            if not draft_path:
                # 先尝试统一训练框架的标准输出目录
                try:
                    draft_path = cfg.get_mtp_output_dir()
                except Exception:
                    draft_path = ""
            if not draft_path or not os.path.exists(draft_path):
                # 再尝试旧默认路径
                draft_path = os.path.expanduser(f"~/models/drafts/{model_name}_mtp")
                if not os.path.exists(draft_path):
                    draft_path = ""
            if not draft_path:
                # 回退到 Hermes bootstrap 里声明的 draft 路径，便于状态面给出明确缺失位置。
                try:
                    from app.shared.hermes_router import Bootstrap
                    hermes_info = dict(Bootstrap.DRAFT_MODELS.get(model_name) or {})
                    hermes_path = str(hermes_info.get("draft_model_path") or "").strip()
                    if hermes_path:
                        if not os.path.isabs(hermes_path):
                            hermes_path = os.path.join(_repo_root(), hermes_path)
                        draft_path = hermes_path
                except Exception:
                    pass

            backend = "mlx"
            availability_reason = ""
            try:
                from app.shared.hermes_router import Bootstrap
                hermes_info = dict(Bootstrap.DRAFT_MODELS.get(model_name) or {})
                verify_cfg = dict(hermes_info.get("verify_loop_config") or {})
                assistant_model_path = str(verify_cfg.get("assistant_model_path") or "").strip()
                if assistant_model_path:
                    if not os.path.isabs(assistant_model_path):
                        assistant_model_path = os.path.join(_repo_root(), assistant_model_path)
                    if os.path.exists(assistant_model_path):
                        draft_path = draft_path or os.path.dirname(assistant_model_path)
                        backend = "verify_loop_assistant"
                        availability_reason = "executor_managed_assistant"
            except Exception:
                pass

            trained_artifact_exists = bool(draft_path) and os.path.exists(draft_path)
            if not trained_artifact_exists:
                try:
                    trained_artifact_exists = os.path.exists(cfg.get_checkpoint_path())
                except Exception:
                    trained_artifact_exists = False

            entry = self.register(
                model_name=model_name,
                draft_path=draft_path,
                tokenizer_path=cfg.tokenizer_path,
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                backend=backend,
                tier=1 if draft_path else 2,
                trained=trained_artifact_exists,
            )
            entry.availability_reason = availability_reason
            return entry

        except Exception as e:
            logger.error(f"[draft-registry] Failed to register from registry: {e}")
            return None

    def get_entry(self, model_name: str) -> Optional[DraftEntry]:
        """获取 Draft 注册条目 (不加载)."""
        with self._lock:
            return self._entries.get(model_name)

    def get_draft(self, model_name: str) -> Optional[object]:
        """获取已加载的 Draft 模型 (如果已加载).

        如果未加载, 尝试加载.
        """
        with self._lock:
            if model_name in self._loaded_models:
                entry = self._entries.get(model_name)
                if entry:
                    entry.last_used = time.time()
                return self._loaded_models[model_name]

            # 尝试加载
            return self._load_draft(model_name)

    def _load_draft(self, model_name: str) -> Optional[object]:
        """加载 Draft 模型到内存."""
        entry = self._entries.get(model_name)
        if not entry or not entry.trained or not entry.draft_path:
            return None

        if not os.path.exists(entry.draft_path):
            logger.warning(f"[draft-registry] Draft path not found: {entry.draft_path}")
            return None

        # 检查内存上限, 卸载最久未使用的
        while len(self._loaded_models) >= self._max_loaded:
            self._evict_oldest()

        t0 = time.time()
        try:
            model = self._load_model(entry)
            if model is not None:
                self._loaded_models[model_name] = model
                entry.loaded = True
                entry.load_time = time.time() - t0
                entry.last_used = time.time()
                logger.info(
                    f"[draft-registry] Loaded {model_name} draft in {entry.load_time:.1f}s "
                    f"(backend={entry.backend}, path={entry.draft_path})"
                )
                return model
        except Exception as e:
            logger.error(f"[draft-registry] Failed to load {model_name}: {e}")

        return None

    def _load_model(self, entry: DraftEntry) -> Optional[object]:
        """根据 backend 加载模型."""
        if entry.backend == "mlx":
            return self._load_mlx(entry)
        elif entry.backend == "verify_loop_assistant":
            return self._load_verify_loop_assistant(entry)
        elif entry.backend == "pytorch":
            return self._load_pytorch(entry)
        elif entry.backend == "llama_cpp":
            return self._load_llama_cpp(entry)
        else:
            logger.error(f"[draft-registry] Unknown backend: {entry.backend}")
            return None

    def _load_mlx(self, entry: DraftEntry) -> Optional[object]:
        """加载 MLX 格式的 Draft 模型."""
        try:
            from mlx_lm import load
            model, tokenizer = load(entry.draft_path)
            return {"model": model, "tokenizer": tokenizer, "backend": "mlx"}
        except ImportError:
            logger.warning("[draft-registry] mlx_lm not available, trying fallback")
            return None
        except Exception as e:
            logger.error(f"[draft-registry] MLX load error: {e}")
            return None

    def _load_pytorch(self, entry: DraftEntry) -> Optional[object]:
        """加载 PyTorch 格式的 Draft 模型."""
        try:
            import torch
            ckpt = torch.load(
                os.path.join(entry.draft_path, "mtp_head_kl.pt"),
                weights_only=False,
                map_location="cpu",
            )
            return {"checkpoint": ckpt, "backend": "pytorch"}
        except Exception as e:
            logger.error(f"[draft-registry] PyTorch load error: {e}")
            return None

    def _load_verify_loop_assistant(self, entry: DraftEntry) -> Optional[object]:
        """Mark assistant bundle as executor-managed instead of loading via MLX."""
        if not entry.draft_path or not os.path.exists(entry.draft_path):
            return None
        entry.availability_reason = "executor_managed_assistant"
        return {"backend": "verify_loop_assistant", "draft_path": entry.draft_path}

    def _load_llama_cpp(self, entry: DraftEntry) -> Optional[object]:
        """加载 GGUF 格式的 Draft 模型."""
        try:
            from llama_cpp import Llama
            model = Llama(model_path=entry.draft_path, n_ctx=512, n_gpu_layers=0)
            return {"model": model, "backend": "llama_cpp"}
        except ImportError:
            logger.warning("[draft-registry] llama_cpp not available")
            return None
        except Exception as e:
            logger.error(f"[draft-registry] llama_cpp load error: {e}")
            return None

    def _evict_oldest(self):
        """卸载最久未使用的 Draft 模型."""
        if not self._loaded_models:
            return
        oldest_name = min(
            self._loaded_models.keys(),
            key=lambda n: self._entries.get(n, DraftEntry("","","")).last_used,
        )
        del self._loaded_models[oldest_name]
        if oldest_name in self._entries:
            self._entries[oldest_name].loaded = False
        logger.info(f"[draft-registry] Evicted {oldest_name} (LRU)")

    def set_active(self, model_name: str) -> bool:
        """设置当前活跃的 cloud model, 自动加载对应 Draft.

        Returns:
            True if draft available (loaded or native), False if no draft
        """
        with self._lock:
            self._active_model = model_name
            entry = self._entries.get(model_name)

            if entry is None:
                # 尝试从 registry 注册
                entry = self.register_from_registry(model_name)

            if entry is None:
                # Tier 0: native MTP, no draft needed
                logger.info(f"[draft-registry] {model_name}: native MTP (Tier 0), no draft")
                return True

            if entry.tier == 0:
                logger.info(f"[draft-registry] {model_name}: native MTP (Tier 0)")
                return True

            if not entry.trained:
                logger.info(f"[draft-registry] {model_name}: draft not trained (Tier {entry.tier}), using pattern fallback")
                return False

            # 加载 Draft
            draft = self.get_draft(model_name)
            if draft is not None:
                logger.info(f"[draft-registry] {model_name}: draft loaded (Tier {entry.tier})")
                return True
            else:
                logger.warning(f"[draft-registry] {model_name}: draft load failed, using pattern fallback")
                return False

    def update_accept_rate(self, model_name: str, rate: float):
        """更新 Draft 的 accept rate (滚动统计)."""
        with self._lock:
            entry = self._entries.get(model_name)
            if entry:
                # 指数移动平均
                entry.accept_rate = 0.9 * entry.accept_rate + 0.1 * rate

    def get_status(self) -> dict:
        """获取注册表状态."""
        with self._lock:
            return {
                "active_model": self._active_model,
                "registered": [
                    {
                        "model_name": e.model_name,
                        "tier": e.tier,
                        "trained": e.trained,
                        "loaded": e.loaded,
                        "accept_rate": round(e.accept_rate, 3),
                        "backend": e.backend,
                        "draft_path": e.draft_path,
                        "draft_path_exists": bool(e.draft_path) and os.path.exists(e.draft_path),
                        "availability_reason": (
                            (e.availability_reason or "loaded")
                            if e.loaded
                            else (
                                "draft_path_missing"
                                if not e.draft_path
                                else (
                                    "draft_path_not_found"
                                    if not os.path.exists(e.draft_path)
                                    else (
                                        e.availability_reason
                                        or ("draft_untrained" if not e.trained else "draft_load_failed")
                                    )
                                )
                            )
                        ),
                    }
                    for e in self._entries.values()
                ],
                "loaded_count": len(self._loaded_models),
                "max_loaded": self._max_loaded,
            }


# === 全局单例 ===
_draft_registry: Optional[DraftRegistry] = None
_draft_registry_lock = threading.Lock()


def get_draft_registry() -> DraftRegistry:
    """获取全局 DraftRegistry 单例."""
    global _draft_registry
    if _draft_registry is None:
        with _draft_registry_lock:
            if _draft_registry is None:
                _draft_registry = DraftRegistry()
    return _draft_registry


if __name__ == "__main__":
    # 自测
    registry = get_draft_registry()

    # 注册测试条目
    registry.register("gemma4", "/data/drafts/gemma4_mtp_v1", vocab_size=262144, tier=0, trained=True)
    registry.register("dsv4", "/data/drafts/dsv4_mtp_v1", vocab_size=129280, tier=1, trained=False)
    registry.register("qwen3vl", "/data/drafts/qwen3vl_mtp_v1", vocab_size=151936, tier=1, trained=False)

    print("Draft Registry Status:")
    print(json.dumps(registry.get_status(), indent=2))

    # 测试切换
    print(f"\nSwitch to gemma4: {registry.set_active('gemma4')}")
    print(f"Switch to dsv4: {registry.set_active('dsv4')}")
