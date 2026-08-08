from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


_LAYER_RE = re.compile(r"layer[_\-]?(\d+)", re.IGNORECASE)
_EXPERT_RE = re.compile(r"expert[_\-]?(\d+)", re.IGNORECASE)

_ROUTE_LOCAL_FULL = "local_full"
_ROUTE_LAYER_SPLIT_PD = "layer_split_pd"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "off", "no", ""}


def _normalize_model_name(model_name: str) -> str:
    lowered = str(model_name or "").strip().lower()
    if "gemma" in lowered:
        return "gemma4"
    if "deepseek" in lowered or "dsv4" in lowered or "v4-flash" in lowered:
        return "dsv4"
    if "qwen" in lowered:
        return "qwen3vl"
    return lowered or "generic"


def _stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


@dataclass(frozen=True)
class ExpertBlob:
    key: str
    model: str
    layer_id: int
    expert_id: int
    path: str
    size_bytes: int
    tags: tuple[str, ...] = ()
    offset_bytes: int = 0
    io_backend: str = "file_full"


@dataclass
class ResidentExpert:
    meta: ExpertBlob
    payload: bytes
    loaded_at: float
    last_access: float
    access_count: int = 0
    hot_score: float = 0.0
    pinned: bool = False
    prefetched: bool = False
    prefetch_hits_counted: bool = False
    source: str = "request"
    load_latency_ms: float = 0.0


@dataclass
class ExpertPlan:
    enabled: bool
    reason: str
    model: str
    family: str
    route_mode: str
    # P2+ fix: 保存原 request 的 prompt_hash, advance_window 重建 plan 时沿用
    # 否则 _select_from_layer 的 seed 会变, expert routing 脱离原 thread
    prompt_hash: str = ""
    current_keys: list[str] = field(default_factory=list)
    next_keys: list[str] = field(default_factory=list)
    # P2 double-buffer: 下下个 layer window, decode current 时预先 prefetch
    next_next_keys: list[str] = field(default_factory=list)
    # P2+ pipeline: 3 windows ahead, daemon 持续 prefetch (I/O-compute overlap 细粒度)
    far_keys: list[str] = field(default_factory=list)
    current_bytes: int = 0
    cold_bytes: int = 0
    route_slots: int = 0
    route_swaps: int = 0
    route_agree: float = 1.0
    route_kl: float = 0.0
    frontier_key: str = ""
    layer_cursor: int = 0
    next_layer_cursor: int = 0
    # 所有 layer 信息 (供 daemon 动态计算更远 window)
    all_layer_ids: list[int] = field(default_factory=list)
    current_layer_ids: list[int] = field(default_factory=list)
    next_layer_ids: list[int] = field(default_factory=list)
    next_next_layer_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "model": self.model,
            "family": self.family,
            "route_mode": self.route_mode,
            "current_keys": list(self.current_keys),
            "next_keys": list(self.next_keys),
            "next_next_keys": list(self.next_next_keys),
            "far_keys": list(self.far_keys),
            "predicted_bytes_to_read_mb": round(self.current_bytes / 1024**2, 3),
            "predicted_cold_bytes_mb": round(self.cold_bytes / 1024**2, 3),
            "route_slots": int(self.route_slots),
            "route_swaps": int(self.route_swaps),
            "route_agree": round(float(self.route_agree), 3),
            "route_kl": round(float(self.route_kl), 6),
            "frontier_key": self.frontier_key,
            "layer_cursor": int(self.layer_cursor),
            "next_layer_cursor": int(self.next_layer_cursor),
            "current_layer_ids": list(self.current_layer_ids),
            "next_layer_ids": list(self.next_layer_ids),
            "next_next_layer_ids": list(self.next_next_layer_ids),
        }


@dataclass
class ExpertRequestSession:
    request_id: str
    plan: ExpertPlan
    started_at: float
    frontier_key: str = ""
    cursor_start: int = 0
    cursor_next: int = 0
    loaded_keys: list[str] = field(default_factory=list)
    prefetched_keys: list[str] = field(default_factory=list)
    # P2 double-buffer: 记录下下个 window 的 prefetch (decode current 时后台加载)
    prefetched_ahead_keys: list[str] = field(default_factory=list)
    # P2+ pipeline: 记录 far window 的 prefetch (3 windows ahead)
    prefetched_far_keys: list[str] = field(default_factory=list)
    # decode 推进通知 (可选, 外部调用 advance_window 更新)
    advance_count: int = 0
    cache_hits: int = 0
    prefetch_hits: int = 0


class FullExpertDataPlaneManager:
    def __init__(self) -> None:
        self.enabled = os.environ.get("EDGE_EXPERT_STREAMING_ENABLED", "1") == "1"
        self.dense_layer_streaming_enabled = _env_flag("EDGE_DENSE_LAYER_STREAMING_ENABLED", True)
        self.manifest_path = os.environ.get("EDGE_EXPERT_MANIFEST_PATH", "")
        self.blob_root = os.environ.get("EDGE_EXPERT_BLOB_ROOT", "")
        self.cache_route_enabled = _env_flag(
            "EDGE_EXPERT_CACHE_ROUTE",
            _env_flag("CACHE_ROUTE", True),
        )
        self.ram_budget_bytes = int(os.environ.get("EDGE_EXPERT_RAM_BUDGET_BYTES", str(8 * 1024**3)))
        self.pin_budget_bytes = int(os.environ.get("EDGE_EXPERT_PIN_BUDGET_BYTES", str(2 * 1024**3)))
        self.prefetch_workers = max(int(os.environ.get("EDGE_EXPERT_PREFETCH_WORKERS", "2") or "2"), 1)
        self.top_k = max(int(os.environ.get("EDGE_EXPERT_TOPK", "4") or "4"), 1)
        self.top_j = max(int(os.environ.get("EDGE_EXPERT_ROUTE_J", "2") or "2"), 1)
        self.top_m = max(int(os.environ.get("EDGE_EXPERT_ROUTE_M", "12") or "12"), self.top_k)
        self.route_alpha = max(float(os.environ.get("EDGE_EXPERT_ROUTE_ALPHA", "1.0") or "1.0"), 0.01)
        self.current_layer_window = max(int(os.environ.get("EDGE_EXPERT_CURRENT_LAYER_WINDOW", "6") or "6"), 1)
        self.prefetch_layer_window = max(int(os.environ.get("EDGE_EXPERT_PREFETCH_LAYER_WINDOW", "6") or "6"), 1)
        self.pin_promote_threshold = max(int(os.environ.get("EDGE_EXPERT_PIN_PROMOTE_THRESHOLD", "3") or "3"), 1)
        self.autodiscover_catalog = _env_flag("EDGE_EXPERT_AUTODISCOVER", True)
        self.persist_enabled = _env_flag("EDGE_EXPERT_PERSIST_STATE", True)
        self.persist_interval_sec = max(float(os.environ.get("EDGE_EXPERT_PERSIST_INTERVAL_SEC", "15") or "15"), 1.0)
        self.preload_hot_count = max(int(os.environ.get("EDGE_EXPERT_PRELOAD_HOT_COUNT", "8") or "8"), 0)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.state_path = os.environ.get(
            "EDGE_EXPERT_STATE_PATH",
            str(
                self.repo_root
                / "ComputeGraphCompiler-main"
                / "Output"
                / "edge_first_proxy_reports"
                / "expert_data_plane.state.json"
            ),
        )

        self._lock = threading.RLock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.prefetch_workers,
            thread_name_prefix="expert-prefetch",
        )
        self._by_model: dict[str, list[ExpertBlob]] = {}
        self._by_key: dict[str, ExpertBlob] = {}
        self._resident: dict[str, ResidentExpert] = {}
        self._inflight_prefetch: dict[str, concurrent.futures.Future[Any]] = {}
        self._family_affinity: dict[tuple[str, str], dict[str, float]] = {}
        self._frontier_cursor: dict[str, int] = {}
        self._persisted_pinned_keys: list[str] = []
        self._last_plan: dict[str, Any] = {}
        self._catalog_source = "disabled"
        self._catalog_loaded_at = 0.0
        self._last_state_persist_ts = 0.0
        self._warm_start_loaded = 0

        # P2+ pipeline: 持续性 I/O-compute overlap (background daemon)
        # daemon 定期扫描 active sessions, 持续 prefetch far window
        self._active_sessions: dict[str, "ExpertRequestSession"] = {}
        self._pipeline_stop = threading.Event()
        self._pipeline_thread: Optional[threading.Thread] = None
        self._pipeline_interval = max(
            float(os.environ.get("EDGE_EXPERT_PIPELINE_INTERVAL_SEC", "0.1") or "0.1"), 0.02
        )
        self._pipeline_lookahead_far = _env_flag(
            "EDGE_EXPERT_PIPELINE_LOOKAHEAD_FAR", True
        )

        # P3+ dense residency monitor: 持续性 dense backbone residency 管理
        # 跟踪 expert 占用 EMA, 变化超阈值时标记 reload
        self._expert_occupied_ema: float = 0.0
        self._dense_residency_hint: dict[str, Any] = {}
        self._dense_reload_pending: bool = False
        self._dense_reload_n_gpu_layers: int = -1
        self._dense_reload_threshold = max(
            float(os.environ.get("EDGE_DENSE_RELOAD_THRESHOLD", "0.15") or "0.15"), 0.05
        )
        self._stats: dict[str, float | int] = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "prefetch_requests": 0,
            "prefetch_completed": 0,
            "prefetch_hits": 0,
            "route_requests": 0,
            "route_slots": 0,
            "route_swaps": 0,
            "route_agree_sum": 0.0,
            "route_kl_sum": 0.0,
            "bytes_read": 0,
            "bytes_prefetched": 0,
            "resident_bytes": 0,
            "pinned_bytes": 0,
            "promotions": 0,
            "evictions": 0,
            "last_update_ts": time.time(),
        }

        self._load_catalog()
        self._load_state()
        self._warm_start()

    def _dense_virtual_key(self, *, model: str, layer_id: int) -> str:
        return f"dense:{model}:layer:{int(layer_id)}"

    def _dense_virtual_meta(self, key: str) -> Optional[ExpertBlob]:
        match = re.fullmatch(r"dense:(.+):layer:(\d+)", str(key or ""))
        if match is None:
            return None
        model = str(match.group(1) or "").strip() or "generic"
        layer_id = int(match.group(2))
        hint = dict(self._dense_residency_hint or {})
        total_layers = max(int(hint.get("_total_layers", 0) or 0), 0)
        model_size_gb = max(float(hint.get("_model_size_gb", 0.0) or 0.0), 0.0)
        model_path = str(hint.get("_model_path") or "").strip()
        model_size_bytes = int(model_size_gb * (1024**3)) if model_size_gb > 0 else 0
        layer_bytes = int(model_size_bytes / max(total_layers, 1)) if total_layers > 0 and model_size_bytes > 0 else 0
        offset_bytes = min(max(layer_id, 0) * layer_bytes, max(model_size_bytes - layer_bytes, 0)) if layer_bytes > 0 else 0
        # 最后一层吃掉余数，避免 byte range 总和小于文件大小。
        if total_layers > 0 and layer_id >= total_layers - 1 and model_size_bytes > 0 and layer_bytes > 0:
            layer_bytes = max(model_size_bytes - offset_bytes, layer_bytes)
        return ExpertBlob(
            key=key,
            model=model,
            layer_id=layer_id,
            expert_id=-1,
            path=model_path,
            size_bytes=max(layer_bytes, 0),
            tags=("dense", "layer_streaming"),
            offset_bytes=max(offset_bytes, 0),
            io_backend="file_range" if model_path else "virtual",
        )

    def _resolve_meta(self, key: str) -> Optional[ExpertBlob]:
        meta = self._by_key.get(key)
        if meta is not None:
            return meta
        return self._dense_virtual_meta(key)

    def _build_dense_layer_plan(
        self,
        *,
        model: str,
        family: str,
        route_mode: str,
        prompt_hash: str,
        frontier_key: str,
    ) -> ExpertPlan:
        if not self.dense_layer_streaming_enabled:
            return ExpertPlan(False, "dense_layer_streaming_disabled", model, family, route_mode, prompt_hash=prompt_hash, frontier_key=frontier_key)
        hint = dict(self._dense_residency_hint or {})
        total_layers = max(int(hint.get("_total_layers", 0) or 0), 0)
        n_gpu_layers = int(hint.get("suggested_n_gpu_layers", -1) or -1)
        can_full_resident = bool(hint.get("can_full_resident", True))
        model_size_gb = max(float(hint.get("_model_size_gb", 0.0) or 0.0), 0.0)
        if total_layers <= 1 or can_full_resident or n_gpu_layers < 1 or n_gpu_layers >= total_layers:
            return ExpertPlan(False, "dense_full_resident", model, family, route_mode, prompt_hash=prompt_hash, frontier_key=frontier_key)

        streamed_layers = list(range(n_gpu_layers, total_layers))
        if not streamed_layers:
            return ExpertPlan(False, "dense_full_resident", model, family, route_mode, prompt_hash=prompt_hash, frontier_key=frontier_key)

        layer_cursor = int(self._frontier_cursor.get(frontier_key, 0))
        current_layers = self._rolling_layer_window(
            streamed_layers,
            start=layer_cursor,
            width=min(self.current_layer_window, len(streamed_layers)),
        )
        next_layer_cursor = (layer_cursor + len(current_layers)) % max(len(streamed_layers), 1)
        next_layers = self._rolling_layer_window(
            streamed_layers,
            start=next_layer_cursor,
            width=min(self.prefetch_layer_window, len(streamed_layers)),
        )
        next_next_cursor = (next_layer_cursor + len(next_layers)) % max(len(streamed_layers), 1)
        next_next_layers = self._rolling_layer_window(
            streamed_layers,
            start=next_next_cursor,
            width=min(self.prefetch_layer_window, len(streamed_layers)),
        )
        far_cursor = (next_next_cursor + len(next_next_layers)) % max(len(streamed_layers), 1)
        far_layers = self._rolling_layer_window(
            streamed_layers,
            start=far_cursor,
            width=min(self.prefetch_layer_window, len(streamed_layers)),
        )

        current_keys = [self._dense_virtual_key(model=model, layer_id=layer_id) for layer_id in current_layers]
        next_keys = [
            self._dense_virtual_key(model=model, layer_id=layer_id)
            for layer_id in next_layers
            if layer_id not in current_layers
        ]
        next_next_keys = [
            self._dense_virtual_key(model=model, layer_id=layer_id)
            for layer_id in next_next_layers
            if layer_id not in current_layers and layer_id not in next_layers
        ]
        far_keys = [
            self._dense_virtual_key(model=model, layer_id=layer_id)
            for layer_id in far_layers
            if layer_id not in current_layers and layer_id not in next_layers and layer_id not in next_next_layers
        ]
        layer_bytes = int((model_size_gb * (1024**3)) / max(total_layers, 1)) if model_size_gb > 0 else 0
        current_bytes = max(len(current_keys) * layer_bytes, 0)
        cold_bytes = sum(
            layer_bytes for key in current_keys
            if key not in self._resident
        )
        return ExpertPlan(
            enabled=True,
            reason="dense_layer_streaming_active",
            model=model,
            family=family,
            route_mode=route_mode,
            prompt_hash=prompt_hash,
            current_keys=current_keys,
            next_keys=next_keys,
            next_next_keys=next_next_keys,
            far_keys=far_keys,
            current_bytes=current_bytes,
            cold_bytes=cold_bytes,
            route_slots=len(current_keys),
            route_swaps=0,
            route_agree=1.0,
            route_kl=0.0,
            frontier_key=frontier_key,
            layer_cursor=layer_cursor,
            next_layer_cursor=next_layer_cursor,
            all_layer_ids=streamed_layers,
            current_layer_ids=current_layers,
            next_layer_ids=next_layers,
            next_next_layer_ids=next_next_layers,
        )

    def _load_catalog(self) -> None:
        if not self.enabled:
            return
        entries: list[ExpertBlob] = []
        source = "none"
        if self.manifest_path and os.path.exists(self.manifest_path):
            entries = self._load_from_manifest(Path(self.manifest_path))
            source = "manifest"
        elif self.blob_root and os.path.isdir(self.blob_root):
            entries = self._discover_from_root(Path(self.blob_root))
            source = "blob_root"
        elif self.autodiscover_catalog:
            manifest_path, blob_root = self._autodiscover_catalog()
            if manifest_path is not None:
                self.manifest_path = str(manifest_path)
                entries = self._load_from_manifest(manifest_path)
                source = "autodiscovered_manifest"
            elif blob_root is not None:
                self.blob_root = str(blob_root)
                entries = self._discover_from_root(blob_root)
                source = "autodiscovered_blob_root"
        self._by_model = {}
        self._by_key = {}
        for entry in entries:
            self._by_model.setdefault(entry.model, []).append(entry)
            self._by_key[entry.key] = entry
        for model_entries in self._by_model.values():
            model_entries.sort(key=lambda item: (item.layer_id, item.expert_id, item.key))
        self._catalog_source = source
        self._catalog_loaded_at = time.time()

    def _autodiscover_catalog(self) -> tuple[Optional[Path], Optional[Path]]:
        candidate_roots = [
            self.repo_root / "models",
            self.repo_root / "data" / "models",
            self.repo_root / "colibri",
        ]
        for root in candidate_roots:
            if not root.exists():
                continue
            for pattern in ("*expert*manifest*.json", "*expert*.manifest.json", "*manifest*.json"):
                for manifest_path in root.rglob(pattern):
                    try:
                        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if isinstance(payload.get("models"), dict):
                        return manifest_path, None
            for subdir in root.rglob("*"):
                if not subdir.is_dir():
                    continue
                rel = str(subdir).lower()
                if "expert" not in rel and "moe" not in rel:
                    continue
                has_blob = any(subdir.rglob("*.blob")) or any(subdir.rglob("*.bin")) or any(subdir.rglob("*.safetensors"))
                if has_blob:
                    return None, subdir
        return None, None

    def _load_from_manifest(self, path: Path) -> list[ExpertBlob]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        root = str(payload.get("blob_root") or path.parent)
        models = payload.get("models") or {}
        entries: list[ExpertBlob] = []
        for model_name, model_payload in models.items():
            for item in model_payload.get("experts", []):
                rel_path = str(item.get("path") or "")
                file_path = rel_path if os.path.isabs(rel_path) else os.path.join(root, rel_path)
                if not os.path.exists(file_path):
                    continue
                size_bytes = int(item.get("size_bytes") or os.path.getsize(file_path))
                layer_id = int(item.get("layer_id") or 0)
                expert_id = int(item.get("expert_id") or 0)
                key = str(item.get("key") or f"{_normalize_model_name(model_name)}:L{layer_id:02d}:E{expert_id:03d}")
                entries.append(
                    ExpertBlob(
                        key=key,
                        model=_normalize_model_name(model_name),
                        layer_id=layer_id,
                        expert_id=expert_id,
                        path=file_path,
                        size_bytes=size_bytes,
                        tags=tuple(item.get("tags") or ()),
                    )
                )
        return entries

    def _discover_from_root(self, root: Path) -> list[ExpertBlob]:
        entries: list[ExpertBlob] = []
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".bin", ".blob", ".pt", ".safetensors", ".gguf"}:
                continue
            rel = str(file_path.relative_to(root))
            layer_match = _LAYER_RE.search(rel)
            expert_match = _EXPERT_RE.search(rel)
            if expert_match is None:
                continue
            layer_id = int(layer_match.group(1)) if layer_match else 0
            expert_id = int(expert_match.group(1))
            model = _normalize_model_name(rel.split(os.sep, 1)[0])
            key = f"{model}:L{layer_id:02d}:E{expert_id:03d}"
            entries.append(
                ExpertBlob(
                    key=key,
                    model=model,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    path=str(file_path),
                    size_bytes=file_path.stat().st_size,
                    tags=(),
                )
            )
        return entries

    def _load_state(self) -> None:
        if not self.persist_enabled or not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            payload = json.loads(Path(self.state_path).read_text(encoding="utf-8"))
        except Exception:
            return
        family_affinity = payload.get("family_affinity") or {}
        for bucket_key, entries in family_affinity.items():
            if "|" not in bucket_key or not isinstance(entries, dict):
                continue
            model, family = bucket_key.split("|", 1)
            normalized = {
                key: float(value or 0.0)
                for key, value in entries.items()
                if key in self._by_key
            }
            if normalized:
                self._family_affinity[(model, family)] = normalized
        frontier_cursor = payload.get("frontier_cursor") or {}
        if isinstance(frontier_cursor, dict):
            self._frontier_cursor = {
                str(key): max(int(value or 0), 0)
                for key, value in frontier_cursor.items()
            }
        self._persisted_pinned_keys = [
            key for key in (payload.get("pinned_keys") or [])
            if key in self._by_key
        ]
        last_plan = payload.get("last_plan")
        if isinstance(last_plan, dict):
            self._last_plan = dict(last_plan)

    def _persist_state_locked(self, *, force: bool = False) -> None:
        if not self.persist_enabled or not self.state_path:
            return
        now = time.time()
        if not force and now - self._last_state_persist_ts < self.persist_interval_sec:
            return
        state_dir = Path(self.state_path).parent
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "v1",
            "saved_at": now,
            "catalog_source": self._catalog_source,
            "manifest_path": self.manifest_path,
            "blob_root": self.blob_root,
            "pinned_keys": sorted(key for key, resident in self._resident.items() if resident.pinned),
            "family_affinity": {
                f"{model}|{family}": {
                    key: round(float(score), 6)
                    for key, score in sorted(entries.items(), key=lambda item: (-float(item[1]), item[0]))[:128]
                    if key in self._by_key
                }
                for (model, family), entries in self._family_affinity.items()
            },
            "frontier_cursor": {key: int(value) for key, value in sorted(self._frontier_cursor.items())},
            "last_plan": dict(self._last_plan),
        }
        tmp_path = f"{self.state_path}.tmp"
        Path(tmp_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.state_path)
        self._last_state_persist_ts = now

    def _warm_start(self) -> None:
        if not self.preload_hot_count or not self._by_key:
            return
        ordered_keys: list[str] = []
        for key in self._persisted_pinned_keys:
            if key not in ordered_keys:
                ordered_keys.append(key)
        for entries in self._family_affinity.values():
            for key, _score in sorted(entries.items(), key=lambda item: (-float(item[1]), item[0])):
                if key not in ordered_keys and key in self._by_key:
                    ordered_keys.append(key)
                if len(ordered_keys) >= self.preload_hot_count:
                    break
            if len(ordered_keys) >= self.preload_hot_count:
                break
        for key in ordered_keys[: self.preload_hot_count]:
            meta = self._by_key.get(key)
            if meta is None or key in self._resident:
                continue
            try:
                payload, latency_ms = self._load_blob(meta)
            except Exception:
                continue
            resident = self._store_resident(meta, payload, source="warm_start", latency_ms=latency_ms, prefetched=True)
            resident.prefetched = True
            resident.access_count = max(resident.access_count, self.pin_promote_threshold)
            resident.hot_score = max(resident.hot_score, 1.0)
            if key in self._persisted_pinned_keys and int(self._stats["pinned_bytes"]) + meta.size_bytes <= self.pin_budget_bytes:
                resident.pinned = True
                self._stats["pinned_bytes"] = int(self._stats["pinned_bytes"]) + meta.size_bytes
            self._warm_start_loaded += 1

    def _resident_hot_score(self, key: str) -> float:
        resident = self._resident.get(key)
        return float(resident.hot_score) if resident is not None else 0.0

    def _frontier_key(
        self,
        *,
        model: str,
        family: str,
        prompt_hash: str,
        route_mode: str,
        frontier_id: str = "",
    ) -> str:
        frontier_scope = str(frontier_id or prompt_hash or family or "generic")
        return f"{model}|{family}|{frontier_scope}|{route_mode or 'unknown'}"

    def _rolling_layer_window(self, layers: list[int], *, start: int, width: int) -> list[int]:
        if not layers or width <= 0:
            return []
        count = min(width, len(layers))
        return [layers[(start + offset) % len(layers)] for offset in range(count)]

    def _pseudo_mass(self, index: int) -> float:
        return float(math.exp(-index / max(self.route_alpha, 1e-6)))

    def _route_metrics(self, true_top: list[ExpertBlob], chosen: list[ExpertBlob]) -> tuple[int, int, float, float]:
        if not true_top:
            return 0, 0, 1.0, 0.0
        true_keys = [entry.key for entry in true_top]
        chosen_keys = [entry.key for entry in chosen]
        true_set = set(true_keys)
        overlap = sum(1 for key in chosen_keys if key in true_set)
        swaps = sum(1 for key in chosen_keys if key not in true_set)
        agree = overlap / max(len(true_keys), 1)
        true_mass_raw = [self._pseudo_mass(idx) for idx in range(len(true_keys))]
        true_den = sum(true_mass_raw) or 1.0
        chosen_mass_raw: list[float] = []
        for idx, key in enumerate(chosen_keys):
            if key in true_keys:
                chosen_mass_raw.append(self._pseudo_mass(true_keys.index(key)))
            else:
                chosen_mass_raw.append(self._pseudo_mass(idx) * self.route_alpha)
        chosen_den = sum(chosen_mass_raw) or 1.0
        kl = 0.0
        for idx in range(min(len(true_mass_raw), len(chosen_mass_raw))):
            p = max(true_mass_raw[idx] / true_den, 1e-9)
            q = max(chosen_mass_raw[idx] / chosen_den, 1e-9)
            kl += p * math.log(p / q)
        return len(chosen_keys), swaps, agree, kl

    def _select_from_layer(
        self,
        layer_entries: list[ExpertBlob],
        *,
        model: str,
        family: str,
        route_mode: str,
        prompt_hash: str,
    ) -> tuple[list[ExpertBlob], dict[str, float | int]]:
        affinity = self._family_affinity.get((model, family), {})
        seed = f"{model}:{family}:{route_mode}:{prompt_hash}:{layer_entries[0].layer_id}"
        ranked = sorted(
            layer_entries,
            key=lambda entry: (
                -float(affinity.get(entry.key, 0.0)),
                _stable_hash(f"{seed}:{entry.key}"),
            ),
        )
        true_top = list(ranked[: min(self.top_k, len(ranked))])
        if not self.cache_route_enabled:
            route_slots, route_swaps, route_agree, route_kl = self._route_metrics(true_top, true_top)
            return true_top, {
                "route_slots": route_slots,
                "route_swaps": route_swaps,
                "route_agree": route_agree,
                "route_kl": route_kl,
            }
        chosen: list[ExpertBlob] = list(ranked[: min(self.top_j, len(ranked))])
        chosen_keys = {entry.key for entry in chosen}
        window = ranked[: min(self.top_m, len(ranked))]
        resident_preferred = [
            entry for entry in window
            if entry.key not in chosen_keys and entry.key in self._resident
        ]
        resident_preferred.sort(
            key=lambda entry: (
                -self._resident_hot_score(entry.key),
                -float(self._family_affinity.get((model, family), {}).get(entry.key, 0.0)),
                _stable_hash(f"resident:{seed}:{entry.key}"),
            )
        )
        for entry in resident_preferred:
            if len(chosen) >= self.top_k:
                break
            chosen.append(entry)
            chosen_keys.add(entry.key)
        for entry in window:
            if len(chosen) >= self.top_k:
                break
            if entry.key in chosen_keys:
                continue
            chosen.append(entry)
            chosen_keys.add(entry.key)
        route_slots, route_swaps, route_agree, route_kl = self._route_metrics(true_top, chosen)
        return chosen, {
            "route_slots": route_slots,
            "route_swaps": route_swaps,
            "route_agree": route_agree,
            "route_kl": route_kl,
        }

    def _build_plan(
        self,
        *,
        model_name: str,
        family: str,
        route_mode: str,
        prompt_hash: str,
        frontier_id: str = "",
        _frontier_key_override: str = "",
    ) -> ExpertPlan:
        model = _normalize_model_name(model_name)
        # P2+ fix: advance_window 直接传入原 frontier_key, 避免二次包装
        if _frontier_key_override:
            frontier_key = _frontier_key_override
        else:
            frontier_key = self._frontier_key(
                model=model,
                family=family,
                prompt_hash=prompt_hash,
                route_mode=route_mode,
                frontier_id=frontier_id,
            )
        if route_mode not in {_ROUTE_LOCAL_FULL, _ROUTE_LAYER_SPLIT_PD}:
            return ExpertPlan(False, f"route_mode:{route_mode or 'unknown'}", model, family, route_mode, prompt_hash=prompt_hash, frontier_key=frontier_key)
        if not self.enabled and not self.dense_layer_streaming_enabled:
            return ExpertPlan(False, "expert_streaming_disabled", model, family, route_mode, prompt_hash=prompt_hash, frontier_key=frontier_key)
        entries = self._by_model.get(model) or []
        if not entries:
            can_full_resident = bool((self._dense_residency_hint or {}).get("can_full_resident", True))
            if can_full_resident:
                return ExpertPlan(False, "fit_memory_bypass", model, family, route_mode, prompt_hash=prompt_hash, frontier_key=frontier_key)
            dense_plan = self._build_dense_layer_plan(
                model=model,
                family=family,
                route_mode=route_mode,
                prompt_hash=prompt_hash,
                frontier_key=frontier_key,
            )
            if dense_plan.enabled:
                return dense_plan
            return ExpertPlan(False, "no_expert_catalog", model, family, route_mode, prompt_hash=prompt_hash, frontier_key=frontier_key)
        per_layer: dict[int, list[ExpertBlob]] = {}
        for entry in entries:
            per_layer.setdefault(entry.layer_id, []).append(entry)
        layers = sorted(per_layer)
        layer_cursor = int(self._frontier_cursor.get(frontier_key, 0))
        current_layers = self._rolling_layer_window(
            layers,
            start=layer_cursor,
            width=self.current_layer_window,
        )
        next_layer_cursor = (layer_cursor + len(current_layers)) % max(len(layers), 1)
        next_layers = self._rolling_layer_window(
            layers,
            start=next_layer_cursor,
            width=self.prefetch_layer_window,
        )
        # P2 double-buffer: 下下个 window, decode current 时后台 prefetch
        next_next_cursor = (next_layer_cursor + len(next_layers)) % max(len(layers), 1)
        next_next_layers = self._rolling_layer_window(
            layers,
            start=next_next_cursor,
            width=self.prefetch_layer_window,
        )
        # P2+ pipeline: 3 windows ahead, daemon 持续 prefetch
        far_cursor = (next_next_cursor + len(next_next_layers)) % max(len(layers), 1)
        far_layers = self._rolling_layer_window(
            layers,
            start=far_cursor,
            width=self.prefetch_layer_window,
        )
        current_entries: list[ExpertBlob] = []
        next_entries: list[ExpertBlob] = []
        next_next_entries: list[ExpertBlob] = []
        far_entries: list[ExpertBlob] = []
        route_slots = 0
        route_swaps = 0
        route_agree_sum = 0.0
        route_kl_sum = 0.0
        for layer_id in current_layers:
            selected_entries, metrics = self._select_from_layer(
                per_layer[layer_id],
                model=model,
                family=family,
                route_mode=route_mode,
                prompt_hash=prompt_hash,
            )
            current_entries.extend(selected_entries)
            route_slots += int(metrics.get("route_slots", 0) or 0)
            route_swaps += int(metrics.get("route_swaps", 0) or 0)
            route_agree_sum += float(metrics.get("route_agree", 1.0) or 0.0)
            route_kl_sum += float(metrics.get("route_kl", 0.0) or 0.0)
        for layer_id in next_layers:
            selected_entries, _ = self._select_from_layer(
                per_layer[layer_id],
                model=model,
                family=family,
                route_mode=route_mode,
                prompt_hash=f"{prompt_hash}:next",
            )
            next_entries.extend(selected_entries)
        # P2: 选择下下个 window 的 experts (轻量, 不累计 metrics)
        for layer_id in next_next_layers:
            selected_entries, _ = self._select_from_layer(
                per_layer[layer_id],
                model=model,
                family=family,
                route_mode=route_mode,
                prompt_hash=f"{prompt_hash}:next_next",
            )
            next_next_entries.extend(selected_entries)
        # P2+ pipeline: 选择 3 windows ahead 的 experts (daemon prefetch)
        for layer_id in far_layers:
            selected_entries, _ = self._select_from_layer(
                per_layer[layer_id],
                model=model,
                family=family,
                route_mode=route_mode,
                prompt_hash=f"{prompt_hash}:far",
            )
            far_entries.extend(selected_entries)
        current_keys = [entry.key for entry in current_entries]
        next_keys = [entry.key for entry in next_entries if entry.key not in current_keys]
        next_next_keys = [
            entry.key for entry in next_next_entries
            if entry.key not in current_keys and entry.key not in next_keys
        ]
        far_keys = [
            entry.key for entry in far_entries
            if entry.key not in current_keys and entry.key not in next_keys and entry.key not in next_next_keys
        ]
        current_bytes = sum(entry.size_bytes for entry in current_entries)
        cold_bytes = sum(entry.size_bytes for entry in current_entries if entry.key not in self._resident)
        return ExpertPlan(
            enabled=True,
            reason="ok",
            model=model,
            family=family,
            route_mode=route_mode,
            prompt_hash=prompt_hash,
            current_keys=current_keys,
            next_keys=next_keys,
            next_next_keys=next_next_keys,
            far_keys=far_keys,
            current_bytes=current_bytes,
            cold_bytes=cold_bytes,
            route_slots=route_slots,
            route_swaps=route_swaps,
            route_agree=(route_agree_sum / len(current_layers)) if current_layers else 1.0,
            route_kl=(route_kl_sum / len(current_layers)) if current_layers else 0.0,
            frontier_key=frontier_key,
            layer_cursor=layer_cursor,
            next_layer_cursor=next_layer_cursor,
            all_layer_ids=layers,
            current_layer_ids=current_layers,
            next_layer_ids=next_layers,
            next_next_layer_ids=next_next_layers,
        )

    def preview_request(
        self,
        *,
        model_name: str,
        family_info: dict[str, Any],
        draft_policy: Optional[dict[str, Any]],
        route_mode: str,
    ) -> dict[str, Any]:
        del draft_policy
        with self._lock:
            plan = self._build_plan(
                model_name=model_name,
                family=str(family_info.get("family") or "generic"),
                route_mode=str(route_mode or ""),
                prompt_hash=str(family_info.get("prompt_hash") or ""),
                frontier_id=str(family_info.get("frontier_id") or ""),
            )
            payload = plan.to_dict()
            payload["runtime_unit_plan"] = self.export_runtime_unit_plan(plan=plan)
            payload["runtime"] = self.runtime_snapshot()
            return payload

    def _unit_kind_for_key(self, key: str) -> str:
        if str(key or "").startswith("dense:"):
            return "layer"
        return "expert"

    def _cached_runtime_unit_descriptor(self, key: str) -> Optional[dict[str, Any]]:
        runtime_plan = dict((self._last_plan or {}).get("runtime_unit_plan") or {})
        for lane in ("current", "next", "next_next", "far"):
            for unit in list(runtime_plan.get(lane) or []):
                if str((unit or {}).get("key") or "") == str(key or ""):
                    return dict(unit)
        return None

    def _target_tier_for_unit(self, key: str, *, lane: str) -> str:
        resident = self._resident.get(key)
        if resident is not None and resident.pinned:
            return "pinned_ram"
        if lane == "current":
            return "resident_ram"
        if lane in {"next", "next_next", "far"}:
            return "resident_ram"
        if resident is not None:
            return "resident_ram"
        return "nvme"

    def _pin_priority_for_unit(self, key: str, *, lane: str) -> float:
        resident = self._resident.get(key)
        if resident is not None and resident.pinned:
            return 1.0
        hot_score = float(resident.hot_score) if resident is not None else 0.0
        base = min(hot_score / max(float(self.pin_promote_threshold), 1.0), 1.0)
        lane_bonus = {
            "current": 0.35,
            "next": 0.2,
            "next_next": 0.1,
            "far": 0.05,
        }.get(str(lane or ""), 0.0)
        return round(min(base + lane_bonus, 1.0), 4)

    def _unit_descriptor(self, key: str, *, lane: str = "current") -> dict[str, Any]:
        meta = self._resolve_meta(key)
        resident = self._resident.get(key)
        cached = self._cached_runtime_unit_descriptor(key)
        if meta is None:
            return {
                "key": key,
                "unit_kind": self._unit_kind_for_key(key),
                "model": str((cached or {}).get("model") or ""),
                "layer_id": int((cached or {}).get("layer_id") or 0),
                "expert_id": int((cached or {}).get("expert_id") or 0),
                "path": str((cached or {}).get("path") or ""),
                "size_bytes": int((cached or {}).get("size_bytes") or 0),
                "offset_bytes": int((cached or {}).get("offset_bytes") or 0),
                "io_backend": str((cached or {}).get("io_backend") or "virtual"),
                "tags": list((cached or {}).get("tags") or []),
                "target_tier": self._target_tier_for_unit(key, lane=lane),
                "routing_heat": 0.0,
                "pin_priority": self._pin_priority_for_unit(key, lane=lane),
                "resident": False,
                "available": bool((cached or {}).get("available", False)),
            }
        path = str(meta.path or "")
        size_bytes = int(meta.size_bytes or 0)
        offset_bytes = int(meta.offset_bytes or 0)
        io_backend = str(meta.io_backend or "file_full")
        if (not path or size_bytes <= 0 or io_backend == "virtual") and cached:
            path = path or str(cached.get("path") or "")
            size_bytes = size_bytes if size_bytes > 0 else int(cached.get("size_bytes") or 0)
            offset_bytes = offset_bytes if offset_bytes > 0 else int(cached.get("offset_bytes") or 0)
            if io_backend == "virtual":
                io_backend = str(cached.get("io_backend") or io_backend)
        return {
            "key": key,
            "unit_kind": self._unit_kind_for_key(key),
            "model": meta.model,
            "layer_id": int(meta.layer_id),
            "expert_id": int(meta.expert_id),
            "path": path,
            "size_bytes": size_bytes,
            "offset_bytes": offset_bytes,
            "io_backend": io_backend,
            "tags": list(meta.tags),
            "available": True,
            "resident": resident is not None,
            "pinned": bool(resident.pinned) if resident is not None else False,
            "prefetched": bool(resident.prefetched) if resident is not None else False,
            "target_tier": self._target_tier_for_unit(key, lane=lane),
            "routing_heat": round(float(resident.hot_score) if resident is not None else 0.0, 4),
            "pin_priority": self._pin_priority_for_unit(key, lane=lane),
        }

    def export_runtime_unit_plan(self, *, plan: Optional[ExpertPlan] = None) -> dict[str, Any]:
        with self._lock:
            active_plan = plan
            if active_plan is None:
                last_plan = dict(self._last_plan or {})
                active_plan = ExpertPlan(
                    enabled=bool(last_plan.get("enabled")),
                    reason=str(last_plan.get("reason") or ""),
                    model=str(last_plan.get("model") or "generic"),
                    family=str(last_plan.get("family") or "generic"),
                    route_mode=str(last_plan.get("route_mode") or ""),
                    prompt_hash=str(last_plan.get("prompt_hash") or ""),
                    current_keys=list(last_plan.get("current_keys") or []),
                    next_keys=list(last_plan.get("next_keys") or []),
                    next_next_keys=list(last_plan.get("next_next_keys") or []),
                    far_keys=list(last_plan.get("far_keys") or []),
                    current_bytes=int(last_plan.get("predicted_bytes_to_read_mb") or 0) * 1024 * 1024,
                    cold_bytes=int(last_plan.get("predicted_cold_bytes_mb") or 0) * 1024 * 1024,
                    route_slots=int(last_plan.get("route_slots") or 0),
                    route_swaps=int(last_plan.get("route_swaps") or 0),
                    route_agree=float(last_plan.get("route_agree") or 1.0),
                    route_kl=float(last_plan.get("route_kl") or 0.0),
                    frontier_key=str(last_plan.get("frontier_key") or ""),
                    layer_cursor=int(last_plan.get("layer_cursor") or 0),
                    next_layer_cursor=int(last_plan.get("next_layer_cursor") or 0),
                    current_layer_ids=list(last_plan.get("current_layer_ids") or []),
                    next_layer_ids=list(last_plan.get("next_layer_ids") or []),
                    next_next_layer_ids=list(last_plan.get("next_next_layer_ids") or []),
                )
            mode = "bypass"
            if active_plan.enabled:
                mode = "dense_layer_streaming" if "dense" in str(active_plan.reason or "") else "expert_streaming"
            current_units = [self._unit_descriptor(key, lane="current") for key in active_plan.current_keys]
            next_units = [self._unit_descriptor(key, lane="next") for key in active_plan.next_keys]
            next_next_units = [self._unit_descriptor(key, lane="next_next") for key in active_plan.next_next_keys]
            far_units = [self._unit_descriptor(key, lane="far") for key in active_plan.far_keys]

            tier_counts: dict[str, int] = {}
            current_tier_counts: dict[str, int] = {}
            for unit in current_units + next_units + next_next_units + far_units:
                tier = str((unit or {}).get("target_tier") or "unknown")
                tier_counts[tier] = int(tier_counts.get(tier, 0)) + 1
            for unit in current_units:
                tier = str((unit or {}).get("target_tier") or "unknown")
                current_tier_counts[tier] = int(current_tier_counts.get(tier, 0)) + 1
            return {
                "control_plane": "expert_data_plane",
                "enabled": bool(active_plan.enabled),
                "mode": mode,
                "reason": str(active_plan.reason or ""),
                "model": str(active_plan.model or ""),
                "family": str(active_plan.family or ""),
                "route_mode": str(active_plan.route_mode or ""),
                "frontier_key": str(active_plan.frontier_key or ""),
                "current": current_units,
                "next": next_units,
                "next_next": next_next_units,
                "far": far_units,
                "summary": {
                    "placement_metadata_version": 1,
                    "current_unit_count": len(active_plan.current_keys),
                    "next_unit_count": len(active_plan.next_keys),
                    "next_next_unit_count": len(active_plan.next_next_keys),
                    "far_unit_count": len(active_plan.far_keys),
                    "predicted_bytes_to_read_mb": round(float(active_plan.current_bytes) / 1024**2, 3),
                    "predicted_cold_bytes_mb": round(float(active_plan.cold_bytes) / 1024**2, 3),
                    "tier_counts": tier_counts,
                    "current_tier_counts": current_tier_counts,
                    "resident_budget_bytes": int(self.ram_budget_bytes),
                    "resident_bytes": int(self._stats["resident_bytes"]),
                    "pin_budget_bytes": int(self.pin_budget_bytes),
                    "pinned_bytes": int(self._stats["pinned_bytes"]),
                },
            }

    def _evict_to_budget(self, incoming_bytes: int) -> None:
        while self._resident and int(self._stats["resident_bytes"]) + incoming_bytes > self.ram_budget_bytes:
            evict_candidates = [
                resident for resident in self._resident.values()
                if not resident.pinned
            ]
            if not evict_candidates:
                return
            victim = min(evict_candidates, key=lambda resident: (resident.last_access, resident.loaded_at))
            self._stats["resident_bytes"] = max(int(self._stats["resident_bytes"]) - victim.meta.size_bytes, 0)
            self._resident.pop(victim.meta.key, None)
            self._stats["evictions"] += 1

    def _promote_if_hot(self, resident: ResidentExpert) -> None:
        if resident.pinned:
            return
        if resident.access_count < self.pin_promote_threshold:
            return
        current_pinned = int(self._stats["pinned_bytes"])
        if current_pinned + resident.meta.size_bytes > self.pin_budget_bytes:
            return
        resident.pinned = True
        self._stats["pinned_bytes"] = current_pinned + resident.meta.size_bytes
        self._stats["promotions"] += 1

    def _load_blob(self, meta: ExpertBlob) -> tuple[bytes, float]:
        if not str(meta.path or "").strip():
            return b"", 0.0
        started = time.monotonic()
        with open(meta.path, "rb") as handle:
            if str(meta.io_backend or "") == "file_range" and int(meta.size_bytes or 0) > 0:
                handle.seek(max(int(meta.offset_bytes or 0), 0))
                payload = handle.read(max(int(meta.size_bytes or 0), 0))
            else:
                payload = handle.read()
        latency_ms = (time.monotonic() - started) * 1000
        return payload, latency_ms

    def _store_resident(
        self,
        meta: ExpertBlob,
        payload: bytes,
        *,
        source: str,
        latency_ms: float,
        prefetched: bool,
    ) -> ResidentExpert:
        now = time.time()
        resident = self._resident.get(meta.key)
        if resident is not None:
            resident.last_access = now
            resident.prefetched = resident.prefetched or prefetched
            resident.source = source
            return resident
        self._evict_to_budget(meta.size_bytes)
        resident = ResidentExpert(
            meta=meta,
            payload=payload,
            loaded_at=now,
            last_access=now,
            prefetched=prefetched,
            source=source,
            load_latency_ms=latency_ms,
        )
        self._resident[meta.key] = resident
        self._stats["resident_bytes"] = int(self._stats["resident_bytes"]) + meta.size_bytes
        return resident

    def _touch_resident(self, key: str, *, prefetched_hit_ok: bool) -> tuple[bool, bool]:
        resident = self._resident.get(key)
        if resident is None:
            return False, False
        resident.last_access = time.time()
        resident.access_count += 1
        resident.hot_score = resident.hot_score * 0.85 + 1.0
        self._promote_if_hot(resident)
        prefetch_hit = False
        if prefetched_hit_ok and resident.prefetched and not resident.prefetch_hits_counted:
            resident.prefetch_hits_counted = True
            prefetch_hit = True
            self._stats["prefetch_hits"] += 1
        return True, prefetch_hit

    def _ensure_loaded(self, key: str, *, source: str) -> tuple[bool, bool]:
        hit, prefetch_hit = self._touch_resident(key, prefetched_hit_ok=(source == "request"))
        if hit:
            self._stats["cache_hits"] += 1
            return True, prefetch_hit
        meta = self._resolve_meta(key)
        if meta is None:
            return False, False
        payload, latency_ms = self._load_blob(meta)
        self._stats["cache_misses"] += 1
        self._stats["bytes_read"] = int(self._stats["bytes_read"]) + meta.size_bytes
        resident = self._store_resident(meta, payload, source=source, latency_ms=latency_ms, prefetched=False)
        resident.access_count += 1
        resident.hot_score = resident.hot_score * 0.85 + 1.0
        self._promote_if_hot(resident)
        return False, False

    def _prefetch_one(self, key: str) -> None:
        meta = self._resolve_meta(key)
        if meta is None:
            with self._lock:
                self._inflight_prefetch.pop(key, None)
            return
        try:
            payload, latency_ms = self._load_blob(meta)
            with self._lock:
                resident = self._store_resident(meta, payload, source="prefetch", latency_ms=latency_ms, prefetched=True)
                resident.prefetched = True
                self._stats["prefetch_completed"] += 1
                self._stats["bytes_prefetched"] = int(self._stats["bytes_prefetched"]) + meta.size_bytes
        finally:
            with self._lock:
                self._inflight_prefetch.pop(key, None)

    def _schedule_prefetch(self, keys: list[str]) -> list[str]:
        scheduled: list[str] = []
        for key in keys:
            if key in self._resident or key in self._inflight_prefetch:
                continue
            if self._resolve_meta(key) is None:
                continue
            self._stats["prefetch_requests"] += 1
            future = self._executor.submit(self._prefetch_one, key)
            self._inflight_prefetch[key] = future
            scheduled.append(key)
        return scheduled

    def begin_request(
        self,
        *,
        model_name: str,
        family_info: dict[str, Any],
        draft_policy: Optional[dict[str, Any]],
        route_mode: str,
    ) -> ExpertRequestSession:
        del draft_policy
        with self._lock:
            family = str(family_info.get("family") or "generic")
            plan = self._build_plan(
                model_name=model_name,
                family=family,
                route_mode=str(route_mode or ""),
                prompt_hash=str(family_info.get("prompt_hash") or ""),
                frontier_id=str(family_info.get("frontier_id") or ""),
            )
            session = ExpertRequestSession(
                request_id=f"exp-{int(time.time() * 1000)}-{_stable_hash(str(family_info.get('request_uid') or family_info.get('frontier_id') or family_info.get('prompt_hash') or '')) % 100000}",
                plan=plan,
                started_at=time.time(),
                frontier_key=self._frontier_key(
                    model=plan.model,
                    family=plan.family,
                    prompt_hash=str(family_info.get("prompt_hash") or ""),
                    route_mode=plan.route_mode,
                    frontier_id=str(family_info.get("frontier_id") or ""),
                ),
                cursor_start=plan.layer_cursor,
                cursor_next=plan.next_layer_cursor,
            )
            self._stats["requests"] += 1
            if not plan.enabled:
                self._last_plan = {
                    **plan.to_dict(),
                    "runtime_unit_plan": self.export_runtime_unit_plan(plan=plan),
                }
                return session
            self._stats["route_requests"] = int(self._stats["route_requests"]) + 1
            self._stats["route_slots"] = int(self._stats["route_slots"]) + int(plan.route_slots)
            self._stats["route_swaps"] = int(self._stats["route_swaps"]) + int(plan.route_swaps)
            self._stats["route_agree_sum"] = float(self._stats["route_agree_sum"]) + float(plan.route_agree)
            self._stats["route_kl_sum"] = float(self._stats["route_kl_sum"]) + float(plan.route_kl)

            # P2 double-buffer: 先提交 prefetch (不阻塞, future 立即返回)
            # 顺序: next (近, 即将用到) 优先, 再 next_next (远, double-buffer)
            # 这样 current 同步加载时, next + next_next 在后台并行 prefetch
            scheduled_next = self._schedule_prefetch(plan.next_keys)
            scheduled_ahead = self._schedule_prefetch(plan.next_next_keys)
            session.prefetched_keys = scheduled_next
            session.prefetched_ahead_keys = scheduled_ahead

        # 锁外: 同步加载 current (此时 prefetch 线程在后台并行读盘)
        # _ensure_loaded 内部各自持锁, 不会阻塞 prefetch 线程的 _store_resident
        for key in plan.current_keys:
            hit, prefetch_hit = self._ensure_loaded(key, source="request")
            if hit:
                session.cache_hits += 1
            if prefetch_hit:
                session.prefetch_hits += 1
            session.loaded_keys.append(key)

        with self._lock:
            self._last_plan = {
                **plan.to_dict(),
                "runtime_unit_plan": self.export_runtime_unit_plan(plan=plan),
                "request_id": session.request_id,
                "frontier_key": session.frontier_key,
                "scheduled_prefetch": list(session.prefetched_keys),
                "scheduled_prefetch_ahead": list(session.prefetched_ahead_keys),
                "cache_hits": session.cache_hits,
                "prefetch_hits": session.prefetch_hits,
                "ts": time.time(),
            }
            self._stats["last_update_ts"] = time.time()
            # P2+ pipeline: 注册 active session, 启动 daemon
            self._active_sessions[session.request_id] = session
        self._ensure_pipeline_thread()
        # P3+ dense residency monitor: 更新 EMA
        self._update_dense_residency_monitor()
        return session

    def complete_request(
        self,
        session: Optional[ExpertRequestSession],
        *,
        success: bool,
        response_text: str,
    ) -> None:
        if session is None:
            return
        with self._lock:
            # P2+ pipeline: 移除 active session
            self._active_sessions.pop(session.request_id, None)
            plan = session.plan
            if not plan.enabled:
                return
            affinity = self._family_affinity.setdefault((plan.model, plan.family), {})
            reward = 1.0 if success else 0.25
            if response_text.strip():
                reward += min(len(response_text) / 4096.0, 1.0)
            for key in plan.current_keys:
                affinity[key] = affinity.get(key, 0.0) * 0.9 + reward
            if success and session.frontier_key:
                self._frontier_cursor[session.frontier_key] = int(session.cursor_next)
            self._last_plan = {
                **self._last_plan,
                "completed": True,
                "success": success,
                "execution_success": success,
                "content_success": bool(str(response_text or "").strip()),
                "response_chars": len(response_text),
                "completed_ts": time.time(),
                "frontier_advanced_to": int(self._frontier_cursor.get(session.frontier_key, session.cursor_start)),
            }
            self._stats["last_update_ts"] = time.time()
            self._persist_state_locked(force=True)
        # P3+ dense residency monitor: 更新 EMA (request 完成后 expert 占用可能变化)
        self._update_dense_residency_monitor()

    def reload_catalog(self) -> dict[str, Any]:
        with self._lock:
            self._resident.clear()
            self._inflight_prefetch.clear()
            self._stats["resident_bytes"] = 0
            self._stats["pinned_bytes"] = 0
            self._warm_start_loaded = 0
            self._load_catalog()
            self._load_state()
            self._warm_start()
            self._persist_state_locked(force=True)
            return self.runtime_snapshot()

    def reset_runtime(self, *, drop_resident: bool, drop_affinity: bool) -> dict[str, Any]:
        with self._lock:
            if drop_resident:
                self._resident.clear()
                self._inflight_prefetch.clear()
                self._stats["resident_bytes"] = 0
                self._stats["pinned_bytes"] = 0
            if drop_affinity:
                self._family_affinity.clear()
                self._frontier_cursor.clear()
                self._persisted_pinned_keys = []
            for key in (
                "requests",
                "cache_hits",
                "cache_misses",
                "prefetch_requests",
                "prefetch_completed",
                "prefetch_hits",
                "route_requests",
                "route_slots",
                "route_swaps",
                "route_agree_sum",
                "route_kl_sum",
                "bytes_read",
                "bytes_prefetched",
                "promotions",
                "evictions",
            ):
                self._stats[key] = 0
            self._last_plan = {}
            self._persist_state_locked(force=True)
            return self.runtime_snapshot()

    def set_pin_state(self, keys: list[str], *, pinned: bool) -> dict[str, Any]:
        with self._lock:
            changed = 0
            for key in keys:
                resident = self._resident.get(key)
                if resident is None:
                    continue
                if pinned and not resident.pinned:
                    if int(self._stats["pinned_bytes"]) + resident.meta.size_bytes > self.pin_budget_bytes:
                        continue
                    resident.pinned = True
                    self._stats["pinned_bytes"] = int(self._stats["pinned_bytes"]) + resident.meta.size_bytes
                    changed += 1
                elif not pinned and resident.pinned:
                    resident.pinned = False
                    self._stats["pinned_bytes"] = max(int(self._stats["pinned_bytes"]) - resident.meta.size_bytes, 0)
                    changed += 1
            self._persist_state_locked(force=True)
            return {
                "changed": changed,
                "requested": len(keys),
                "pinned": pinned,
                "runtime": self.runtime_snapshot(),
            }

    def runtime_snapshot(self) -> dict[str, Any]:
        with self._lock:
            expert_count = len(self._by_key)
            resident_count = len(self._resident)
            pinned_count = sum(1 for resident in self._resident.values() if resident.pinned)
            prefetch_inflight = len(self._inflight_prefetch)
            requests = int(self._stats["requests"])
            hit_den = int(self._stats["cache_hits"]) + int(self._stats["cache_misses"])
            prefetch_den = int(self._stats["prefetch_requests"])
            route_requests = int(self._stats["route_requests"])
            top_hot = sorted(
                self._resident.values(),
                key=lambda resident: (-resident.hot_score, resident.meta.layer_id, resident.meta.expert_id),
            )[:8]
            last_plan_payload = dict(self._last_plan)
            if last_plan_payload:
                last_plan_payload["runtime_unit_plan"] = self.export_runtime_unit_plan()
            return {
                "enabled": self.enabled,
                "dense_layer_streaming_enabled": self.dense_layer_streaming_enabled,
                "cache_route_enabled": self.cache_route_enabled,
                "manifest_path": self.manifest_path or "(auto)",
                "blob_root": self.blob_root or "(manifest)",
                "catalog_source": self._catalog_source,
                "catalog_loaded_at": self._catalog_loaded_at,
                "catalog_models": sorted(self._by_model.keys()),
                "expert_count": expert_count,
                "resident_count": resident_count,
                "resident_bytes": int(self._stats["resident_bytes"]),
                "resident_gb": round(int(self._stats["resident_bytes"]) / 1024**3, 3),
                "pinned_count": pinned_count,
                "pinned_bytes": int(self._stats["pinned_bytes"]),
                "pinned_gb": round(int(self._stats["pinned_bytes"]) / 1024**3, 3),
                "ram_budget_gb": round(self.ram_budget_bytes / 1024**3, 3),
                "pin_budget_gb": round(self.pin_budget_bytes / 1024**3, 3),
                "requests": requests,
                "cache_hit_rate": round(int(self._stats["cache_hits"]) / hit_den, 3) if hit_den else 0.0,
                "prefetch_hit_rate": round(int(self._stats["prefetch_hits"]) / prefetch_den, 3) if prefetch_den else 0.0,
                "route_swaps": int(self._stats["route_swaps"]),
                "route_slots": int(self._stats["route_slots"]),
                "route_swap_pct": round(int(self._stats["route_swaps"]) / max(int(self._stats["route_slots"]), 1), 3)
                if int(self._stats["route_slots"])
                else 0.0,
                "route_agree": round(float(self._stats["route_agree_sum"]) / max(route_requests, 1), 3)
                if route_requests
                else 1.0,
                "route_kl": round(float(self._stats["route_kl_sum"]) / max(route_requests, 1), 6)
                if route_requests
                else 0.0,
                "bytes_read_mb": round(int(self._stats["bytes_read"]) / 1024**2, 3),
                "bytes_prefetched_mb": round(int(self._stats["bytes_prefetched"]) / 1024**2, 3),
                "prefetch_inflight": prefetch_inflight,
                "promotions": int(self._stats["promotions"]),
                "evictions": int(self._stats["evictions"]),
                "state_path": self.state_path or "",
                "warm_start_loaded": self._warm_start_loaded,
                "frontier_cursor_count": len(self._frontier_cursor),
                "frontier_cursor_head": [
                    {"key": key, "cursor": int(cursor)}
                    for key, cursor in list(sorted(self._frontier_cursor.items()))[:8]
                ],
                "dense_io_backend": "file_range" if str(self._dense_residency_hint.get("_model_path") or "").strip() else "virtual",
                "dense_residency_hint": dict(self._dense_residency_hint),
                "last_plan": last_plan_payload,
                "top_hot_experts": [
                    {
                        "key": resident.meta.key,
                        "layer_id": resident.meta.layer_id,
                        "expert_id": resident.meta.expert_id,
                        "io_backend": resident.meta.io_backend,
                        "hot_score": round(resident.hot_score, 3),
                        "pinned": resident.pinned,
                        "prefetched": resident.prefetched,
                        "size_mb": round(resident.meta.size_bytes / 1024**2, 3),
                    }
                    for resident in top_hot
                ],
            }

    # ===== P2+ pipeline: 持续性 I/O-compute overlap =====

    def _ensure_pipeline_thread(self) -> None:
        """Lazy 启动 pipeline daemon (第一次 begin_request 时启动)."""
        if self._pipeline_thread is not None:
            return
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop, daemon=True, name="expert-pipeline-daemon"
        )
        self._pipeline_thread.start()

    def _pipeline_loop(self) -> None:
        """Pipeline daemon 主循环: 持续 prefetch active session 的 far window."""
        while not self._pipeline_stop.is_set():
            try:
                self._pipeline_step()
            except Exception:
                pass
            self._pipeline_stop.wait(self._pipeline_interval)

    def _pipeline_step(self) -> None:
        """扫描 active sessions, 持续 prefetch 更远的 window.

        策略:
        1. 检查 next + next_next 是否已 resident
        2. 如果都已 resident, prefetch far_keys (3 windows ahead)
        3. 如果 far_keys 也 resident, 计算更远的 window (4+ ahead)
        """
        with self._lock:
            sessions = list(self._active_sessions.values())
        if not sessions:
            return

        for session in sessions:
            plan = session.plan
            if not plan.enabled:
                continue

            # 检查 next + next_next 是否已 resident
            with self._lock:
                ahead_keys = list(plan.next_keys) + list(plan.next_next_keys)
                ahead_resident = all(k in self._resident for k in ahead_keys) if ahead_keys else True

                if not ahead_resident:
                    # next/next_next 还没就绪, 补充 prefetch (可能之前 evicted)
                    missing = [k for k in ahead_keys if k not in self._resident and k not in self._inflight_prefetch]
                    if missing:
                        self._schedule_prefetch(missing)
                    continue

                # next + next_next 已就绪, prefetch far_keys
                if self._pipeline_lookahead_far and plan.far_keys:
                    far_missing = [
                        k for k in plan.far_keys
                        if k not in self._resident and k not in self._inflight_prefetch
                    ]
                    if far_missing:
                        scheduled = self._schedule_prefetch(far_missing)
                        if scheduled:
                            session.prefetched_far_keys = list(set(
                                session.prefetched_far_keys + scheduled
                            ))

    def advance_window(self, session: "ExpertRequestSession") -> None:
        """外部 decode 循环通知 window 推进 (可选).

        当 decode 完成一个 layer window 后调用, 推进 cursor 并触发更远的 prefetch.
        这让 pipeline 能感知实际 decode 进度, 而非仅靠时间推测.

        Args:
            session: 当前 request 的 session
        """
        if session is None or not session.plan.enabled:
            return
        with self._lock:
            plan = session.plan
            session.advance_count += 1
            # 推进 frontier cursor (用原 frontier_key, 沿原 thread 滚)
            self._frontier_cursor[plan.frontier_key] = plan.next_layer_cursor

            # P2+ fix: 用 _frontier_key_override 避免二次包装 frontier_key
            # 否则 _build_plan 会把完整 frontier_key 当 frontier_id 再包一层
            # P2+ fix: 沿用原 plan.prompt_hash, 保持 _select_from_layer 的 route seed 不变
            # 否则 expert routing 会脱离原 prompt/request 假设
            new_plan = self._build_plan(
                model_name=plan.model,
                family=plan.family,
                route_mode=plan.route_mode,
                prompt_hash=plan.prompt_hash,
                frontier_id="",
                _frontier_key_override=plan.frontier_key,
            )
            session.plan = new_plan

            # 触发新一轮 prefetch
            scheduled_next = self._schedule_prefetch(new_plan.next_keys)
            scheduled_ahead = self._schedule_prefetch(new_plan.next_next_keys)
            session.prefetched_keys = scheduled_next
            session.prefetched_ahead_keys = scheduled_ahead

    # ===== P3+ dense residency monitor: 持续性 dense backbone residency =====

    def _update_dense_residency_monitor(self) -> None:
        """更新 expert 占用 EMA, 变化超阈值时标记 reload.

        在 begin_request / complete_request 时调用.
        """
        with self._lock:
            current_occupied_gb = int(self._stats["resident_bytes"]) / 1024**3

        # EMA (alpha=0.3, 平滑跟踪)
        if self._expert_occupied_ema == 0.0:
            self._expert_occupied_ema = current_occupied_gb
        else:
            self._expert_occupied_ema = 0.7 * self._expert_occupied_ema + 0.3 * current_occupied_gb

        # 检查变化是否超阈值
        if self._dense_residency_hint:
            last_budget = float(self._dense_residency_hint.get("dense_budget_gb", 0.0))
            # 重新计算当前 budget
            system_reserve_gb = self.ram_budget_bytes / 1024**3 * 0.2
            current_budget = max(
                0.0,
                self.ram_budget_bytes / 1024**3
                - self._expert_occupied_ema
                - 1.0  # kv_cache_reserve
                - system_reserve_gb,
            )
            if last_budget > 0:
                change_ratio = abs(current_budget - last_budget) / last_budget
                if change_ratio >= self._dense_reload_threshold:
                    # 重新计算 hint
                    model_size_gb = float(self._dense_residency_hint.get("_model_size_gb", 0.0))
                    total_layers = int(self._dense_residency_hint.get("_total_layers", 0))
                    model_path = str(self._dense_residency_hint.get("_model_path") or "").strip()
                    new_hint = self.recommend_dense_residency(
                        model_size_gb=model_size_gb,
                        total_layers=total_layers,
                    )
                    new_hint["_model_size_gb"] = model_size_gb
                    new_hint["_total_layers"] = total_layers
                    new_hint["_model_path"] = model_path
                    old_n_gpu = int(self._dense_residency_hint.get("suggested_n_gpu_layers", -1))
                    new_n_gpu = int(new_hint.get("suggested_n_gpu_layers", -1))
                    if new_n_gpu != old_n_gpu:
                        self._dense_residency_hint = new_hint
                        self._dense_reload_pending = True
                        self._dense_reload_n_gpu_layers = new_n_gpu
        else:
            # 首次: 初始化 hint (用默认值, edge_first_proxy 会在首次加载时设置真实 model_size)
            self._dense_residency_hint = {
                "dense_budget_gb": max(
                    0.0,
                    self.ram_budget_bytes / 1024**3 * 0.8 - self._expert_occupied_ema - 1.0,
                ),
                "expert_occupied_gb": round(self._expert_occupied_ema, 3),
                "can_full_resident": True,
                "suggested_n_gpu_layers": -1,
                "offload_ratio": 0.0,
                "reason": "monitor_init",
                "_model_size_gb": 0.0,
                "_total_layers": 0,
            }

    def set_dense_residency_baseline(
        self, *, model_size_gb: float, total_layers: int, model_path: str = ""
    ) -> None:
        """设置 dense residency baseline (edge_first_proxy 首次加载后调用).

        Args:
            model_size_gb: dense backbone 模型大小 (GB)
            total_layers: dense backbone 总层数
        """
        hint = self.recommend_dense_residency(
            model_size_gb=model_size_gb,
            total_layers=total_layers,
        )
        hint["_model_size_gb"] = model_size_gb
        hint["_total_layers"] = total_layers
        hint["_model_path"] = str(model_path or "").strip()
        with self._lock:
            self._dense_residency_hint = hint
            self._dense_reload_pending = False
            self._dense_reload_n_gpu_layers = int(hint.get("suggested_n_gpu_layers", -1))

    def check_dense_residency_reload(self) -> Optional[dict[str, Any]]:
        """检查是否需要 reload dense backbone (edge_first_proxy 在 request 间调用).

        Returns:
            如果需要 reload: {"n_gpu_layers": int, "reason": str, "hint": dict}
            否则: None
        """
        forced_raw = str(os.environ.get("EDGE_LOCAL_N_GPU_LAYERS", "") or "").strip()
        if forced_raw:
            return None
        # 先更新 EMA
        self._update_dense_residency_monitor()

        with self._lock:
            if not self._dense_reload_pending:
                return None
            self._dense_reload_pending = False  # 消费标记
            return {
                "n_gpu_layers": self._dense_reload_n_gpu_layers,
                "reason": self._dense_residency_hint.get("reason", ""),
                "hint": dict(self._dense_residency_hint),
            }

    def recommend_dense_residency(
        self,
        *,
        model_size_gb: float = 0.0,
        total_layers: int = 0,
        kv_cache_reserve_gb: float = 1.0,
    ) -> dict[str, Any]:
        """估算 dense backbone 可用的 GPU/memory 预算与建议 offload 比例.

        colibri 设计: dense backbone 常驻内存, routed experts 落盘 NVMe.
        当 expert 缓存占用大时, dense 可用预算减少, 需部分层 offload 到 CPU.

        Args:
            model_size_gb: dense backbone 模型大小 (GB)
            total_layers: dense backbone 总层数 (用于估算 n_gpu_layers)
            kv_cache_reserve_gb: KV cache 预留 (GB), 默认 1GB

        Returns:
            {
                "dense_budget_gb": float,        # dense 可用内存预算
                "expert_occupied_gb": float,     # expert 已占用
                "can_full_resident": bool,       # 是否可全 resident
                "suggested_n_gpu_layers": int,   # 建议 GPU 层数 (-1=全部, 0=CPU)
                "offload_ratio": float,          # offload 比例 (0=全GPU, 1=全CPU)
                "reason": str,
            }
        """
        with self._lock:
            resident_gb = int(self._stats["resident_bytes"]) / 1024**3
            pinned_gb = int(self._stats["pinned_bytes"]) / 1024**3
            # expert 占用 = resident (含 pinned), pinned 不可被驱逐
            expert_occupied_gb = resident_gb

        # dense 可用 = ram_budget - expert_resident - kv_cache - 系统 reserve
        # 系统 reserve: 20% of ram_budget (留给 OS + 其他进程)
        system_reserve_gb = self.ram_budget_bytes / 1024**3 * 0.2
        dense_budget_gb = max(
            0.0,
            self.ram_budget_bytes / 1024**3
            - expert_occupied_gb
            - kv_cache_reserve_gb
            - system_reserve_gb,
        )

        if model_size_gb <= 0:
            return {
                "dense_budget_gb": round(dense_budget_gb, 3),
                "expert_occupied_gb": round(expert_occupied_gb, 3),
                "can_full_resident": True,  # 未知大小, 默认全 resident
                "suggested_n_gpu_layers": -1,
                "offload_ratio": 0.0,
                "reason": "model_size未知, 默认全 resident",
            }

        can_full = dense_budget_gb >= model_size_gb
        forced_raw = str(os.environ.get("EDGE_LOCAL_N_GPU_LAYERS", "") or "").strip()
        if forced_raw:
            try:
                forced_n_gpu = int(forced_raw)
                return {
                    "dense_budget_gb": round(dense_budget_gb, 3),
                    "expert_occupied_gb": round(expert_occupied_gb, 3),
                    "can_full_resident": can_full,
                    "suggested_n_gpu_layers": forced_n_gpu,
                    "offload_ratio": 0.0 if forced_n_gpu == -1 else round(max(0.0, 1.0 - (forced_n_gpu / max(total_layers, 1))), 3),
                    "reason": f"forced_n_gpu_layers={forced_n_gpu}",
                }
            except ValueError:
                pass
        if can_full:
            return {
                "dense_budget_gb": round(dense_budget_gb, 3),
                "expert_occupied_gb": round(expert_occupied_gb, 3),
                "can_full_resident": True,
                "suggested_n_gpu_layers": -1,  # 全 GPU
                "offload_ratio": 0.0,
                "reason": f"budget {dense_budget_gb:.1f}GB >= model {model_size_gb:.1f}GB",
            }

        # 预算不足: 按比例 offload 到 CPU
        offload_ratio = 1.0 - (dense_budget_gb / max(model_size_gb, 0.01))
        offload_ratio = min(max(offload_ratio, 0.0), 0.9)  # 最多 offload 90% (留至少几层在 GPU)
        if total_layers > 0:
            n_gpu = max(int(total_layers * (1.0 - offload_ratio)), 1)
        else:
            n_gpu = -1 if offload_ratio < 0.1 else 0  # 无层数信息: 小 offload 全GPU, 大 offload 全CPU

        return {
            "dense_budget_gb": round(dense_budget_gb, 3),
            "expert_occupied_gb": round(expert_occupied_gb, 3),
            "can_full_resident": False,
            "suggested_n_gpu_layers": n_gpu,
            "offload_ratio": round(offload_ratio, 3),
            "reason": f"budget {dense_budget_gb:.1f}GB < model {model_size_gb:.1f}GB, offload {offload_ratio:.0%} to CPU",
        }


_MANAGER: Optional[FullExpertDataPlaneManager] = None
_MANAGER_LOCK = threading.Lock()


def get_expert_data_plane_manager() -> FullExpertDataPlaneManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = FullExpertDataPlaneManager()
        return _MANAGER
