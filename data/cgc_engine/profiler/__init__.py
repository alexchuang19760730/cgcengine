# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CGC 性能分析與可視化模組

功能：
1. 整圖性能剖析：計算圖熱力圖、算子耗時統計、通信延遲分解
2. 記憶體生命週期可視化：記憶體洩漏/浪費檢測
3. 張量時間線追蹤：分配/拷貝/釋放時間線
4. 動態精度選擇：編譯期數值範圍分析，動態選擇 FP8/BF16/INT4

整合：
- CGC Dashboard (cgc_dashboard.py)
- Visualizer (visualizer.py)
- Profiler (profiler.py)
"""

import time
import threading
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime
import uuid
import weakref

import torch
import torch.nn as nn

try:
    from ..cgc.cgc_opcodes import CGC_OP_CODES, CGC_CATEGORIES
    from ..cgc.cgc_simd_executor import CGCExecutor, CGCCommand
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False


class MemorySegment:
    """記憶體區段"""
    def __init__(
        self,
        name: str,
        size_bytes: int,
        device: str,
        allocation_time: float,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.size_bytes = size_bytes
        self.device = device
        self.allocation_time = allocation_time
        self.release_time: Optional[float] = None
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.access_count = 0
        self.last_access_time = allocation_time

    def access(self, timestamp: float):
        self.access_count += 1
        self.last_access_time = timestamp

    def release(self, timestamp: float):
        self.release_time = timestamp

    @property
    def lifetime_ms(self) -> float:
        if self.release_time:
            return (self.release_time - self.allocation_time) * 1000
        return (time.time() - self.allocation_time) * 1000

    @property
    def is_leaked(self) -> bool:
        return self.release_time is None and self.access_count <= 1


class MemoryTracker:
    """記憶體生命週期追蹤器"""

    def __init__(self, enable_leak_detection: bool = True):
        self.enable_leak_detection = enable_leak_detection
        self.segments: Dict[str, MemorySegment] = {}
        self.timeline: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._tensor_refs: Dict[int, weakref.ref] = {}
        self.peak_usage = 0
        self.peak_time: Optional[float] = None
        self.device_usage: Dict[str, int] = defaultdict(int)

    def track_allocation(
        self,
        name: str,
        tensor: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> str:
        """追蹤張量分配"""
        timestamp = time.time()
        segment = MemorySegment(
            name=name,
            size_bytes=tensor.numel() * tensor.element_size(),
            device=str(tensor.device),
            allocation_time=timestamp,
            tensor_shape=tuple(tensor.shape),
            dtype=tensor.dtype,
        )
        segment.metadata = metadata or {}

        with self._lock:
            self.segments[segment.id] = segment
            self._tensor_refs[id(tensor)] = weakref.ref(tensor)
            self._update_peak(timestamp)

            self.timeline.append({
                "event": "allocate",
                "segment_id": segment.id,
                "name": name,
                "size_bytes": segment.size_bytes,
                "device": segment.device,
                "timestamp": timestamp,
            })

        return segment.id

    def track_copy(
        self,
        segment_id: str,
        src_device: str,
        dst_device: str,
        bytes_transferred: int,
    ):
        """追蹤張量拷貝"""
        timestamp = time.time()

        with self._lock:
            if segment_id in self.segments:
                self.segments[segment_id].access(timestamp)

            self.timeline.append({
                "event": "copy",
                "segment_id": segment_id,
                "src_device": src_device,
                "dst_device": dst_device,
                "bytes_transferred": bytes_transferred,
                "timestamp": timestamp,
            })

    def track_release(self, segment_id: str):
        """追蹤張量釋放"""
        timestamp = time.time()

        with self._lock:
            if segment_id in self.segments:
                self.segments[segment_id].release(timestamp)

                self.timeline.append({
                    "event": "release",
                    "segment_id": segment_id,
                    "lifetime_ms": self.segments[segment_id].lifetime_ms,
                    "timestamp": timestamp,
                })

    def _update_peak(self, timestamp: float):
        """更新峰值使用量"""
        current_usage = sum(
            seg.size_bytes for seg in self.segments.values()
            if seg.release_time is None
        )

        if current_usage > self.peak_usage:
            self.peak_usage = current_usage
            self.peak_time = timestamp

        self.device_usage[str(timestamp)] = current_usage

    def detect_leaks(self) -> List[MemorySegment]:
        """檢測記憶體洩漏"""
        if not self.enable_leak_detection:
            return []

        leaked = []
        for seg in self.segments.values():
            if seg.is_leaked:
                leaked.append(seg)
        return leaked

    def get_waste_analysis(self) -> Dict[str, Any]:
        """分析記憶體浪費"""
        active_segs = [s for s in self.segments.values() if s.release_time is None]
        low_access = [s for s in active_segs if s.access_count <= 2]

        return {
            "total_segments": len(self.segments),
            "active_segments": len(active_segs),
            "peak_usage_bytes": self.peak_usage,
            "peak_time": self.peak_time,
            "low_access_segments": len(low_access),
            "potential_waste_bytes": sum(s.size_bytes for s in low_access),
            "leaked_segments": len(self.detect_leaks()),
        }

    def export_timeline(self, output_path: str):
        """導出時間線為 JSON"""
        with open(output_path, 'w') as f:
            json.dump({
                "segments": [asdict(s) for s in self.segments.values()],
                "timeline": self.timeline,
                "peak_usage": self.peak_usage,
                "leaks": [asdict(s) for s in self.detect_leaks()],
            }, f, indent=2)

    def generate_memory_heatmap(self) -> Dict[str, float]:
        """生成記憶體使用熱力圖"""
        device_timeline: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

        sorted_events = sorted(self.timeline, key=lambda x: x["timestamp"])

        for event in sorted_events:
            if event["event"] in ("allocate", "release"):
                device = event["device"]
                size = event.get("size_bytes", 0)
                device_timeline[device].append((event["timestamp"], size))

        return dict(device_timeline)


class TensorTimeline:
    """張量時間線追蹤器"""

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.tensor_history: Dict[int, List[Dict]] = {}
        self._lock = threading.Lock()

    def record_allocation(
        self,
        tensor_id: int,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: str,
        size_bytes: int,
    ):
        """記錄張量分配"""
        event = {
            "type": "allocation",
            "tensor_id": tensor_id,
            "name": name,
            "shape": shape,
            "dtype": str(dtype),
            "device": device,
            "size_bytes": size_bytes,
            "timestamp": time.time(),
        }

        with self._lock:
            self.events.append(event)
            if tensor_id not in self.tensor_history:
                self.tensor_history[tensor_id] = []
            self.tensor_history[tensor_id].append(event)

    def record_copy(
        self,
        tensor_id: int,
        src: str,
        dst: str,
        bytes_transferred: int,
    ):
        """記錄張量拷貝"""
        event = {
            "type": "copy",
            "tensor_id": tensor_id,
            "src": src,
            "dst": dst,
            "bytes_transferred": bytes_transferred,
            "timestamp": time.time(),
        }

        with self._lock:
            self.events.append(event)
            if tensor_id in self.tensor_history:
                self.tensor_history[tensor_id].append(event)

    def record_release(self, tensor_id: int):
        """記錄張量釋放"""
        event = {
            "type": "release",
            "tensor_id": tensor_id,
            "timestamp": time.time(),
        }

        with self._lock:
            self.events.append(event)
            if tensor_id in self.tensor_history:
                self.tensor_history[tensor_id].append(event)

    def get_tensor_lifecycle(self, tensor_id: int) -> List[Dict]:
        """獲取張量生命週期"""
        with self._lock:
            return self.tensor_history.get(tensor_id, [])

    def export_to_chrome_trace(self, output_path: str):
        """導出為 Chrome 追蹤格式"""
        trace_events = []

        for event in self.events:
            trace_events.append({
                "name": f"{event['type']}: {event.get('name', event['tensor_id'])}",
                "cat": event["type"],
                "ts": event["timestamp"] * 1_000_000,
                "pid": 1,
                "tid": hash(event.get("device", "cpu")) % 1000,
                "args": event,
            })

        with open(output_path, 'w') as f:
            json.dump({"traceEvents": trace_events}, f)


class OperatorStats:
    """算子統計"""
    def __init__(self, opcode: int, name: str):
        self.opcode = opcode
        self.name = name
        self.total_time_us = 0
        self.total_calls = 0
        self.min_time_us = float('inf')
        self.max_time_us = 0
        self.input_shapes: List[str] = []
        self.output_shapes: List[str] = []

    def record(self, duration_us: int, input_shape: str = "", output_shape: str = ""):
        self.total_time_us += duration_us
        self.total_calls += 1
        self.min_time_us = min(self.min_time_us, duration_us)
        self.max_time_us = max(self.max_time_us, duration_us)
        if input_shape:
            self.input_shapes.append(input_shape)
        if output_shape:
            self.output_shapes.append(output_shape)

    @property
    def avg_time_us(self) -> float:
        return self.total_time_us / self.total_calls if self.total_calls > 0 else 0

    def to_dict(self) -> Dict:
        return {
            "opcode": f"0x{self.opcode:02x}",
            "name": self.name,
            "total_time_ms": self.total_time_us / 1000,
            "calls": self.total_calls,
            "avg_time_us": self.avg_time_us,
            "min_time_us": self.min_time_us if self.min_time_us != float('inf') else 0,
            "max_time_us": self.max_time_us,
        }


class CGCPerformanceAnalyzer:
    """CGC 性能分析器"""

    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.operator_stats: Dict[int, OperatorStats] = {}
        self.communication_latency: Dict[str, List[float]] = defaultdict(list)
        self.moe_expert_stats: Dict[int, Dict[str, Any]] = {}
        self.fsdp_latency: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.memory_tracker = MemoryTracker()
        self.tensor_timeline = TensorTimeline()

    def start_op(self, opcode: int, name: str) -> int:
        """開始算子計時"""
        if not self.enable_profiling:
            return 0
        return time.perf_counter_ns()

    def end_op(
        self,
        opcode: int,
        name: str,
        start_time_ns: int,
        input_shape: str = "",
        output_shape: str = "",
    ):
        """結束算子計時"""
        if not self.enable_profiling or start_time_ns == 0:
            return

        duration_us = (time.perf_counter_ns() - start_time_ns) // 1000

        with self._lock:
            if opcode not in self.operator_stats:
                self.operator_stats[opcode] = OperatorStats(opcode, name)
            self.operator_stats[opcode].record(duration_us, input_shape, output_shape)

    def record_communication(
        self,
        comm_type: str,
        size_bytes: int,
        duration_us: float,
    ):
        """記錄通信延遲"""
        with self._lock:
            self.communication_latency[comm_type].append(duration_us)

    def record_moe_expert(
        self,
        expert_id: int,
        load_time_us: float,
        compute_time_us: float,
        memory_bytes: int,
    ):
        """記錄 MoE 專家統計"""
        with self._lock:
            if expert_id not in self.moe_expert_stats:
                self.moe_expert_stats[expert_id] = {
                    "loads": [],
                    "computes": [],
                    "memory_usage": [],
                }
            self.moe_expert_stats[expert_id]["loads"].append(load_time_us)
            self.moe_expert_stats[expert_id]["computes"].append(compute_time_us)
            self.moe_expert_stats[expert_id]["memory_usage"].append(memory_bytes)

    def record_fsdp_operation(
        self,
        operation: str,
        shard_size_bytes: int,
        allreduce_time_us: float,
        gather_time_us: float,
    ):
        """記錄 FSDP 操作"""
        with self._lock:
            self.fsdp_latency[operation] = {
                "shard_size_bytes": shard_size_bytes,
                "allreduce_time_us": allreduce_time_us,
                "gather_time_us": gather_time_us,
                "total_time_us": allreduce_time_us + gather_time_us,
            }

    def generate_heatmap_data(self) -> Dict[str, Any]:
        """生成計算圖熱力圖數據"""
        heatmap = {}
        for opcode, stats in self.operator_stats.items():
            heatmap[stats.name] = {
                "total_time_ms": stats.total_time_us / 1000,
                "avg_time_us": stats.avg_time_us,
                "calls": stats.total_calls,
                "percentage": 0,
            }

        total_time = sum(s.total_time_us for s in self.operator_stats.values())
        if total_time > 0:
            for name, data in heatmap.items():
                data["percentage"] = (data["total_time_ms"] * 1000 / total_time) * 100

        return heatmap

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """瓶頸分析"""
        sorted_ops = sorted(
            self.operator_stats.values(),
            key=lambda x: x.total_time_us,
            reverse=True
        )

        top_5 = [op.to_dict() for op in sorted_ops[:5]]

        comm_total = sum(sum(times) for times in self.communication_latency.values())
        compute_total = sum(op.total_time_us for op in self.operator_stats.values())

        moe_stats = {}
        for expert_id, stats in self.moe_expert_stats.items():
            moe_stats[f"expert_{expert_id}"] = {
                "avg_load_time_us": sum(stats["loads"]) / len(stats["loads"]) if stats["loads"] else 0,
                "avg_compute_time_us": sum(stats["computes"]) / len(stats["computes"]) if stats["computes"] else 0,
                "total_memory_bytes": sum(stats["memory_usage"]),
            }

        return {
            "top_5_slowest_operators": top_5,
            "total_compute_time_ms": compute_total / 1000,
            "total_comm_time_ms": comm_total / 1000,
            "comm_vs_compute_ratio": comm_total / compute_total if compute_total > 0 else 0,
            "moe_expert_stats": moe_stats,
            "fsdp_latency": self.fsdp_latency,
        }

    def export_report(self, output_dir: str):
        """導出性能報告"""
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "operator_stats": {f"0x{opcode:02x}": stats.to_dict() for opcode, stats in self.operator_stats.items()},
            "communication_latency": dict(self.communication_latency),
            "bottleneck_analysis": self.get_bottleneck_analysis(),
            "memory_analysis": self.memory_tracker.get_waste_analysis(),
        }

        with open(os.path.join(output_dir, "performance_report.json"), 'w') as f:
            json.dump(report, f, indent=2)

        self.memory_tracker.export_timeline(os.path.join(output_dir, "memory_timeline.json"))
        self.tensor_timeline.export_to_chrome_trace(os.path.join(output_dir, "tensor_trace.json"))


class DynamicPrecisionSelector:
    """動態精度選擇器 - 編譯期數值範圍分析"""

    def __init__(self):
        self.tensor_ranges: Dict[str, Tuple[float, float]] = {}
        self.recommended_precision: Dict[str, torch.dtype] = {}
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}

    def analyze_tensor_range(
        self,
        name: str,
        tensor: torch.Tensor,
        sample_size: int = 1000,
    ) -> Dict[str, Any]:
        """分析張量數值範圍"""
        if name in self.analysis_cache:
            return self.analysis_cache[name]

        tensor_flat = tensor.flatten().float()

        if tensor_flat.numel() > sample_size:
            indices = torch.randperm(tensor_flat.numel())[:sample_size]
            sample = tensor_flat[indices]
        else:
            sample = tensor_flat

        abs_max = sample.abs().max().item()
        abs_min = sample.abs().min().item()
        mean = sample.mean().item()
        std = sample.std().item()

        inf_count = tensor_flat.isinf().sum().item()
        nan_count = tensor_flat.isnan().sum().item()

        dynamic_range = abs_max / (abs_min + 1e-8)

        analysis = {
            "name": name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "abs_max": abs_max,
            "abs_min": abs_min,
            "mean": mean,
            "std": std,
            "dynamic_range": dynamic_range,
            "inf_count": inf_count,
            "nan_count": nan_count,
        }

        self.tensor_ranges[name] = (abs_min, abs_max)
        self.analysis_cache[name] = analysis

        return analysis

    def recommend_precision(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> torch.dtype:
        """推薦精度"""
        analysis = self.analyze_tensor_range(name, tensor)

        abs_max = analysis["abs_max"]
        dynamic_range = analysis["dynamic_range"]

        if dynamic_range < 10 and abs_max < 448:
            if torch.cuda.is_available():
                recommended = torch.float8_e4m3fn
            else:
                recommended = torch.float16
        elif abs_max < 6e4:
            recommended = torch.bfloat16
        elif abs_max < 1e5:
            recommended = torch.float16
        else:
            recommended = torch.float32

        self.recommended_precision[name] = recommended

        return recommended

    def get_precision_stats(self) -> Dict[str, Any]:
        """獲取精度選擇統計"""
        precision_counts: Dict[str, int] = defaultdict(int)

        for name, dtype in self.recommended_precision.items():
            precision_counts[str(dtype)] += 1

        return {
            "total_tensors": len(self.recommended_precision),
            "precision_distribution": dict(precision_counts),
            "tensor_ranges": self.tensor_ranges,
        }

    def export_analysis(self, output_path: str):
        """導出分析結果"""
        with open(output_path, 'w') as f:
            json.dump({
                "analysis": self.analysis_cache,
                "recommended_precision": {k: str(v) for k, v in self.recommended_precision.items()},
                "precision_stats": self.get_precision_stats(),
            }, f, indent=2)


class UnifiedProfiler:
    """統一性能分析器 - 整合所有分析功能"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.performance_analyzer = CGCPerformanceAnalyzer()
        self.memory_tracker = MemoryTracker()
        self.tensor_timeline = TensorTimeline()
        self.precision_selector = DynamicPrecisionSelector()
        self._initialized = True

    def profile_cgc_command(
        self,
        command: "CGCCommand",
        execute_fn,
    ) -> Any:
        """性能分析 CGC 命令執行"""
        opcode = command.opcode
        opcode_name = getattr(CGC_OP_CODES, f"0x{opcode:02X}", "UNKNOWN")

        start_time_ns = self.performance_analyzer.start_op(opcode, opcode_name)

        try:
            result = execute_fn()
            return result
        finally:
            input_shapes = [str(t.shape) for t in command.inputs]
            output_shapes = []

            self.performance_analyzer.end_op(
                opcode, opcode_name, start_time_ns,
                input_shape=str(input_shapes),
                output_shape=str(output_shapes),
            )

    def start_memory_tracking(self, name: str, tensor: torch.Tensor, metadata: Dict = None):
        """開始記憶體追蹤"""
        segment_id = self.memory_tracker.track_allocation(name, tensor, metadata)
        self.tensor_timeline.record_allocation(
            tensor_id=id(tensor),
            name=name,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=str(tensor.device),
            size_bytes=tensor.numel() * tensor.element_size(),
        )
        return segment_id

    def end_memory_tracking(self, segment_id: str):
        """結束記憶體追蹤"""
        self.memory_tracker.track_release(segment_id)

    def analyze_and_recommend(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型並推薦精度"""
        recommendations = {}

        for name, param in model.named_parameters():
            dtype = self.precision_selector.recommend_precision(name, param)
            recommendations[name] = {
                "current_dtype": str(param.dtype),
                "recommended_dtype": str(dtype),
                "analysis": self.precision_selector.analysis_cache.get(name, {}),
            }

        return recommendations

    def generate_full_report(self, output_dir: str = "./profiler_reports"):
        """生成完整報告"""
        os.makedirs(output_dir, exist_ok=True)

        self.performance_analyzer.export_report(output_dir)
        self.precision_selector.export_analysis(os.path.join(output_dir, "precision_analysis.json"))

        summary = {
            "timestamp": datetime.now().isoformat(),
            "performance": self.performance_analyzer.get_bottleneck_analysis(),
            "memory": self.memory_tracker.get_waste_analysis(),
            "precision": self.precision_selector.get_precision_stats(),
        }

        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


def get_profiler() -> UnifiedProfiler:
    """獲取統一性能分析器單例"""
    return UnifiedProfiler()
