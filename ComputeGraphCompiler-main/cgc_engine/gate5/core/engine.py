"""
CGC Gate 5.0 核心执行引擎
实现四大能力：可审计、可追踪、可回溯、可可视化
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import json
import time
import hashlib
import uuid
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path


@dataclass
class ExecutionContext:
    """执行上下文"""
    task_id: str
    user_id: Optional[str]
    timestamp: float
    metadata: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    status: str = "running"
    error: Optional[str] = None


@dataclass
class TraceSpan:
    """调用链 Span"""
    span_id: str
    parent_id: Optional[str]
    task_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "active"
    metadata: Dict[str, Any] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.metrics is None:
            self.metrics = {}
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class AuditRecord:
    """审计记录"""
    audit_id: str
    task_id: str
    user_id: Optional[str]
    action: str
    timestamp: float
    details: Dict[str, Any]
    status: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class Snapshot:
    """执行快照"""
    snapshot_id: str
    task_id: str
    timestamp: float
    state: Dict[str, Any]
    context: ExecutionContext
    spans: List[TraceSpan] = None
    
    def __post_init__(self):
        if self.spans is None:
            self.spans = []


class StorageBackend(ABC):
    """存储后端基类"""
    
    @abstractmethod
    def save_audit(self, record: AuditRecord) -> None:
        pass
    
    @abstractmethod
    def load_audit(self, audit_id: str) -> Optional[AuditRecord]:
        pass
    
    @abstractmethod
    def save_trace(self, span: TraceSpan) -> None:
        pass
    
    @abstractmethod
    def load_trace(self, task_id: str) -> List[TraceSpan]:
        pass
    
    @abstractmethod
    def save_snapshot(self, snapshot: Snapshot) -> None:
        pass
    
    @abstractmethod
    def load_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        pass
    
    @abstractmethod
    def list_snapshots(self, task_id: str) -> List[Snapshot]:
        pass


class FileStorageBackend(StorageBackend):
    """文件系统存储后端"""
    
    def __init__(self, base_path: str = "gate5_data"):
        self.base_path = Path(base_path)
        self.audit_path = self.base_path / "audit"
        self.trace_path = self.base_path / "trace"
        self.snapshot_path = self.base_path / "snapshot"
        
        # 确保目录存在
        self.audit_path.mkdir(parents=True, exist_ok=True)
        self.trace_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.mkdir(parents=True, exist_ok=True)
    
    def _serialize(self, obj: Any) -> str:
        """序列化对象"""
        return json.dumps(obj, default=lambda o: o.__dict__, ensure_ascii=False, indent=2)
    
    def _deserialize(self, data: str, cls):
        """反序列化对象"""
        raw = json.loads(data)
        return cls(**raw)
    
    def save_audit(self, record: AuditRecord) -> None:
        file_path = self.audit_path / f"{record.audit_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self._serialize(record))
    
    def load_audit(self, audit_id: str) -> Optional[AuditRecord]:
        file_path = self.audit_path / f"{audit_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return self._deserialize(f.read(), AuditRecord)
        return None
    
    def save_trace(self, span: TraceSpan) -> None:
        task_dir = self.trace_path / span.task_id
        task_dir.mkdir(exist_ok=True)
        file_path = task_dir / f"{span.span_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self._serialize(span))
    
    def load_trace(self, task_id: str) -> List[TraceSpan]:
        task_dir = self.trace_path / task_id
        spans = []
        if task_dir.exists():
            for file in task_dir.glob("*.json"):
                with open(file, "r", encoding="utf-8") as f:
                    spans.append(self._deserialize(f.read(), TraceSpan))
        return sorted(spans, key=lambda s: s.start_time)
    
    def save_snapshot(self, snapshot: Snapshot) -> None:
        file_path = self.snapshot_path / f"{snapshot.snapshot_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self._serialize(snapshot))
    
    def load_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        file_path = self.snapshot_path / f"{snapshot_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return self._deserialize(f.read(), Snapshot)
        return None
    
    def list_snapshots(self, task_id: str) -> List[Snapshot]:
        snapshots = []
        for file in self.snapshot_path.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if raw.get("task_id") == task_id:
                    snapshots.append(Snapshot(**raw))
        return sorted(snapshots, key=lambda s: s.timestamp)


class Gate5Engine:
    """CGC Gate 5.0 核心引擎"""
    
    def __init__(self, storage_backend: Optional[StorageBackend] = None):
        self.storage = storage_backend or FileStorageBackend()
        self.active_tasks: Dict[str, ExecutionContext] = {}
        self.active_spans: Dict[str, TraceSpan] = {}
        self._audit_enabled = True
        self._trace_enabled = True
        self._snapshot_enabled = True
    
    @property
    def audit_enabled(self) -> bool:
        return self._audit_enabled
    
    @audit_enabled.setter
    def audit_enabled(self, value: bool):
        self._audit_enabled = value
    
    @property
    def trace_enabled(self) -> bool:
        return self._trace_enabled
    
    @trace_enabled.setter
    def trace_enabled(self, value: bool):
        self._trace_enabled = value
    
    @property
    def snapshot_enabled(self) -> bool:
        return self._snapshot_enabled
    
    @snapshot_enabled.setter
    def snapshot_enabled(self, value: bool):
        self._snapshot_enabled = value
    
    def create_task(self, user_id: Optional[str], inputs: Dict[str, Any]) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        context = ExecutionContext(
            task_id=task_id,
            user_id=user_id,
            timestamp=time.time(),
            metadata={},
            inputs=inputs,
        )
        self.active_tasks[task_id] = context
        
        # 记录审计
        self._audit(task_id, user_id, "task_created", {"inputs": inputs})
        
        # 创建根 Span
        self.start_span(task_id, "task", None)
        
        return task_id
    
    def start_span(
        self,
        task_id: str,
        name: str,
        parent_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """创建并启动一个 Trace Span"""
        if not self.trace_enabled:
            return ""
        
        span_id = str(uuid.uuid4())
        span = TraceSpan(
            span_id=span_id,
            parent_id=parent_id,
            task_id=task_id,
            name=name,
            start_time=time.time(),
            metadata=dict(metadata or {}),
        )
        self.active_spans[span_id] = span
        return span_id
    
    def end_span(self, span_id: str, status: str = "completed", **metrics) -> None:
        """结束一个 Trace Span"""
        if span_id not in self.active_spans:
            return
        
        span = self.active_spans[span_id]
        span.end_time = time.time()
        span.status = status
        span.metrics.update(metrics)
        
        # 保存 Span
        self.storage.save_trace(span)
        del self.active_spans[span_id]
    
    def update_task(self, task_id: str, **updates) -> None:
        """更新任务状态"""
        if task_id not in self.active_tasks:
            return
        
        context = self.active_tasks[task_id]
        for key, value in updates.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        # 如果任务完成，结束根 Span 并创建快照
        if context.status in ["completed", "failed"]:
            self._complete_task(task_id)
    
    def _complete_task(self, task_id: str) -> None:
        """任务完成处理"""
        # 结束所有相关 Span
        for span_id in list(self.active_spans.keys()):
            span = self.active_spans.get(span_id)
            if span is None:
                continue
            if span.task_id == task_id:
                self.end_span(span_id, "completed")
        
        # 记录审计
        context = self.active_tasks.get(task_id)
        if context:
            self._audit(
                task_id,
                context.user_id,
                "task_completed",
                {"status": context.status, "error": context.error}
            )
            
            # 创建完成快照
            if self.snapshot_enabled:
                self.create_snapshot(task_id)
        
        # 从活跃任务中移除
        self.active_tasks.pop(task_id, None)
    
    def _audit(self, task_id: str, user_id: Optional[str], action: str, details: Dict[str, Any]) -> None:
        """记录审计日志"""
        if not self.audit_enabled:
            return
        
        record = AuditRecord(
            audit_id=str(uuid.uuid4()),
            task_id=task_id,
            user_id=user_id,
            action=action,
            timestamp=time.time(),
            details=details,
            status="success",
        )
        self.storage.save_audit(record)
    
    def create_snapshot(self, task_id: str) -> str:
        """创建执行快照"""
        if not self.snapshot_enabled:
            return ""
        
        snapshot_id = str(uuid.uuid4())
        context = self.active_tasks.get(task_id)
        
        # 获取所有相关 Span
        spans = self.storage.load_trace(task_id)
        
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            task_id=task_id,
            timestamp=time.time(),
            state={
                "active_tasks": len(self.active_tasks),
                "active_spans": len(self.active_spans),
            },
            context=context,
            spans=spans,
        )
        
        self.storage.save_snapshot(snapshot)
        return snapshot_id
    
    def replay_task(self, task_id: str, speed: float = 1.0) -> List[Dict[str, Any]]:
        """回溯执行任务"""
        snapshots = self.storage.list_snapshots(task_id)
        if not snapshots:
            return []
        
        # 按时间排序
        snapshots.sort(key=lambda s: s.timestamp)
        
        replay_events = []
        for snapshot in snapshots:
            replay_events.append({
                "timestamp": snapshot.timestamp,
                "snapshot_id": snapshot.snapshot_id,
                "state": snapshot.state,
                "context": snapshot.context.__dict__ if snapshot.context else None,
                "spans": [s.__dict__ for s in snapshot.spans] if snapshot.spans else [],
            })
        
        return replay_events
    
    def get_task_history(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取任务历史"""
        # 从审计日志中重建任务历史
        history = []
        for file in Path(self.storage.audit_path).glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                record = json.load(f)
                if user_id is None or record.get("user_id") == user_id:
                    history.append(record)
        
        return sorted(history, key=lambda r: r["timestamp"], reverse=True)
    
    def get_task_trace(self, task_id: str) -> Dict[str, Any]:
        """获取任务完整追踪信息"""
        spans = self.storage.load_trace(task_id)
        snapshots = self.storage.list_snapshots(task_id)
        
        # 构建调用链树
        span_tree = self._build_span_tree(spans)
        
        return {
            "task_id": task_id,
            "spans": [s.__dict__ for s in spans],
            "span_tree": span_tree,
            "snapshots": len(snapshots),
            "total_duration": sum(s.duration for s in spans),
        }
    
    def _build_span_tree(self, spans: List[TraceSpan]) -> Dict[str, Any]:
        """构建 Span 调用树"""
        span_map = {s.span_id: s for s in spans}
        tree = []
        
        for span in spans:
            node = {
                "span_id": span.span_id,
                "name": span.name,
                "duration": span.duration,
                "status": span.status,
                "children": [],
            }
            
            if span.parent_id and span.parent_id in span_map:
                # 找到父节点并添加
                parent = span_map[span.parent_id]
                # 递归查找父节点在树中的位置
                self._add_child_to_tree(tree, parent.span_id, node)
            else:
                # 根节点
                tree.append(node)
        
        return tree
    
    def _add_child_to_tree(self, nodes: List[Dict], parent_id: str, child: Dict):
        """递归添加子节点"""
        for node in nodes:
            if node["span_id"] == parent_id:
                node["children"].append(child)
                return
            if node["children"]:
                self._add_child_to_tree(node["children"], parent_id, child)
    
    def generate_audit_report(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Dict[str, Any]:
        """生成审计报告"""
        records = []
        for file in Path(self.storage.audit_path).glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                record = json.load(f)
                ts = record["timestamp"]
                if (start_time is None or ts >= start_time) and (end_time is None or ts <= end_time):
                    records.append(record)
        
        # 统计
        stats = {
            "total_records": len(records),
            "tasks_created": sum(1 for r in records if r["action"] == "task_created"),
            "tasks_completed": sum(1 for r in records if r["action"] == "task_completed"),
            "errors": sum(1 for r in records if r.get("details", {}).get("status") == "failed"),
        }
        
        return {
            "report_id": str(uuid.uuid4()),
            "generated_at": time.time(),
            "time_range": {"start": start_time, "end": end_time},
            "statistics": stats,
            "records": records,
        }

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取单个任务详情"""
        # 从审计日志中重建任务信息
        for file in Path(self.storage.audit_path).glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                record = json.load(f)
                if record.get("task_id") == task_id and record.get("action") == "task_created":
                    return {
                        "task_id": task_id,
                        "user_id": record.get("user_id"),
                        "status": "completed",
                        "inputs": record.get("details", {}).get("inputs", {}),
                        "created_at": record.get("timestamp"),
                    }
        return None

    def list_tasks(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """列出所有任务"""
        tasks = {}
        for file in Path(self.storage.audit_path).glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                record = json.load(f)
                if record.get("action") == "task_created":
                    if user_id is None or record.get("user_id") == user_id:
                        tid = record.get("task_id")
                        if tid not in tasks:
                            tasks[tid] = {
                                "task_id": tid,
                                "user_id": record.get("user_id"),
                                "status": "running",
                                "inputs": record.get("details", {}).get("inputs", {}),
                                "created_at": record.get("timestamp"),
                            }
        return sorted(tasks.values(), key=lambda t: t.get("created_at", 0), reverse=True)[:limit]

    def query_audit(self, start_time: Optional[float] = None, end_time: Optional[float] = None,
                    action: Optional[str] = None, user_id: Optional[str] = None,
                    task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """查询审计日志"""
        records = []
        for file in Path(self.storage.audit_path).glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                record = json.load(f)
                ts = record.get("timestamp", 0)
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                if action and record.get("action") != action:
                    continue
                if user_id and record.get("user_id") != user_id:
                    continue
                if task_id and record.get("task_id") != task_id:
                    continue
                records.append(record)
        return sorted(records, key=lambda r: r.get("timestamp", 0), reverse=True)

    def export_trace(self, task_id: str, format: str = "json") -> str:
        """导出追踪数据"""
        trace = self.get_task_trace(task_id)
        if format == "json":
            return json.dumps(trace, ensure_ascii=False, indent=2, default=str)
        elif format == "csv":
            lines = ["span_id,name,start_time,end_time,duration,status"]
            for span in trace.get("spans", []):
                duration = span.get("end_time", time.time()) - span.get("start_time", 0)
                lines.append(f"{span.get('span_id','')},{span.get('name','')},{span.get('start_time','')},{span.get('end_time','')},{duration:.4f},{span.get('status','')}")
            return "\n".join(lines)
        else:
            return json.dumps(trace, ensure_ascii=False, indent=2, default=str)
