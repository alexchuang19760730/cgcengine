"""
统一知识库架构设计与实现
整合：后端感知 + 图结构感知 + 算子模式感知 + 硬件感知
"""

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from cgc_engine.utils.envs import cgc_output_dir


@dataclass
class KnowledgeEntry:
    """统一知识库条目"""

    entry_id: str
    entry_type: str
    name: str
    description: str = ""

    backend_ids: List[str] = field(default_factory=list)
    backend_features: Dict[str, Any] = field(default_factory=dict)

    hardware_ids: List[str] = field(default_factory=list)
    hardware_features: Dict[str, Any] = field(default_factory=dict)

    graph_pattern: Dict[str, Any] = field(default_factory=dict)
    node_patterns: List[str] = field(default_factory=list)

    operator_patterns: List[str] = field(default_factory=list)
    optimization_type: str = ""

    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 10

    performance_data: Dict[str, float] = field(default_factory=dict)
    code_reference: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedKnowledgeStorage:
    """统一知识库存储"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.path.join(cgc_output_dir(), "unified_knowledge.db")
        self._init_database()
        self._load_default_knowledge()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT UNIQUE,
                entry_type TEXT,
                name TEXT,
                description TEXT,
                backend_ids TEXT,
                backend_features TEXT,
                hardware_ids TEXT,
                hardware_features TEXT,
                graph_pattern TEXT,
                node_patterns TEXT,
                operator_patterns TEXT,
                optimization_type TEXT,
                conditions TEXT,
                actions TEXT,
                priority INTEGER,
                performance_data TEXT,
                code_reference TEXT,
                metadata TEXT,
                created_at TEXT
            )
            """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_type ON knowledge_entries(entry_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backend_ids ON knowledge_entries(backend_ids)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hardware_ids ON knowledge_entries(hardware_ids)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON knowledge_entries(priority)")

        conn.commit()
        conn.close()

    def save_entry(self, entry: KnowledgeEntry):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO knowledge_entries (
                entry_id, entry_type, name, description,
                backend_ids, backend_features,
                hardware_ids, hardware_features,
                graph_pattern, node_patterns,
                operator_patterns, optimization_type,
                conditions, actions, priority,
                performance_data, code_reference, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.entry_type,
                entry.name,
                entry.description,
                json.dumps(entry.backend_ids),
                json.dumps(entry.backend_features),
                json.dumps(entry.hardware_ids),
                json.dumps(entry.hardware_features),
                json.dumps(entry.graph_pattern),
                json.dumps(entry.node_patterns),
                json.dumps(entry.operator_patterns),
                entry.optimization_type,
                json.dumps(entry.conditions),
                json.dumps(entry.actions),
                entry.priority,
                json.dumps(entry.performance_data),
                entry.code_reference,
                json.dumps(entry.metadata),
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM knowledge_entries WHERE entry_id = ?", (entry_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return KnowledgeEntry(
            entry_id=row[1],
            entry_type=row[2],
            name=row[3],
            description=row[4],
            backend_ids=json.loads(row[5]),
            backend_features=json.loads(row[6]),
            hardware_ids=json.loads(row[7]),
            hardware_features=json.loads(row[8]),
            graph_pattern=json.loads(row[9]),
            node_patterns=json.loads(row[10]),
            operator_patterns=json.loads(row[11]),
            optimization_type=row[12],
            conditions=json.loads(row[13]),
            actions=json.loads(row[14]),
            priority=row[15],
            performance_data=json.loads(row[16]),
            code_reference=row[17],
            metadata=json.loads(row[18]),
        )

    def find_entries(
        self, entry_type: Optional[str] = None, backend_id: Optional[str] = None, hardware_id: Optional[str] = None
    ) -> List[KnowledgeEntry]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM knowledge_entries WHERE 1=1"
        params: List[Any] = []

        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        entries: List[KnowledgeEntry] = []
        for row in rows:
            entries.append(
                KnowledgeEntry(
                    entry_id=row[1],
                    entry_type=row[2],
                    name=row[3],
                    description=row[4],
                    backend_ids=json.loads(row[5]),
                    backend_features=json.loads(row[6]),
                    hardware_ids=json.loads(row[7]),
                    hardware_features=json.loads(row[8]),
                    graph_pattern=json.loads(row[9]),
                    node_patterns=json.loads(row[10]),
                    operator_patterns=json.loads(row[11]),
                    optimization_type=row[12],
                    conditions=json.loads(row[13]),
                    actions=json.loads(row[14]),
                    priority=row[15],
                    performance_data=json.loads(row[16]),
                    code_reference=row[17],
                    metadata=json.loads(row[18]),
                )
            )

        return entries

    def _load_default_knowledge(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
        if cursor.fetchone()[0] > 0:
            conn.close()
            print("✅ 统一知识库已有数据，跳过初始化")
            return
        conn.close()

        self.save_entry(
            KnowledgeEntry(
                entry_id="backend-vllm",
                entry_type="backend",
                name="vLLM",
                description="高性能LLM推理引擎",
                backend_ids=["vllm"],
                backend_features={
                    "supports_flash_attention": True,
                    "supports_paged_attention": True,
                    "supports_cuda_graph": True,
                    "supports_tensor_parallel": True,
                    "supports_pipeline_parallel": True,
                },
                performance_data={"throughput": 250, "latency": 8},
            )
        )

        self.save_entry(
            KnowledgeEntry(
                entry_id="backend-mlx",
                entry_type="backend",
                name="MLX",
                description="Apple Metal ML框架",
                backend_ids=["mlx"],
                backend_features={
                    "supports_flash_attention": True,
                    "supports_unified_memory": True,
                    "supports_mps_graph": True,
                    "supports_metal_tensor_parallel": True,
                },
                performance_data={"throughput": 100, "latency": 12},
            )
        )

        self.save_entry(
            KnowledgeEntry(
                entry_id="hardware-nvidia-rtx-5090",
                entry_type="hardware",
                name="NVIDIA RTX 5090",
                description="NVIDIA旗舰GPU",
                hardware_ids=["nvidia-rtx-5090"],
                hardware_features={
                    "memory_gb": 32,
                    "compute_capability": "9.0",
                    "supports_gds": True,
                    "supports_spdk": True,
                    "supports_cuda_graph": True,
                    "supports_dflash": True,
                },
                performance_data={"fp16_tflops": 1500, "hbm_bandwidth": 1200},
            )
        )

        self.save_entry(
            KnowledgeEntry(
                entry_id="hardware-apple-m4-ultra",
                entry_type="hardware",
                name="Apple M4 Ultra",
                description="Apple Silicon旗舰芯片",
                hardware_ids=["apple-m4-ultra"],
                hardware_features={
                    "memory_gb": 96,
                    "unified_memory": True,
                    "supports_mps_graph": True,
                    "supports_metal_tensor_parallel": True,
                    "supports_dflash": True,
                },
                performance_data={"fp16_tflops": 800, "unified_memory_bandwidth": 500},
            )
        )

        self.save_entry(
            KnowledgeEntry(
                entry_id="graph-attention-pattern",
                entry_type="graph_pattern",
                name="Attention图模式",
                description="Transformer注意力机制的计算图模式",
                graph_pattern={
                    "nodes": ["Q_proj", "K_proj", "V_proj", "attention", "output_proj"],
                    "edges": [
                        ("Q_proj", "attention"),
                        ("K_proj", "attention"),
                        ("V_proj", "attention"),
                        ("attention", "output_proj"),
                    ],
                    "compute_type": "compute_bound",
                },
                node_patterns=["query", "key", "value", "attention_score", "output"],
                operator_patterns=["linear", "attention", "softmax"],
            )
        )

        self.save_entry(
            KnowledgeEntry(
                entry_id="optimization-flash-attention",
                entry_type="pattern_optimization",
                name="Flash Attention优化",
                description="针对注意力算子的Flash Attention优化",
                backend_ids=["vllm", "mlx"],
                hardware_ids=["nvidia-rtx-5090", "apple-m4-ultra"],
                graph_pattern={"pattern_type": "attention", "optimization_target": "memory_efficiency"},
                operator_patterns=["attention"],
                optimization_type="flash_attention",
                performance_data={"speedup": 3.5, "memory_savings": 0.5},
                code_reference="cgc_engine/cuda/flash_attention.py",
            )
        )

        self.save_entry(
            KnowledgeEntry(
                entry_id="optimization-moe",
                entry_type="pattern_optimization",
                name="MoE优化",
                description="针对MoE算子的优化策略",
                backend_ids=["vllm"],
                hardware_ids=["nvidia-rtx-5090"],
                graph_pattern={"pattern_type": "moe", "optimization_target": "expert_efficiency"},
                operator_patterns=["moe_gate", "expert_forward"],
                optimization_type="moe",
                performance_data={"speedup": 2.0, "memory_savings": 0.3},
                code_reference="cgc_engine/cgc/moe_optimizer.py",
            )
        )

        self.save_entry(
            KnowledgeEntry(
                entry_id="strategy-dflash-dflash-hybrid",
                entry_type="strategy",
                name="DFlash-DFlash端云一体",
                description="双端DFlash配置：云端DFlash Prefill + 端侧DFlash Decode",
                backend_ids=["vllm", "mlx"],
                hardware_ids=["nvidia-rtx-5090", "apple-m4-ultra"],
                conditions={
                    "requests_per_second": {"$gt": 30},
                    "prefill_seq_len": {"$gt": 256},
                    "cloud_has_dflash": True,
                    "edge_has_dflash": True,
                },
                actions=[
                    {"action": "select_backend", "cloud_backend": "vllm", "edge_backend": "mlx"},
                    {"action": "set_role", "cloud_role": "prefill", "edge_role": "decode"},
                    {
                        "action": "optimization",
                        "cloud": ["dflash", "tp=2", "cuda_graph", "spdk"],
                        "edge": ["dflash", "mtp=2", "mps_graph", "unified_memory"],
                    },
                ],
                priority=1,
                performance_data={"speedup": 3.8, "latency_reduction": 0.6},
                optimization_type="hybrid",
            )
        )

        print("✅ 统一知识库初始化完成")


def test_unified_knowledge():
    print("=" * 100)
    print("📚 统一知识库测试")
    print("=" * 100)

    knowledge = UnifiedKnowledgeStorage()

    print("\n🔧 后端感知知识")
    print("-" * 50)
    backends = knowledge.find_entries(entry_type="backend")
    for b in backends:
        print(f"ID: {b.entry_id}")
        print(f"  名称: {b.name}")
        print(f"  特性: {b.backend_features}")
        print(f"  性能: {b.performance_data}")
        print()

    print("\n💻 硬件感知知识")
    print("-" * 50)
    hardwares = knowledge.find_entries(entry_type="hardware")
    for h in hardwares:
        print(f"ID: {h.entry_id}")
        print(f"  名称: {h.name}")
        print(f"  特性: {h.hardware_features}")
        print(f"  性能: {h.performance_data}")
        print()

    print("\n⚡ 算子模式感知知识")
    print("-" * 50)
    optimizations = knowledge.find_entries(entry_type="pattern_optimization")
    for o in optimizations:
        print(f"ID: {o.entry_id}")
        print(f"  名称: {o.name}")
        print(f"  优化类型: {o.optimization_type}")
        print(f"  适用后端: {o.backend_ids}")
        print(f"  适用硬件: {o.hardware_ids}")
        print(f"  性能提升: {o.performance_data}")
        print(f"  代码路径: {o.code_reference}")
        print()

    print("\n🎯 策略知识")
    print("-" * 50)
    strategies = knowledge.find_entries(entry_type="strategy")
    for s in strategies:
        print(f"ID: {s.entry_id}")
        print(f"  名称: {s.name}")
        print(f"  优先级: {s.priority}")
        print(f"  条件: {s.conditions}")
        print(f"  动作: {s.actions}")
        print(f"  性能: {s.performance_data}")
        print()

    print("=" * 100)
    print("✅ 统一知识库测试完成")
    print("=" * 100)


if __name__ == "__main__":
    test_unified_knowledge()
