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
CGC Tracking Visualization Dashboard

功能:
- CGC 命令执行追踪
- 性能指标可视化
- 实时监控面板
- 吞吐量/延迟统计

架构:
    CGC Executor → Trace Collector → WebSocket Server → Dashboard (HTML/JS)
"""

import json
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime
import uuid
import asyncio

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None


class OpcodeCategory(Enum):
    """操作码类别"""
    ATTENTION = "attention"
    LINEAR = "linear"
    NORM = "norm"
    ROPE = "rope"
    ACTIVATION = "activation"
    SAMPLING = "sampling"
    MEMORY = "memory"
    KDA = "kda"
    DISTRIBUTED = "distributed"
    QUANTIZATION = "quantization"
    OTHER = "other"


@dataclass
class CGCTraceEvent:
    """CGC 追踪事件"""
    event_id: str
    timestamp: float
    opcode: int
    opcode_name: str
    category: str
    duration_us: int
    input_shapes: Dict[str, str]
    output_shapes: Dict[str, str]
    params: Dict[str, Any]
    success: bool
    error_message: str = ""
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "timestamp_str": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3],
            "opcode": self.opcode,
            "opcode_name": self.opcode_name,
            "category": self.category,
            "duration_us": self.duration_us,
            "duration_ms": self.duration_us / 1000,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "params": self.params,
            "success": self.success,
            "error_message": self.error_message,
            "rank": self.rank,
        }


@dataclass
class CGCStats:
    """CGC 统计"""
    total_commands: int = 0
    successful_commands: int = 0
    failed_commands: int = 0
    total_duration_us: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    opcode_counts: Dict[str, int] = field(default_factory=dict)
    avg_duration_by_opcode: Dict[str, float] = field(default_factory=dict)
    throughput: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_commands": self.total_commands,
            "successful_commands": self.successful_commands,
            "failed_commands": self.failed_commands,
            "total_duration_us": self.total_duration_us,
            "avg_duration_ms": self.total_duration_us / max(self.total_commands, 1) / 1000,
            "category_counts": self.category_counts,
            "opcode_counts": self.opcode_counts,
            "avg_duration_by_opcode": self.avg_duration_by_opcode,
            "throughput": self.throughput,
        }


class CGCTracer:
    """
    CGC 追踪器

    收集 CGC 命令执行事件并生成统计
    """

    OPCODE_TO_NAME = {
        0x01: "WEIGHT_STAY",
        0x02: "LAYER_STREAM_LOAD",
        0x03: "LAYER_FORWARD",
        0x07: "ORTHO_BASIS_UPDATE",
        0x10: "ATTENTION_SDPA",
        0x11: "ATTENTION_KDA",
        0x12: "ATTENTION_PAGED",
        0x13: "ATTENTION_FLASH",
        0x20: "LINEAR_GEMM",
        0x21: "LINEAR_BIAS",
        0x22: "GEMM_BATCHED",
        0x30: "LAYER_NORM",
        0x31: "RMS_NORM",
        0x32: "GROUP_NORM",
        0x40: "ROPE",
        0x41: "ROPE_FUSED",
        0x42: "YARN_ROPE",
        0x50: "SILU",
        0x51: "GELU",
        0x52: "GELU_TANH",
        0x53: "RELU",
        0x54: "SIGMOID",
        0x60: "SOFTMAX",
        0x61: "LOG_SOFTMAX",
        0x62: "TOP_K",
        0x63: "TOP_P",
        0x64: "TEMPERATURE",
        0x70: "KV_CACHE_LOAD",
        0x71: "KV_CACHE_STORE",
        0x72: "KV_CACHE_UPDATE",
        0x73: "EMBEDDING_LOOKUP",
        0x80: "KDA_CHUNK",
        0x81: "KDA_PROJECT",
        0x82: "KDA_ORTHO_UPDATE",
        0x83: "KDA_BACKWARD",
        0x90: "ALL_REDUCE",
        0x91: "ALL_GATHER",
        0x92: "REDUCE_SCATTER",
        0xA0: "QUANTIZE_W8A16",
        0xA1: "QUANTIZE_W4A16",
        0xA2: "DEQUANTIZE",
        0xA3: "GPTQ_KERNEL",
        0xA4: "AWQ_KERNEL",
    }

    OPCODE_TO_CATEGORY = {
        0x01: OpcodeCategory.OTHER,
        0x02: OpcodeCategory.MEMORY,
        0x03: OpcodeCategory.OTHER,
        0x07: OpcodeCategory.KDA,
        0x10: OpcodeCategory.ATTENTION,
        0x11: OpcodeCategory.ATTENTION,
        0x12: OpcodeCategory.ATTENTION,
        0x13: OpcodeCategory.ATTENTION,
        0x20: OpcodeCategory.LINEAR,
        0x21: OpcodeCategory.LINEAR,
        0x22: OpcodeCategory.LINEAR,
        0x30: OpcodeCategory.NORM,
        0x31: OpcodeCategory.NORM,
        0x32: OpcodeCategory.NORM,
        0x40: OpcodeCategory.ROPE,
        0x41: OpcodeCategory.ROPE,
        0x42: OpcodeCategory.ROPE,
        0x50: OpcodeCategory.ACTIVATION,
        0x51: OpcodeCategory.ACTIVATION,
        0x52: OpcodeCategory.ACTIVATION,
        0x53: OpcodeCategory.ACTIVATION,
        0x54: OpcodeCategory.ACTIVATION,
        0x60: OpcodeCategory.SAMPLING,
        0x61: OpcodeCategory.SAMPLING,
        0x62: OpcodeCategory.SAMPLING,
        0x63: OpcodeCategory.SAMPLING,
        0x64: OpcodeCategory.SAMPLING,
        0x70: OpcodeCategory.MEMORY,
        0x71: OpcodeCategory.MEMORY,
        0x72: OpcodeCategory.MEMORY,
        0x73: OpcodeCategory.MEMORY,
        0x80: OpcodeCategory.KDA,
        0x81: OpcodeCategory.KDA,
        0x82: OpcodeCategory.KDA,
        0x83: OpcodeCategory.KDA,
        0x90: OpcodeCategory.DISTRIBUTED,
        0x91: OpcodeCategory.DISTRIBUTED,
        0x92: OpcodeCategory.DISTRIBUTED,
        0xA0: OpcodeCategory.QUANTIZATION,
        0xA1: OpcodeCategory.QUANTIZATION,
        0xA2: OpcodeCategory.QUANTIZATION,
        0xA3: OpcodeCategory.QUANTIZATION,
        0xA4: OpcodeCategory.QUANTIZATION,
    }

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.stats = CGCStats()
        self._lock = threading.RLock()
        self._start_time = time.time()

    def record(
        self,
        opcode: int,
        duration_us: int,
        input_shapes: Optional[Dict[str, str]] = None,
        output_shapes: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: str = "",
        rank: int = 0,
    ):
        """记录 CGC 命令执行"""
        event = CGCTraceEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            opcode=opcode,
            opcode_name=self.OPCODE_TO_NAME.get(opcode, f"UNKNOWN_0x{opcode:02X}"),
            category=self.OPCODE_TO_CATEGORY.get(opcode, OpcodeCategory.OTHER).value,
            duration_us=duration_us,
            input_shapes=input_shapes or {},
            output_shapes=output_shapes or {},
            params=params or {},
            success=success,
            error_message=error_message,
            rank=rank,
        )

        with self._lock:
            self.events.append(event)
            self._update_stats(event)

    def _update_stats(self, event: CGCTraceEvent):
        """更新统计"""
        self.stats.total_commands += 1

        if event.success:
            self.stats.successful_commands += 1
        else:
            self.stats.failed_commands += 1

        self.stats.total_duration_us += event.duration_us

        self.stats.category_counts[event.category] = \
            self.stats.category_counts.get(event.category, 0) + 1

        self.stats.opcode_counts[event.opcode_name] = \
            self.stats.opcode_counts.get(event.opcode_name, 0) + 1

        total_for_opcode = sum(
            e.duration_us for e in self.events
            if e.opcode == event.opcode
        )
        count_for_opcode = max(1, self.stats.opcode_counts.get(event.opcode_name, 1))
        self.stats.avg_duration_by_opcode[event.opcode_name] = \
            total_for_opcode / count_for_opcode / 1000

        elapsed = time.time() - self._start_time
        self.stats.throughput = self.stats.total_commands / max(elapsed, 0.001)

    def get_recent_events(self, n: int = 100) -> List[dict]:
        """获取最近的 n 个事件"""
        with self._lock:
            return [e.to_dict() for e in list(self.events)[-n:]]

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return self.stats.to_dict()

    def get_category_breakdown(self) -> dict:
        """获取类别分布"""
        with self._lock:
            total = max(self.stats.total_commands, 1)
            breakdown = {}
            for cat, count in self.stats.category_counts.items():
                breakdown[cat] = {
                    "count": count,
                    "percentage": count / total * 100,
                    "avg_duration_ms": sum(
                        e.duration_us for e in self.events if e.category == cat
                    ) / max(count, 1) / 1000,
                }
            return breakdown

    def reset(self):
        """重置追踪"""
        with self._lock:
            self.events.clear()
            self.stats = CGCStats()
            self._start_time = time.time()


class CGCDashboardServer:
    """
    CGC Dashboard Web 服务器

    提供实时监控面板
    """

    def __init__(self, tracer: CGCTracer, port: int = 8080):
        self.tracer = tracer
        self.port = port
        self.app = None
        self.runner = None
        self._websocket_clients: List = []

    async def _handle_api_stats(self, request):
        """API: 获取统计"""
        return web.json_response(self.tracer.get_stats())

    async def _handle_api_events(self, request):
        """API: 获取最近事件"""
        n = int(request.query.get("n", 100))
        return web.json_response(self.tracer.get_recent_events(n))

    async def _handle_api_categories(self, request):
        """API: 获取类别分布"""
        return web.json_response(self.tracer.get_category_breakdown())

    async def _handle_api_reset(self, request):
        """API: 重置"""
        self.tracer.reset()
        return web.json_response({"status": "reset"})

    async def _handle_dashboard(self, request):
        """Dashboard HTML"""
        return web.Response(
            text=self._get_dashboard_html(),
            content_type="text/html",
        )

    def _get_dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html>
<head>
    <title>CGC Command Tracking Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            padding: 20px;
        }
        h1 {
            color: #00ff88;
            margin-bottom: 20px;
            font-size: 24px;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: #1a1a3e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #2a2a4e;
        }
        .card h3 {
            color: #8888ff;
            font-size: 14px;
            margin-bottom: 15px;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #00ff88;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .chart-container { height: 200px; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #2a2a4e;
        }
        th { color: #8888ff; }
        .success { color: #00ff88; }
        .failed { color: #ff4444; }
        .opcode { color: #ffaa00; }
        #events-table { max-height: 300px; overflow-y: auto; }
        .refresh-btn {
            background: #8888ff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
        }
        .refresh-btn:hover { background: #6666dd; }
    </style>
</head>
<body>
    <h1>CGC Command Tracking Dashboard</h1>

    <div class="dashboard">
        <div class="card">
            <h3>Total Commands</h3>
            <div class="stat-value" id="total-commands">0</div>
            <div class="stat-label">commands executed</div>
        </div>
        <div class="card">
            <h3>Throughput</h3>
            <div class="stat-value" id="throughput">0</div>
            <div class="stat-label">commands/sec</div>
        </div>
        <div class="card">
            <h3>Avg Duration</h3>
            <div class="stat-value" id="avg-duration">0</div>
            <div class="stat-label">ms per command</div>
        </div>
        <div class="card">
            <h3>Success Rate</h3>
            <div class="stat-value" id="success-rate">100%</div>
            <div class="stat-label">successful</div>
        </div>
    </div>

    <div class="dashboard">
        <div class="card">
            <h3>Category Distribution</h3>
            <div class="chart-container">
                <canvas id="category-chart"></canvas>
            </div>
        </div>
        <div class="card">
            <h3>Command Timeline</h3>
            <div class="chart-container">
                <canvas id="timeline-chart"></canvas>
            </div>
        </div>
    </div>

    <div class="dashboard">
        <div class="card" style="grid-column: span 2;">
            <h3>Recent Commands</h3>
            <button class="refresh-btn" onclick="refreshData()">Refresh</button>
            <div id="events-table">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Opcode</th>
                            <th>Category</th>
                            <th>Duration</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="events-tbody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let categoryChart, timelineChart;

        async function fetchData() {
            const [stats, events, categories] = await Promise.all([
                fetch('/api/stats').then(r => r.json()),
                fetch('/api/events?n=50').then(r => r.json()),
                fetch('/api/categories').then(r => r.json())
            ]);
            return { stats, events, categories };
        }

        function updateStats(stats) {
            document.getElementById('total-commands').textContent = stats.total_commands;
            document.getElementById('throughput').textContent = stats.throughput.toFixed(2);
            document.getElementById('avg-duration').textContent = stats.avg_duration_ms.toFixed(3);
            const rate = stats.total_commands > 0
                ? (stats.successful_commands / stats.total_commands * 100).toFixed(1)
                : 100;
            document.getElementById('success-rate').textContent = rate + '%';
        }

        function updateCategoryChart(categories) {
            const labels = Object.keys(categories);
            const data = labels.map(l => categories[l].count);

            if (!categoryChart) {
                const ctx = document.getElementById('category-chart').getContext('2d');
                categoryChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: data,
                            backgroundColor: [
                                '#00ff88', '#8888ff', '#ffaa00', '#ff4444',
                                '#00ffff', '#ff00ff', '#ffff00', '#8844ff',
                                '#44ff88', '#ff8844'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { position: 'right' } }
                    }
                });
            } else {
                categoryChart.data.labels = labels;
                categoryChart.data.datasets[0].data = data;
                categoryChart.update();
            }
        }

        function updateTimelineChart(events) {
            const labels = events.map(e => e.timestamp_str);
            const data = events.map(e => e.duration_ms);

            if (!timelineChart) {
                const ctx = document.getElementById('timeline-chart').getContext('2d');
                timelineChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Duration (ms)',
                            data: data,
                            borderColor: '#00ff88',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { display: false },
                            y: { beginAtZero: true }
                        }
                    }
                });
            } else {
                timelineChart.data.labels = labels;
                timelineChart.data.datasets[0].data = data;
                timelineChart.update();
            }
        }

        function updateEventsTable(events) {
            const tbody = document.getElementById('events-tbody');
            tbody.innerHTML = events.slice(-20).reverse().map(e => `
                <tr>
                    <td>${e.timestamp_str}</td>
                    <td class="opcode">${e.opcode_name}</td>
                    <td>${e.category}</td>
                    <td>${e.duration_ms.toFixed(3)} ms</td>
                    <td class="${e.success ? 'success' : 'failed'}">${e.success ? 'OK' : 'FAIL'}</td>
                </tr>
            `).join('');
        }

        async function refreshData() {
            try {
                const { stats, events, categories } = await fetchData();
                updateStats(stats);
                updateCategoryChart(categories);
                updateTimelineChart(events);
                updateEventsTable(events);
            } catch (e) {
                console.error('Refresh failed:', e);
            }
        }

        setInterval(refreshData, 2000);
        refreshData();
    </script>
</body>
</html>"""

    async def start(self):
        """启动 Dashboard 服务器"""
        if not AIOHTTP_AVAILABLE:
            print("[Dashboard] aiohttp not available. Run: pip install aiohttp")
            return

        self.app = web.Application()
        self.app.router.add_get("/", self._handle_dashboard)
        self.app.router.add_get("/api/stats", self._handle_api_stats)
        self.app.router.add_get("/api/events", self._handle_api_events)
        self.app.router.add_get("/api/categories", self._handle_api_categories)
        self.app.router.add_post("/api/reset", self._handle_api_reset)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await site.start()

        print(f"[Dashboard] Started on http://0.0.0.0:{self.port}")

    async def stop(self):
        """停止 Dashboard 服务器"""
        if self.runner:
            await self.runner.cleanup()


_global_tracer: Optional[CGCTracer] = None
_dashboard_server: Optional[CGCDashboardServer] = None


def get_tracer() -> CGCTracer:
    """获取全局 tracer"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = CGCTracer()
    return _global_tracer


def record_cgc_command(
    opcode: int,
    duration_us: int,
    **kwargs
):
    """便捷函数: 记录 CGC 命令"""
    get_tracer().record(opcode, duration_us, **kwargs)


def get_dashboard_info() -> dict:
    """获取 Dashboard 信息"""
    tracer = get_tracer()
    return {
        "total_commands": tracer.get_stats().total_commands,
        "active": tracer.is_active(),
        "events_count": len(tracer.events),
    }


async def start_dashboard(port: int = 8080):
    """启动 Dashboard"""
    global _dashboard_server
    _dashboard_server = CGCDashboardServer(get_tracer(), port)
    await _dashboard_server.start()


if __name__ == "__main__":
    tracer = get_tracer()

    for i in range(100):
        import random
        tracer.record(
            opcode=random.choice([0x10, 0x11, 0x80, 0x20, 0x31, 0x50]),
            duration_us=random.randint(100, 5000),
            success=random.random() > 0.05,
        )

    print("Stats:", json.dumps(tracer.get_stats(), indent=2))
    print("\nCategories:", json.dumps(tracer.get_category_breakdown(), indent=2))
