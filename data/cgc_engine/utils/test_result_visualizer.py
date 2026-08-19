# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
测试结果可视化工具

提供:
1. 性能指标图表生成
2. HTML 报告生成
3. 对比分析可视化
4. 支持多种图表类型
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from cgc_engine.utils.envs import cgc_output_dir

@dataclass
class ChartConfig:
    """图表配置"""
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    chart_type: str = "bar"  # bar, line, pie, scatter
    colors: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ])
    show_grid: bool = True
    show_values: bool = True


class TestResultVisualizer:
    """测试结果可视化工具"""
    
    def __init__(self, storage_path: str | None = None):
        self.storage_path = storage_path or os.path.join(cgc_output_dir(), "test_results.db")
        self._load_storage()
    
    def _load_storage(self):
        """延迟加载存储模块"""
        from cgc_engine.utils.test_result_storage import TestResultStorage
        self.storage = TestResultStorage(self.storage_path)
    
    def generate_html_report(
        self,
        module_name: Optional[str] = None,
        output_file: str | None = None
    ):
        """生成 HTML 报告"""
        output_file = output_file or os.path.join(cgc_output_dir(), "reports", "test_report.html")
        records = self.storage.get_records_by_module(module_name) if module_name else self.storage.get_all_records()
        
        if not records:
            print("❌ 没有找到测试记录")
            return
        
        html_content = self._generate_report_html(records)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ HTML 报告已生成: {output_file}")
    
    def _generate_report_html(self, records: List) -> str:
        """生成报告 HTML 内容"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        modules = sorted(set(r.module_name for r in records))
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Harness Agent 测试报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        .header p {{ color: #7f8c8d; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stat-card .label {{ color: #7f8c8d; font-size: 14px; }}
        .stat-card .value {{ color: #2c3e50; font-size: 28px; font-weight: bold; }}
        .chart-section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 25px; }}
        .chart-section h2 {{ color: #2c3e50; margin-bottom: 20px; font-size: 18px; }}
        .chart-container {{ height: 350px; }}
        .record-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .record-table th, .record-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        .record-table th {{ background: #3498db; color: white; }}
        .record-table tr:hover {{ background: #f8f9fa; }}
        .success {{ color: #27ae60; }}
        .failed {{ color: #e74c3c; }}
        .tabs {{ display: flex; gap: 10px; margin-bottom: 20px; }}
        .tab {{ padding: 10px 20px; background: white; border: none; border-radius: 5px; cursor: pointer; }}
        .tab.active {{ background: #3498db; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Harness Agent 测试报告</h1>
            <p>生成时间: {timestamp} | 模块: {', '.join(modules) if modules else '全部'}</p>
        </div>
        
        {self._generate_stats_section(records)}
        {self._generate_charts_section(records)}
        {self._generate_records_table(records)}
    </div>
    
    <script>
        {self._generate_chart_scripts(records)}
    </script>
</body>
</html>
        """
        return html
    
    def _generate_stats_section(self, records: List) -> str:
        """生成统计卡片区域"""
        total = len(records)
        success = sum(1 for r in records if r.success)
        avg_time = sum(r.total_time_ms for r in records) / total if total > 0 else 0
        avg_memory = sum(r.peak_memory_gb for r in records) / total if total > 0 else 0
        
        return f"""
<div class="stats-grid">
    <div class="stat-card">
        <div class="label">总测试次数</div>
        <div class="value">{total}</div>
    </div>
    <div class="stat-card">
        <div class="label">成功率</div>
        <div class="value">{(success/total*100):.1f}%</div>
    </div>
    <div class="stat-card">
        <div class="label">平均耗时</div>
        <div class="value">{avg_time:.2f} ms</div>
    </div>
    <div class="stat-card">
        <div class="label">平均内存峰值</div>
        <div class="value">{avg_memory:.2f} GB</div>
    </div>
</div>
        """
    
    def _generate_charts_section(self, records: List) -> str:
        """生成图表区域"""
        return f"""
<div class="chart-section">
    <h2>📊 性能对比分析</h2>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
        <div class="chart-container">
            <canvas id="timeChart"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="memoryChart"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="gflopsChart"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="ioChart"></canvas>
        </div>
    </div>
</div>
        """
    
    def _generate_chart_scripts(self, records: List) -> str:
        """生成图表 JavaScript"""
        labels = [f"{r.module_name} ({r.device})" for r in records]
        times = [r.total_time_ms for r in records]
        memories = [r.peak_memory_gb for r in records]
        gflops = [r.gflops for r in records]
        h2d = [r.h2d_bytes / (1024**2) for r in records]  # MB
        d2h = [r.d2h_bytes / (1024**2) for r in records]  # MB
        
        return f"""
// 延迟对比图
new Chart(document.getElementById('timeChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: '总耗时 (ms)',
            data: {json.dumps(times)},
            backgroundColor: '#3498db',
            borderColor: '#2980b9',
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: '毫秒' }} }} }}
    }}
}});

// 内存对比图
new Chart(document.getElementById('memoryChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: '内存峰值 (GB)',
            data: {json.dumps(memories)},
            backgroundColor: '#2ecc71',
            borderColor: '#27ae60',
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'GB' }} }} }}
    }}
}});

// GFLOPS 对比图
new Chart(document.getElementById('gflopsChart'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: 'GFLOPS',
            data: {json.dumps(gflops)},
            borderColor: '#e74c3c',
            backgroundColor: 'rgba(231,76,60,0.1)',
            fill: true,
            tension: 0.4
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{ y: {{ beginAtZero: true }} }}
    }}
}});

// IO 流量对比图
new Chart(document.getElementById('ioChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [
            {{
                label: 'H2D (MB)',
                data: {json.dumps(h2d)},
                backgroundColor: '#9b59b6',
                borderColor: '#8e44ad',
                borderWidth: 1
            }},
            {{
                label: 'D2H (MB)',
                data: {json.dumps(d2h)},
                backgroundColor: '#f39c12',
                borderColor: '#e67e22',
                borderWidth: 1
            }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'MB' }} }} }}
    }}
}});
        """
    
    def _generate_records_table(self, records: List) -> str:
        """生成测试记录表格"""
        rows = ""
        for r in records:
            status = '<span class="success">✓ 成功</span>' if r.success else '<span class="failed">✗ 失败</span>'
            rows += f"""
<tr>
    <td>{r.test_id[:20]}...</td>
    <td>{r.module_name}</td>
    <td>{r.device}</td>
    <td>{r.backend}</td>
    <td>{status}</td>
    <td>{r.total_time_ms:.2f}</td>
    <td>{r.peak_memory_gb:.2f}</td>
    <td>{r.gflops:.1f}</td>
    <td>{r.timestamp[:19]}</td>
</tr>
            """
        
        return f"""
<div class="chart-section">
    <h2>📋 测试记录详情</h2>
    <table class="record-table">
        <thead>
            <tr>
                <th>测试ID</th>
                <th>模块</th>
                <th>设备</th>
                <th>后端</th>
                <th>状态</th>
                <th>耗时(ms)</th>
                <th>内存(GB)</th>
                <th>GFLOPS</th>
                <th>时间</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</div>
        """
    
    def generate_ablation_chart(
        self,
        results: List,
        output_file: str = "ablation_chart.html"
    ):
        """生成消融实验图表"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>消融实验结果</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f5f7fa; padding: 30px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
        .chart-wrapper {{ background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .chart-container {{ height: 400px; }}
        .summary {{ margin-top: 30px; padding: 20px; background: #ecf0f1; border-radius: 10px; }}
        .highlight {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 消融实验结果对比</h1>
        <div class="chart-wrapper">
            <div class="chart-container">
                <canvas id="ablationChart"></canvas>
            </div>
        </div>
        <div class="summary">
            <h3>📊 关键发现</h3>
            {self._generate_ablation_summary(results)}
        </div>
    </div>
    
    <script>
        {self._generate_ablation_script(results)}
    </script>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✅ 消融实验图表已生成: {output_file}")
    
    def _generate_ablation_summary(self, results: List) -> str:
        """生成消融实验摘要"""
        baseline = results[0] if results else None
        if not baseline:
            return "<p>无数据</p>"
        
        best_result = min(results, key=lambda r: r.expert_loading_time_ms)
        
        summary = f"""
<ul>
    <li><strong>基线耗时:</strong> {baseline.expert_loading_time_ms:.2f} ms</li>
    <li><strong>最佳配置:</strong> {best_result.config_name} ({best_result.expert_loading_time_ms:.2f} ms)</li>
    <li><strong>最高加速比:</strong> <span class="highlight">{best_result.speedup_vs_baseline:.2f}x</span></li>
</ul>
        """
        return summary
    
    def _generate_ablation_script(self, results: List) -> str:
        """生成消融实验图表脚本"""
        labels = [r.config_name for r in results]
        times = [r.expert_loading_time_ms for r in results]
        kv_write = [r.kv_cache_write_throughput_mbs for r in results]
        kv_read = [r.kv_cache_read_throughput_mbs for r in results]
        
        return f"""
new Chart(document.getElementById('ablationChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [
            {{
                label: '专家加载时间 (ms)',
                data: {json.dumps(times)},
                backgroundColor: '#3498db',
                borderColor: '#2980b9',
                borderWidth: 1,
                yAxisID: 'y'
            }},
            {{
                label: 'KV写入 (MB/s)',
                data: {json.dumps(kv_write)},
                backgroundColor: '#2ecc71',
                borderColor: '#27ae60',
                borderWidth: 1,
                yAxisID: 'y1'
            }},
            {{
                label: 'KV读取 (MB/s)',
                data: {json.dumps(kv_read)},
                backgroundColor: '#e74c3c',
                borderColor: '#c0392b',
                borderWidth: 1,
                yAxisID: 'y1'
            }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ display: true, position: 'top' }}
        }},
        scales: {{
            y: {{
                type: 'linear',
                position: 'left',
                title: {{ display: true, text: '时间 (ms)' }}
            }},
            y1: {{
                type: 'linear',
                position: 'right',
                title: {{ display: true, text: '吞吐量 (MB/s)' }}
            }}
        }}
    }}
}});
        """
    
    def generate_architecture_profit_chart(
        self,
        results: List,
        output_file: str = "architecture_profit.html"
    ):
        """生成架构收益对比图表"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>架构收益分析</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 30px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 30px; }}
        h1 {{ text-align: center; color: white; margin-bottom: 30px; font-size: 28px; }}
        .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        .chart-container {{ height: 350px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; }}
        .metric-card .value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .metric-card .label {{ color: #7f8c8d; margin-top: 5px; }}
        .highlight {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏗️ 架构收益分析</h1>
        {self._generate_architecture_metrics(results)}
        <div class="card">
            <div class="chart-row">
                <div class="chart-container"><canvas id="latencyChart"></canvas></div>
                <div class="chart-container"><canvas id="throughputChart"></canvas></div>
            </div>
        </div>
        <div class="card">
            <div class="chart-row">
                <div class="chart-container"><canvas id="gpuUtilChart"></canvas></div>
                <div class="chart-container"><canvas id="componentChart"></canvas></div>
            </div>
        </div>
    </div>
    
    <script>
        {self._generate_architecture_scripts(results)}
    </script>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✅ 架构收益图表已生成: {output_file}")
    
    def _generate_architecture_metrics(self, results: List) -> str:
        """生成架构指标卡片"""
        if not results:
            return ""
        
        baseline = results[0]
        best = results[-1]
        
        return f"""
<div class="card" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
    <div class="metric-card">
        <div class="value">{baseline.end_to_end_latency_ms:.0f} ms</div>
        <div class="label">基线延迟</div>
    </div>
    <div class="metric-card">
        <div class="value highlight">{best.end_to_end_latency_ms:.0f} ms</div>
        <div class="label">优化后延迟</div>
    </div>
    <div class="metric-card">
        <div class="value highlight">x{best.latency_vs_baseline:.1f}</div>
        <div class="label">延迟降低</div>
    </div>
    <div class="metric-card">
        <div class="value">{best.gpu_utilization:.0f}%</div>
        <div class="label">GPU利用率</div>
    </div>
</div>
        """
    
    def _generate_architecture_scripts(self, results: List) -> str:
        """生成架构收益图表脚本"""
        labels = [r.config_name for r in results]
        latencies = [r.end_to_end_latency_ms for r in results]
        throughputs = [r.throughput_tokens_per_sec for r in results]
        gpu_utils = [r.gpu_utilization for r in results]
        prefill = [r.prefill_latency_ms for r in results]
        decode = [r.decode_latency_ms for r in results]
        kv = [r.kv_access_latency_ms for r in results]
        
        return f"""
// 延迟对比
new Chart(document.getElementById('latencyChart'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: '端到端延迟 (ms)',
            data: {json.dumps(latencies)},
            borderColor: '#e74c3c',
            backgroundColor: 'rgba(231,76,60,0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 6,
            pointBackgroundColor: '#e74c3c'
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: '毫秒' }} }} }}
    }}
}});

// 吞吐量对比
new Chart(document.getElementById('throughputChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: '吞吐量 (tok/s)',
            data: {json.dumps(throughputs)},
            backgroundColor: '#27ae60',
            borderColor: '#1e8449',
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'tokens/s' }} }} }}
    }}
}});

// GPU 利用率
new Chart(document.getElementById('gpuUtilChart'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: 'GPU 利用率 (%)',
            data: {json.dumps(gpu_utils)},
            borderColor: '#3498db',
            backgroundColor: 'rgba(52,152,219,0.1)',
            fill: true,
            tension: 0.3
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{ y: {{ min: 0, max: 100, title: {{ display: true, text: '百分比' }} }} }}
    }}
}});

// 组件分解
new Chart(document.getElementById('componentChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [
            {{ label: 'Prefill', data: {json.dumps(prefill)}, backgroundColor: '#9b59b6' }},
            {{ label: 'Decode', data: {json.dumps(decode)}, backgroundColor: '#f39c12' }},
            {{ label: 'KV访问', data: {json.dumps(kv)}, backgroundColor: '#1abc9c' }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: true }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: '毫秒' }} }}, stack: true }}
    }}
}});
        """
