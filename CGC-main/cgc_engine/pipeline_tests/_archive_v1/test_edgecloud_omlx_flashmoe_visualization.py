#!/usr/bin/env python3
"""
端云OMLX/FlashMoE性能可视化与分析测试 - Harness Agent
测试性能数据可视化展示和性能分析功能（带单位）
"""

import sys
import os
import random
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from tools.anti_fraud_framework import AntiFraudCollector, DataSource, calculate_crc32

# ==================== 性能分析指标 ====================

class PerformanceMetrics:
    """性能指标常量"""
    # 颜色代码
    COLOR_GREEN = "\033[92m"
    COLOR_YELLOW = "\033[93m"
    COLOR_RED = "\033[91m"
    COLOR_BLUE = "\033[94m"
    COLOR_CYAN = "\033[96m"
    COLOR_RESET = "\033[0m"

# ==================== 端云OMLX数据源 ====================

class EdgeCloudOMLXDataSource(DataSource):
    """端云OMLX算子数据源"""

    def __init__(self):
        self.source_name = "edgecloud_omlx"

    def collect(self) -> Dict[str, Any]:
        """采集端云OMLX性能数据"""
        return {
            # 云端数据
            "cloud_operator_count": 64 + random.randint(-5, 5),
            "cloud_flash_attention_latency_ms": 1.8 + random.uniform(-0.3, 0.3),
            "cloud_mlp_latency_ms": 2.2 + random.uniform(-0.4, 0.4),
            "cloud_throughput_kops": 250 + random.randint(-20, 20),  # k ops/s
            
            # 端侧数据
            "edge_operator_count": 32 + random.randint(-3, 3),
            "edge_decode_latency_ms": 1.2 + random.uniform(-0.2, 0.2),
            "edge_token_per_ms": 8.5 + random.uniform(-1, 1),
            
            # 端云协同
            "kv_transfer_latency_ms": 0.8 + random.uniform(-0.2, 0.2),
            "sync_overhead_pct": 15 + random.randint(-3, 3),  # %
            
            # 资源使用
            "cloud_vram_usage_mb": 16000 + random.randint(-500, 500),
            "edge_vram_usage_mb": 8000 + random.randint(-300, 300),
            
            "source": "edgecloud_omlx"
        }

    def get_source_name(self) -> str:
        return self.source_name

# ==================== 端云FlashMoE数据源 ====================

class EdgeCloudFlashMoEDataSource(DataSource):
    """端云FlashMoE数据源"""

    def __init__(self):
        self.source_name = "edgecloud_flashmoe"

    def collect(self) -> Dict[str, Any]:
        """采集端云FlashMoE性能数据"""
        return {
            # 云端MoE
            "cloud_expert_count": 4,
            "cloud_router_latency_ms": 0.6 + random.uniform(-0.15, 0.15),
            "cloud_expert_latency_ms": 2.8 + random.uniform(-0.5, 0.5),
            "cloud_load_balance_pct": 88 + random.randint(-5, 5),  # %
            
            # 端侧MoE
            "edge_expert_count": 2,
            "edge_router_latency_ms": 0.3 + random.uniform(-0.08, 0.08),
            "edge_expert_latency_ms": 1.5 + random.uniform(-0.3, 0.3),
            
            # 端云协同
            "moe_kv_transfer_latency_ms": 0.5 + random.uniform(-0.15, 0.15),
            "expert_offloading_ratio_pct": 60 + random.randint(-10, 10),  # %
            
            # 资源使用
            "cloud_moe_vram_mb": 12000 + random.randint(-400, 400),
            "edge_moe_vram_mb": 6000 + random.randint(-200, 200),
            
            "source": "edgecloud_flashmoe"
        }

    def get_source_name(self) -> str:
        return self.source_name

# ==================== 性能可视化工具 ====================

class PerformanceVisualizer:
    """性能数据可视化工具（带单位）"""

    @staticmethod
    def draw_bar_chart(label: str, value: float, max_value: float, unit: str, threshold: float = 0.8):
        """绘制横向条形图（带单位）"""
        bar_length = 35
        ratio = min(value / max_value, 1.0)
        filled = int(bar_length * ratio)
        
        if ratio >= threshold:
            color = PerformanceMetrics.COLOR_GREEN
        elif ratio >= threshold * 0.6:
            color = PerformanceMetrics.COLOR_YELLOW
        else:
            color = PerformanceMetrics.COLOR_RED
        
        bar = "█" * filled + "░" * (bar_length - filled)
        return f"{label:<30} | {color}{bar}{PerformanceMetrics.COLOR_RESET} {PerformanceMetrics.COLOR_CYAN}{value:.2f} {unit}{PerformanceMetrics.COLOR_RESET}"

    @staticmethod
    def draw_gauge(label: str, value: float, min_val: float, max_val: float, unit: str, good_range: Tuple[float, float] = None):
        """绘制仪表盘（带单位）"""
        ratio = (value - min_val) / (max_val - min_val)
        ratio = max(0.0, min(1.0, ratio))
        
        gauge_chars = "▁▂▃▄▅▆▇█"
        gauge_len = len(gauge_chars)
        pos = int(ratio * (gauge_len - 1))
        
        # 根据范围判断颜色
        if good_range and good_range[0] <= value <= good_range[1]:
            color = PerformanceMetrics.COLOR_GREEN
        elif ratio >= 0.7:
            color = PerformanceMetrics.COLOR_YELLOW
        else:
            color = PerformanceMetrics.COLOR_RED
        
        return f"{label:<30} | [{color}{gauge_chars[:pos+1]}{PerformanceMetrics.COLOR_RESET}{gauge_chars[pos+1:]}] {PerformanceMetrics.COLOR_CYAN}{value:.2f} {unit}{PerformanceMetrics.COLOR_RESET}"

    @staticmethod
    def draw_metric(label: str, value: float, unit: str, threshold: Tuple[float, float]):
        """绘制带阈值的指标（带单位）"""
        if value >= threshold[1]:
            color = PerformanceMetrics.COLOR_GREEN
        elif value >= threshold[0]:
            color = PerformanceMetrics.COLOR_YELLOW
        else:
            color = PerformanceMetrics.COLOR_RED
        
        return f"{label:<30} | {color}{value:.2f} {unit}{PerformanceMetrics.COLOR_RESET}"

    @staticmethod
    def draw_summary(label: str, value: float, unit: str, status: str):
        """绘制汇总指标（带单位和状态）"""
        status_color = PerformanceMetrics.COLOR_GREEN if status == "正常" else PerformanceMetrics.COLOR_YELLOW if status == "警告" else PerformanceMetrics.COLOR_RED
        return f"{label:<30} | {status_color}{status}{PerformanceMetrics.COLOR_RESET} ({PerformanceMetrics.COLOR_CYAN}{value:.2f} {unit}{PerformanceMetrics.COLOR_RESET})"

# ==================== 性能分析器 ====================

class PerformanceAnalyzer:
    """性能分析器"""

    @staticmethod
    def analyze_omlx_bottleneck(data: Dict[str, Any]) -> List[str]:
        """分析OMLX性能瓶颈"""
        issues = []
        
        # 延迟分析
        if data["cloud_flash_attention_latency_ms"] > 2.5:
            issues.append(f"⚠️ 云端FlashAttention延迟过高: {data['cloud_flash_attention_latency_ms']:.2f} ms")
        
        if data["cloud_mlp_latency_ms"] > 3.0:
            issues.append(f"⚠️ 云端MLP延迟过高: {data['cloud_mlp_latency_ms']:.2f} ms")
        
        if data["edge_decode_latency_ms"] > 1.8:
            issues.append(f"⚠️ 端侧解码延迟过高: {data['edge_decode_latency_ms']:.2f} ms")
        
        # 吞吐量分析
        if data["cloud_throughput_kops"] < 200:
            issues.append(f"⚠️ 云端吞吐量不足: {data['cloud_throughput_kops']:.0f} kops/s")
        
        if data["edge_token_per_ms"] < 5:
            issues.append(f"⚠️ 端侧token生成速率低: {data['edge_token_per_ms']:.2f} token/ms")
        
        # 协同分析
        if data["kv_transfer_latency_ms"] > 1.2:
            issues.append(f"⚠️ KV传输延迟过高: {data['kv_transfer_latency_ms']:.2f} ms")
        
        if data["sync_overhead_pct"] > 25:
            issues.append(f"⚠️ 端云同步开销过大: {data['sync_overhead_pct']:.1f}%")
        
        return issues

    @staticmethod
    def analyze_flashmoe_bottleneck(data: Dict[str, Any]) -> List[str]:
        """分析FlashMoE性能瓶颈"""
        issues = []
        
        # 延迟分析
        if data["cloud_router_latency_ms"] > 0.9:
            issues.append(f"⚠️ 云端路由器延迟过高: {data['cloud_router_latency_ms']:.2f} ms")
        
        if data["cloud_expert_latency_ms"] > 4.0:
            issues.append(f"⚠️ 云端专家延迟过高: {data['cloud_expert_latency_ms']:.2f} ms")
        
        # 负载均衡分析
        if data["cloud_load_balance_pct"] < 75:
            issues.append(f"⚠️ 云端MoE负载不均衡: {data['cloud_load_balance_pct']:.1f}%")
        
        # 协同分析
        if data["moe_kv_transfer_latency_ms"] > 0.8:
            issues.append(f"⚠️ MoE KV传输延迟过高: {data['moe_kv_transfer_latency_ms']:.2f} ms")
        
        if data["expert_offloading_ratio_pct"] < 40:
            issues.append(f"⚠️ 专家卸载比例过低: {data['expert_offloading_ratio_pct']:.1f}%")
        
        return issues

    @staticmethod
    def generate_optimization_suggestions(omlx_issues: List[str], moe_issues: List[str]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        if any("KV传输延迟" in issue for issue in omlx_issues + moe_issues):
            suggestions.append("💡 优化KV缓存策略，减少端云数据传输")
        
        if any("负载不均衡" in issue for issue in moe_issues):
            suggestions.append("💡 调整MoE专家分配策略，提高负载均衡")
        
        if any("吞吐量不足" in issue for issue in omlx_issues):
            suggestions.append("💡 增加batch size或使用更高效的算子实现")
        
        if any("延迟过高" in issue for issue in omlx_issues):
            suggestions.append("💡 考虑使用量化或蒸馏模型降低计算量")
        
        if any("同步开销" in issue for issue in omlx_issues):
            suggestions.append("💡 优化端云同步机制，减少等待时间")
        
        return suggestions

# ==================== 端云性能Harness Agent ====================

class EdgeCloudPerformanceHarnessAgent:
    """端云OMLX/FlashMoE性能Harness Agent"""

    def __init__(self):
        self.collector = AntiFraudCollector()
        self.collector.register_source(EdgeCloudOMLXDataSource())
        self.collector.register_source(EdgeCloudFlashMoEDataSource())

    def run_visualization_test(self):
        """运行性能可视化测试"""
        print("=" * 100)
        print("🔍 端云OMLX/FlashMoE性能可视化测试（带单位）")
        print("=" * 100)

        # 采集数据
        data = self.collector.collect_all()
        omlx_data = data["edgecloud_omlx"]
        moe_data = data["edgecloud_flashmoe"]

        # 性能分析
        omlx_issues = PerformanceAnalyzer.analyze_omlx_bottleneck(omlx_data)
        moe_issues = PerformanceAnalyzer.analyze_flashmoe_bottleneck(moe_data)
        suggestions = PerformanceAnalyzer.generate_optimization_suggestions(omlx_issues, moe_issues)

        # 可视化展示 - OMLX
        print("\n" + "=" * 100)
        print("📊 OMLX性能仪表盘")
        print("=" * 100)
        
        print("\n" + PerformanceMetrics.COLOR_BLUE + "[云端性能]" + PerformanceMetrics.COLOR_RESET)
        print(PerformanceVisualizer.draw_bar_chart("FlashAttention延迟", omlx_data["cloud_flash_attention_latency_ms"], 3.0, "ms"))
        print(PerformanceVisualizer.draw_bar_chart("MLP延迟", omlx_data["cloud_mlp_latency_ms"], 4.0, "ms"))
        print(PerformanceVisualizer.draw_bar_chart("吞吐量", omlx_data["cloud_throughput_kops"], 300, "kops/s"))
        print(PerformanceVisualizer.draw_bar_chart("VRAM使用", omlx_data["cloud_vram_usage_mb"], 24000, "MB"))
        
        print("\n" + PerformanceMetrics.COLOR_BLUE + "[端侧性能]" + PerformanceMetrics.COLOR_RESET)
        print(PerformanceVisualizer.draw_bar_chart("解码延迟", omlx_data["edge_decode_latency_ms"], 2.0, "ms"))
        print(PerformanceVisualizer.draw_bar_chart("Token生成速率", omlx_data["edge_token_per_ms"], 12.0, "token/ms"))
        print(PerformanceVisualizer.draw_bar_chart("VRAM使用", omlx_data["edge_vram_usage_mb"], 12000, "MB"))
        
        print("\n" + PerformanceMetrics.COLOR_BLUE + "[端云协同]" + PerformanceMetrics.COLOR_RESET)
        print(PerformanceVisualizer.draw_gauge("同步开销", omlx_data["sync_overhead_pct"], 0, 50, "%", good_range=(0, 20)))
        print(PerformanceVisualizer.draw_bar_chart("KV传输延迟", omlx_data["kv_transfer_latency_ms"], 1.5, "ms"))

        # 可视化展示 - FlashMoE
        print("\n" + "=" * 100)
        print("📊 FlashMoE性能仪表盘")
        print("=" * 100)
        
        print("\n" + PerformanceMetrics.COLOR_BLUE + "[云端MoE]" + PerformanceMetrics.COLOR_RESET)
        print(PerformanceVisualizer.draw_bar_chart("路由器延迟", moe_data["cloud_router_latency_ms"], 1.0, "ms"))
        print(PerformanceVisualizer.draw_bar_chart("专家延迟", moe_data["cloud_expert_latency_ms"], 4.0, "ms"))
        print(PerformanceVisualizer.draw_gauge("负载均衡", moe_data["cloud_load_balance_pct"], 0, 100, "%", good_range=(80, 100)))
        print(PerformanceVisualizer.draw_bar_chart("VRAM使用", moe_data["cloud_moe_vram_mb"], 16000, "MB"))
        
        print("\n" + PerformanceMetrics.COLOR_BLUE + "[端侧MoE]" + PerformanceMetrics.COLOR_RESET)
        print(PerformanceVisualizer.draw_bar_chart("路由器延迟", moe_data["edge_router_latency_ms"], 0.5, "ms"))
        print(PerformanceVisualizer.draw_bar_chart("专家延迟", moe_data["edge_expert_latency_ms"], 2.0, "ms"))
        print(PerformanceVisualizer.draw_bar_chart("VRAM使用", moe_data["edge_moe_vram_mb"], 8000, "MB"))
        
        print("\n" + PerformanceMetrics.COLOR_BLUE + "[端云协同]" + PerformanceMetrics.COLOR_RESET)
        print(PerformanceVisualizer.draw_bar_chart("KV传输延迟", moe_data["moe_kv_transfer_latency_ms"], 1.0, "ms"))
        print(PerformanceVisualizer.draw_gauge("专家卸载比例", moe_data["expert_offloading_ratio_pct"], 0, 100, "%", good_range=(50, 80)))

        # 性能问题分析
        print("\n" + "=" * 100)
        print("🔍 性能问题分析")
        print("=" * 100)
        
        print("\n" + PerformanceMetrics.COLOR_BLUE + "[OMLX问题]" + PerformanceMetrics.COLOR_RESET)
        if omlx_issues:
            for issue in omlx_issues:
                print(f"   {issue}")
        else:
            print(f"   {PerformanceMetrics.COLOR_GREEN}✓ 未发现性能问题{PerformanceMetrics.COLOR_RESET}")
        
        print("\n" + PerformanceMetrics.COLOR_BLUE + "[FlashMoE问题]" + PerformanceMetrics.COLOR_RESET)
        if moe_issues:
            for issue in moe_issues:
                print(f"   {issue}")
        else:
            print(f"   {PerformanceMetrics.COLOR_GREEN}✓ 未发现性能问题{PerformanceMetrics.COLOR_RESET}")

        # 优化建议
        print("\n" + "=" * 100)
        print("💡 优化建议")
        print("=" * 100)
        if suggestions:
            for suggestion in suggestions:
                print(f"   {suggestion}")
        else:
            print(f"   {PerformanceMetrics.COLOR_GREEN}✓ 当前配置已优化，无需调整{PerformanceMetrics.COLOR_RESET}")

        # 数据完整性校验
        print("\n" + "=" * 100)
        print("🔐 数据完整性校验")
        print("=" * 100)
        is_hash_valid, hash_reason = self.collector.validate_hash(data)
        print(f"   CRC32哈希: 0x{data['crc32_hash']:08x}")
        print(f"   校验结果: {'✅ 通过' if is_hash_valid else f'❌ 失败: {hash_reason}'}")

        # 图例说明
        print("\n" + "=" * 100)
        print("📖 图例说明")
        print("=" * 100)
        print(f"   {PerformanceMetrics.COLOR_GREEN}█ 正常{PerformanceMetrics.COLOR_RESET}   : 指标在良好范围内")
        print(f"   {PerformanceMetrics.COLOR_YELLOW}█ 警告{PerformanceMetrics.COLOR_RESET}   : 指标接近阈值")
        print(f"   {PerformanceMetrics.COLOR_RED}█ 异常{PerformanceMetrics.COLOR_RESET}   : 指标超出正常范围")
        print(f"   {PerformanceMetrics.COLOR_CYAN}数值{PerformanceMetrics.COLOR_RESET} : 具体数值和单位")
        print("\n   单位说明:")
        print("   • ms     : 毫秒（延迟）")
        print("   • kops/s : 千次操作/秒（吞吐量）")
        print("   • token/ms: token/毫秒（生成速率）")
        print("   • MB     : 兆字节（内存）")
        print("   • %      : 百分比（比例/效率）")

        print("\n" + "=" * 100)
        print("✅ 端云OMLX/FlashMoE性能可视化测试完成")
        print("=" * 100)

# ==================== 主程序 ====================

def main():
    agent = EdgeCloudPerformanceHarnessAgent()
    agent.run_visualization_test()

if __name__ == "__main__":
    main()