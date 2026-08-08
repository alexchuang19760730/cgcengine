#!/usr/bin/env python3
"""
GLM API 延迟分析工具
分解延迟来源并提供优化建议
"""

import os
import time
import json
import requests
from typing import Dict, Any

# 设置环境变量
os.environ["GLM_API_KEY"] = "bcdebd9b768f40528603f07626be2493.VGi12nP3d3zf1gvh"

class LatencyAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("GLM_API_KEY")
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"
        
    def measure_network_latency(self) -> float:
        """测量网络 RTT 延迟"""
        print("正在测量网络 RTT...")
        try:
            start = time.time()
            response = requests.get("https://open.bigmodel.cn", timeout=10)
            rtt = (time.time() - start) * 1000
            print(f"✓ 网络 RTT: {rtt:.2f}ms")
            return rtt
        except Exception as e:
            print(f"✗ 网络测量失败: {e}")
            return -1
            
    def analyze_api_latency(self, prompt: str) -> Dict[str, Any]:
        """分析 API 调用延迟"""
        print(f"\n正在分析 API 延迟 (prompt长度: {len(prompt)} chars)...")
        
        # 测量各阶段延迟
        phases = {}
        
        # 阶段1: 请求准备
        start_total = time.time()
        
        # 阶段2: 网络传输 + API处理
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7
        }
        
        start_request = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=60)
        request_duration = (time.time() - start_request) * 1000
        
        # 阶段3: 响应解析
        start_parse = time.time()
        if response.status_code == 200:
            result = response.json()
            parse_duration = (time.time() - start_parse) * 1000
            total_duration = (time.time() - start_total) * 1000
            
            # 提取指标
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            # 计算推理速率
            inference_time = request_duration - 50  # 粗略估计网络开销
            tokens_per_second = completion_tokens / (inference_time / 1000) if inference_time > 0 else 0
            
            phases = {
                "status": "success",
                "total_latency_ms": round(total_duration, 2),
                "request_latency_ms": round(request_duration, 2),
                "parse_latency_ms": round(parse_duration, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": round(tokens_per_second, 2),
                "estimated_inference_time_ms": round(inference_time, 2),
                "response_length_chars": len(result["choices"][0]["message"]["content"]),
                "http_status": response.status_code
            }
            
            print(f"✓ 总延迟: {phases['total_latency_ms']}ms")
            print(f"  - 请求延迟: {phases['request_latency_ms']}ms")
            print(f"  - 解析延迟: {phases['parse_latency_ms']}ms")
            print(f"  - Token: {prompt_tokens} 输入 + {completion_tokens} 输出")
            print(f"  - 推理速率: {phases['tokens_per_second']} tokens/sec")
            
        else:
            phases = {
                "status": "failed",
                "http_status": response.status_code,
                "error": response.text
            }
            print(f"✗ API 调用失败: {response.status_code}")
            
        return phases
        
    def batch_analysis(self, prompts: list) -> Dict[str, Any]:
        """批量分析不同 prompt 长度的延迟"""
        print("\n=== 批量延迟分析 ===")
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n--- 测试 {i+1}/{len(prompts)} ---")
            result = self.analyze_api_latency(prompt)
            result["prompt_length"] = len(prompt)
            results.append(result)
            
        # 统计分析
        successful = [r for r in results if r["status"] == "success"]
        if successful:
            avg_latency = sum(r["total_latency_ms"] for r in successful) / len(successful)
            avg_tps = sum(r["tokens_per_second"] for r in successful) / len(successful)
            max_latency = max(r["total_latency_ms"] for r in successful)
            min_latency = min(r["total_latency_ms"] for r in successful)
            
            summary = {
                "total_tests": len(prompts),
                "successful": len(successful),
                "failed": len(prompts) - len(successful),
                "avg_latency_ms": round(avg_latency, 2),
                "avg_tokens_per_second": round(avg_tps, 2),
                "max_latency_ms": round(max_latency, 2),
                "min_latency_ms": round(min_latency, 2),
                "latency_std_dev_ms": self._calculate_std_dev([r["total_latency_ms"] for r in successful]),
                "results": results
            }
            
            print("\n=== 批量分析汇总 ===")
            print(f"测试数: {summary['total_tests']}")
            print(f"成功率: {summary['successful']}/{summary['total_tests']}")
            print(f"平均延迟: {summary['avg_latency_ms']}ms")
            print(f"平均推理速率: {summary['avg_tokens_per_second']} tokens/sec")
            print(f"延迟范围: {summary['min_latency_ms']}ms - {summary['max_latency_ms']}ms")
            print(f"延迟标准差: {summary['latency_std_dev_ms']}ms")
            
            return summary
        else:
            return {"error": "所有测试均失败"}
            
    def _calculate_std_dev(self, values: list) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return round(variance ** 0.5, 2)
        
    def generate_optimization_report(self, analysis: Dict[str, Any]) -> str:
        """生成优化建议报告"""
        report = "\n" + "="*70 + "\n"
        report += "                    GLM API 延迟优化建议报告\n"
        report += "="*70 + "\n\n"
        
        avg_latency = analysis.get("avg_latency_ms", 0)
        
        # 延迟评估
        report += "【延迟评估】\n"
        if avg_latency < 2000:
            report += "  ✓ 延迟优秀 (< 2s)\n"
        elif avg_latency < 5000:
            report += "  ⚠ 延迟一般 (2-5s)\n"
        else:
            report += "  ✗ 延迟较高 (> 5s)\n"
        report += f"  当前平均延迟: {avg_latency}ms\n\n"
        
        # 优化建议
        report += "【优化建议】\n"
        
        if avg_latency > 5000:
            report += "  1. 更换 API 端点\n"
            report += "     - 当前: open.bigmodel.cn (北京)\n"
            report += "     - 可选: 检查是否有更靠近的区域节点\n"
            report += "     - 效果: 可能降低 100-500ms\n\n"
            
        report += "  2. 启用请求缓存\n"
        report += "     - 对相同/相似请求使用本地缓存\n"
        report += "     - 实现 LRU 缓存策略\n"
        report += "     - 效果: 重复请求延迟降至 < 10ms\n\n"
        
        report += "  3. 减少输出 Token 数量\n"
        report += "     - 当前 max_tokens: 256\n"
        report += "     - 根据实际需求调整\n"
        report += "     - 效果: 每减少 100 token 约减少 500-1000ms\n\n"
        
        report += "  4. 考虑本地部署\n"
        report += "     - 部署 GLM 模型到本地或就近服务器\n"
        report += "     - 延迟可降至 100-500ms\n"
        report += "     - 成本较高，但性能提升显著\n\n"
        
        report += "  5. 使用流式响应\n"
        report += "     - 启用 stream: true\n"
        report += "     - 首字符延迟降低\n"
        report += "     - 用户体验更好\n\n"
        
        # 预期效果
        report += "【预期优化效果】\n"
        report += "  ┌──────────────────┬─────────────┐\n"
        report += "  │ 优化措施          │ 预期延迟    │\n"
        report += "  ├──────────────────┼─────────────┤\n"
        report += f"  │ 当前状态          │ {avg_latency:.0f}ms   │\n"
        report += "  │ + 缓存            │ ~500ms     │\n"
        report += "  │ + 减少输出        │ ~2000ms    │\n"
        report += "  │ + 本地部署        │ ~300ms     │\n"
        report += "  └──────────────────┴─────────────┘\n"
        
        report += "="*70 + "\n"
        return report

if __name__ == "__main__":
    analyzer = LatencyAnalyzer()
    
    # 1. 测量网络延迟
    rtt = analyzer.measure_network_latency()
    
    # 2. 批量分析不同场景
    prompts = [
        "你好",  # 短请求
        "请解释什么是人工智能",  # 中等请求
        "请详细分析端云协同架构的优势和挑战，包括技术实现、性能考量和适用场景",  # 较长请求
    ]
    
    analysis = analyzer.batch_analysis(prompts)
    
    # 3. 生成优化报告
    report = analyzer.generate_optimization_report(analysis)
    print(report)
    
    # 4. 保存结果
    with open("glm_latency_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"\n分析结果已保存: glm_latency_analysis.json")