#!/usr/bin/env python3
"""
双模式推理测试脚本
模式1: DeepSeek V4 + MiniCPM5 (FusionRoute)
模式2: GLM 5.2 MOA 模式
"""

import os
import json
import time
from typing import Dict, Any

# 设置环境变量
os.environ["GLM_API_KEY"] = "bcdebd9b768f40528603f07626be2493.VGi12nP3d3zf1gvh"
os.environ["GLM_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"

class DualModeTester:
    def __init__(self):
        self.results = {}
        
    def test_glm_moa_mode(self) -> Dict[str, Any]:
        """测试 GLM 5.2 MOA 模式"""
        print("=== 开始测试 GLM MOA 模式 ===")
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=os.getenv("GLM_API_KEY"),
                base_url=os.getenv("GLM_API_BASE")
            )
            
            start_time = time.time()
            response = client.chat.completions.create(
                model="glm-5.2",
                messages=[
                    {"role": "system", "content": "你是一个智能助手，擅长分析和解决问题"},
                    {"role": "user", "content": "请分析端云协同架构的优势和挑战"}
                ],
                max_tokens=512,
                temperature=0.7
            )
            latency = time.time() - start_time
            
            result = {
                "mode": "glm_moa",
                "status": "success",
                "model": "glm-5.2",
                "latency_ms": round(latency * 1000, 2),
                "token_count": response.usage.total_tokens,
                "response": response.choices[0].message.content[:200] + "..."
            }
            print(f"✓ GLM MOA 测试成功 | 延迟: {result['latency_ms']}ms | Token: {result['token_count']}")
            return result
            
        except Exception as e:
            print(f"✗ GLM MOA 测试失败: {str(e)}")
            return {
                "mode": "glm_moa",
                "status": "failed",
                "error": str(e)
            }
    
    def test_deepseek_fusionroute_mode(self) -> Dict[str, Any]:
        """测试 DeepSeek V4 + MiniCPM5 FusionRoute 模式"""
        print("=== 开始测试 DeepSeek V4 + MiniCPM5 FusionRoute 模式 ===")
        try:
            from cgc_engine.bridge.megatrain_vllm_bridge import MegatrainVLLMBridge
            
            bridge = MegatrainVLLMBridge()
            start_time = time.time()
            
            result = bridge.zero_copy_transfer(
                target_backend="sglang",
                routing_mode="fusionroute",
                router_model="minicpm5-1b",
                cloud_instances=4,
                keep_on_device=True
            )
            latency = time.time() - start_time
            
            test_result = {
                "mode": "deepseek_fusionroute",
                "status": "success",
                "router_model": "minicpm5-1b",
                "cloud_instances": 4,
                "latency_ms": round(latency * 1000, 2),
                "routing_mode": result.get("routing_mode", "fusionroute"),
                "edge_cloud_protocol": result.get("edge_cloud_protocol", {}).get("schema_version", "unknown")
            }
            print(f"✓ DeepSeek FusionRoute 测试成功 | 延迟: {test_result['latency_ms']}ms")
            return test_result
            
        except ImportError as e:
            print(f"⚠️ DeepSeek FusionRoute 测试跳过: 依赖未安装 - {str(e)}")
            return {
                "mode": "deepseek_fusionroute",
                "status": "skipped",
                "reason": "依赖未安装"
            }
        except Exception as e:
            print(f"✗ DeepSeek FusionRoute 测试失败: {str(e)}")
            return {
                "mode": "deepseek_fusionroute",
                "status": "failed",
                "error": str(e)
            }
    
    def test_gate_5_0_tracing(self) -> Dict[str, Any]:
        """测试 Gate 5.0 追踪能力"""
        print("=== 开始测试 Gate 5.0 追踪能力 ===")
        try:
            from cgc_engine.tracing.audit_trace import AuditTracer
            
            tracer = AuditTracer()
            start_time = time.time()
            
            trace_data = tracer.trace_api_call(
                endpoint="https://open.bigmodel.cn/api/paas/v4/chat/completions",
                method="POST",
                parameters={"model": "glm-5.2", "messages": [...]},
                response_code=200,
                latency_ms=856
            )
            latency = time.time() - start_time
            
            result = {
                "mode": "gate_5_0_tracing",
                "status": "success",
                "trace_id": trace_data.get("trace_id", "unknown"),
                "latency_ms": round(latency * 1000, 2),
                "components": ["audit_trace", "performance_analysis", "compliance_reporting"]
            }
            print(f"✓ Gate 5.0 追踪测试成功 | 延迟: {result['latency_ms']}ms")
            return result
            
        except ImportError as e:
            print(f"⚠️ Gate 5.0 追踪测试跳过: 依赖未安装 - {str(e)}")
            return {
                "mode": "gate_5_0_tracing",
                "status": "skipped",
                "reason": "依赖未安装"
            }
        except Exception as e:
            print(f"✗ Gate 5.0 追踪测试失败: {str(e)}")
            return {
                "mode": "gate_5_0_tracing",
                "status": "failed",
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("="*60)
        print("双模式推理测试套件 v1.0")
        print("="*60)
        print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GLM API Key: {os.getenv('GLM_API_KEY', '未设置')[:10]}...")
        print()
        
        self.results = {
            "test_summary": {},
            "test_results": []
        }
        
        tests = [
            self.test_glm_moa_mode,
            self.test_deepseek_fusionroute_mode,
            self.test_gate_5_0_tracing
        ]
        
        for test_func in tests:
            result = test_func()
            self.results["test_results"].append(result)
            print()
        
        # 统计结果
        total = len(self.results["test_results"])
        success = sum(1 for r in self.results["test_results"] if r["status"] == "success")
        failed = sum(1 for r in self.results["test_results"] if r["status"] == "failed")
        skipped = sum(1 for r in self.results["test_results"] if r["status"] == "skipped")
        
        self.results["test_summary"] = {
            "total_tests": total,
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "success_rate": round(success / total * 100, 2) if total > 0 else 0
        }
        
        # 输出汇总
        print("="*60)
        print("测试汇总")
        print("="*60)
        print(f"总计: {total} | 成功: {success} | 失败: {failed} | 跳过: {skipped}")
        print(f"成功率: {self.results['test_summary']['success_rate']}%")
        
        # 保存结果
        output_path = "dual_mode_test_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n测试结果已保存: {output_path}")
        
        return self.results

if __name__ == "__main__":
    tester = DualModeTester()
    tester.run_all_tests()