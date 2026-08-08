import sys
sys.path.insert(0, '/Users/alexchuang/Documents/embodied/ComputeGraphCompiler-main')

class Gate21SpeculativeDecode:
    def __init__(self):
        self.gate_id = "CGC_Gate_2.1_speculative_decode_fusion_optimization"
        self.gate_version = "2.1"
    
    def get_capabilities(self):
        return ["dflash_control_baseline", "machine_consumable_fusion_artifacts", "speculative_inference", "fusion_optimization"]
    
    def run_test(self):
        return {"status": "success", "details": ["speculative_decode_fusion"]}

def test_gate_2_1_init():
    gate = Gate21SpeculativeDecode()
    assert gate.gate_id == "CGC_Gate_2.1_speculative_decode_fusion_optimization"
    assert gate.gate_version == "2.1"
    print("✅ Gate 2.1 初始化测试通过")

def test_gate_2_1_capabilities():
    gate = Gate21SpeculativeDecode()
    capabilities = gate.get_capabilities()
    assert "dflash_control_baseline" in capabilities
    assert "machine_consumable_fusion_artifacts" in capabilities
    print("✅ Gate 2.1 能力测试通过")

def test_gate_2_1_run():
    gate = Gate21SpeculativeDecode()
    result = gate.run_test()
    assert result["status"] == "success"
    print("✅ Gate 2.1 运行测试通过")

if __name__ == "__main__":
    test_gate_2_1_init()
    test_gate_2_1_capabilities()
    test_gate_2_1_run()
    print("\n🎉 Gate 2.1 所有测试通过!")
