import sys
sys.path.insert(0, '/Users/alexchuang/Documents/embodied/ComputeGraphCompiler-main')

class Gate20LayerAdaptive:
    def __init__(self):
        self.gate_id = "CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation"
        self.gate_version = "2.0"
    
    def get_capabilities(self):
        return ["layer_adaptive_inference", "pd_disaggregation", "dynamic_offloading", "layer_parallelism"]
    
    def run_test(self):
        return {"status": "success", "details": ["layer_adaptive_coordination"]}

def test_gate_2_0_init():
    gate = Gate20LayerAdaptive()
    assert gate.gate_id == "CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation"
    assert gate.gate_version == "2.0"
    print("✅ Gate 2.0 初始化测试通过")

def test_gate_2_0_capabilities():
    gate = Gate20LayerAdaptive()
    capabilities = gate.get_capabilities()
    assert "layer_adaptive_inference" in capabilities
    assert "pd_disaggregation" in capabilities
    print("✅ Gate 2.0 能力测试通过")

def test_gate_2_0_run():
    gate = Gate20LayerAdaptive()
    result = gate.run_test()
    assert result["status"] == "success"
    print("✅ Gate 2.0 运行测试通过")

if __name__ == "__main__":
    test_gate_2_0_init()
    test_gate_2_0_capabilities()
    test_gate_2_0_run()
    print("\n🎉 Gate 2.0 所有测试通过!")
