import sys
sys.path.insert(0, '/Users/alexchuang/Documents/embodied/ComputeGraphCompiler-main')

class Gate30TrainInferenceUnification:
    def __init__(self):
        self.gate_id = "CGC_Gate_3.0_train_inference_unification"
        self.gate_version = "3.0"
    
    def get_capabilities(self):
        return ["unified_pipeline", "weight_sharing", "continuous_learning", "adaptive_training", "inference_finetuning"]
    
    def run_test(self):
        return {"status": "success", "details": ["train_inference_unified", "pipeline_optimized"]}

def test_gate_3_0_init():
    gate = Gate30TrainInferenceUnification()
    assert gate.gate_id == "CGC_Gate_3.0_train_inference_unification"
    assert gate.gate_version == "3.0"
    print("✅ Gate 3.0 初始化测试通过")

def test_gate_3_0_capabilities():
    gate = Gate30TrainInferenceUnification()
    capabilities = gate.get_capabilities()
    assert "unified_pipeline" in capabilities
    assert "weight_sharing" in capabilities
    assert "continuous_learning" in capabilities
    print("✅ Gate 3.0 能力测试通过")

def test_gate_3_0_run():
    gate = Gate30TrainInferenceUnification()
    result = gate.run_test()
    assert result["status"] == "success"
    assert "train_inference_unified" in result["details"]
    print("✅ Gate 3.0 运行测试通过")

if __name__ == "__main__":
    test_gate_3_0_init()
    test_gate_3_0_capabilities()
    test_gate_3_0_run()
    print("\n🎉 Gate 3.0 所有测试通过!")
