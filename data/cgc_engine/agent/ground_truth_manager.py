from .llama_cpp_ground_truth import LlamaCppGroundTruth
from .vllm_ground_truth import VLLMGroundTruth
from .kimi_kda_ground_truth import KimiKDAGroundTruth

class GroundTruthManager:
    def __init__(self, device="metal"):
        self.device = device
        self.llama = LlamaCppGroundTruth()
        self.vllm = VLLMGroundTruth()
        self.kimi = KimiKDAGroundTruth()  # 👈 Kimi KDA 专用 GT

    def get_optimal_strategy(self, device="metal", hidden_dim=3584, head_dim=128):
        """
        给定模型规格 → 回传 llama.cpp / vLLM 最优策略
        """
        if device == "metal":
            return self.llama.get_strategy(hidden_dim, head_dim)
        elif device == "cuda":
            return self.vllm.get_strategy(hidden_dim, head_dim)
        else:
            return self.default_strategy()

    # 🔥 重点在这里！！！
    def get_strategy_for_op(self, op_name: str):
        """
        系统就是靠这里判断 kimi_kda 是什么
        """
        if op_name == "kimi_kda":
            return self.kimi.get_strategy()  # 👈 直接回传 KDA 策略
        elif op_name == "attention":
            return self.llama.get_strategy()
        elif op_name == "flash_attention":
            return self.vllm.get_strategy()
        else:
            return self.default_strategy()

    def default_strategy(self):
        return {
            "fusion_boundary": [["qkv_proj", "attn"], ["mlp"]],
            "tiling_config": {"Tile_M": 32, "Tile_N": 32, "Tile_K": 32},
            "memory_hierarchy": {"qkv": "register", "attn": "l1"},
            "scheduling_plan": {"unroll": 4, "prefetch": False}
        }