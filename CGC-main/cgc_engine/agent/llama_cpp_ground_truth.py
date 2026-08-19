class LlamaCppGroundTruth:
    """
    llama.cpp 官方最佳参数
    系统通过这个类"理解" llama.cpp attention 是什么
    """

    def get_strategy(self, hidden_dim=3584, head_dim=128):
        return {
            "op_type": "attention",
            "name": "llama.cpp - Native Attention",

            "tiling_config": {
                "Tile_M": 32,
                "Tile_N": 32,
                "Tile_K": 32,
                "causal": True,
            },

            "fusion_boundary": [
                ["qkv_proj", "attention_score", "softmax", "attn_output"],
            ],

            "memory_hierarchy": {
                "q": "register",
                "k": "l2",
                "v": "l2",
            },

            "scheduling_plan": {
                "simd_width": 32,
                "unroll": 2,
                "use_metal_simd": True,
                "use_flash_attention": True,
            },
        }

    def default_strategy(self):
        return self.get_strategy()
