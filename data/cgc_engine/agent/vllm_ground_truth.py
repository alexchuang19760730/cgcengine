class VLLMGroundTruth:
    """
    vLLM 官方最佳参数 (CUDA GPU)
    系统通过这个类"理解" vLLM attention 是什么
    """

    def get_strategy(self, hidden_dim=3584, head_dim=128):
        return {
            "op_type": "attention",
            "name": "vLLM - PagedAttention",

            "tiling_config": {
                "Tile_M": 64,
                "Tile_N": 64,
                "Tile_K": 64,
                "block_size": 16,
                "causal": True,
            },

            "fusion_boundary": [
                ["qkv_proj", "attention_score", "softmax", "attn_output"],
            ],

            "memory_hierarchy": {
                "q": "register",
                "k": "gpu_vram",
                "v": "gpu_vram",
                "kv_cache": "gpu_vram",
            },

            "scheduling_plan": {
                "use_cuda_graph": True,
                "use_paged_attention": True,
                "max_num_blocks": 1024,
            },
        }

    def default_strategy(self):
        return self.get_strategy()
