class KimiKDAGroundTruth:
    """
    Kimi KDA (Kernel Delta Attention) 官方论文最佳参数
    系统通过这个类"理解" kimi_kda 是什么
    """

    def get_strategy(self):
        return {
            "op_type": "kimi_kda",
            "name": "Kimi - Kernel Delta Attention",
            
            # --------------------------
            # 🔥 KDA 专属分块大小（论文参数）
            # --------------------------
            "tiling_config": {
                "Tile_M": 64,
                "Tile_N": 64,
                "Tile_K": 64,
                "chunk_size": 128,  # KDA 核心参数
            },

            # --------------------------
            # 🔥 KDA 专属算子融合
            # --------------------------
            "fusion_boundary": [
                ["qkv_proj", "kda_recurrent", "kda_apply"]
            ],

            # --------------------------
            # 🔥 KDA 专属内存排布
            # --------------------------
            "memory_hierarchy": {
                "q": "register",
                "k": "l1",
                "v": "l1",
                "state": "l1",  # KDA 状态必须在 L1
            },

            # --------------------------
            # 🔥 KDA 专属 Metal SIMD 设定
            # --------------------------
            "scheduling_plan": {
                "simd_width": 32,
                "unroll": 4,
                "use_metal_simd": True,
                "use_recurrent": True
            },

            # --------------------------
            # 🔥 KDA 论文公式参数
            # --------------------------
            "kda_beta": 0.1,
            "kda_use_delta_update": True,
            "kda_use_dplr": True
        }