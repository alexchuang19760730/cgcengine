# ============================================================
# 🔥 CGC Compiler 完整 Ground Truth
# 合併：ggml 原始 + llama.cpp + vLLM + Kimi KDA
# 架構：4 層全優化（計算 / 儲存 / 設備IO / 調度）
# 相容：舊版 ggml ground truth 100% 沿用
# ============================================================
GROUND_TRUTH_FULL = {

    # -------------------------------------------------------------------------
    # 【舊版相容層】你一開始的 ggml ground truth
    # 完全保留、完全可用、完全不衝突
    # -------------------------------------------------------------------------
    "ggml_legacy": {
        "TILE_M": 32,
        "TILE_N": 32,
        "TILE_K": 32,
        "simd_width": 32,
        "unroll_factor": 4,
        "attn_block_size": 64,
        "fuse_qkv_rope_attn": True,
        "fuse_mlp_silu": True,
        "mem_align_bytes": 64,
        "quant_block_size": 32,
    },

    # -------------------------------------------------------------------------
    # 【多後端全知識庫】llama.cpp + vLLM 雙引擎
    # 你的編譯器可自動切換策略
    # -------------------------------------------------------------------------
    "backends": {

        # ---------------------------------------------------------------------
        # 後端 A：llama.cpp Metal（單使用者、低延遲、Mac 原生）
        # ---------------------------------------------------------------------
        "llama_cpp_metal": {
            "device": "metal",
            "source": "ggml-metal.metal",

            # 1. 計算策略
            "compute": {
                "tile_m": 32,
                "tile_n": 32,
                "tile_k": 32,
                "simd_width": 32,
                "unroll": 4,
                "fusion": [
                    "qkv_proj + rope + attn_norm + attention",
                    "mlp_up + silu + mlp_down"
                ]
            },

            # 2. 儲存策略
            "storage": {
                "weight_layout": "row-major (ggml)",
                "kv_layout": "BSHN",
                "mem_align": 64,
                "quant_block": 32,
                "memory_pool": True,
                "scratch_buffer_reuse": True,
                "no_realloc": True
            },

            # 3. 設備 IO 策略
            "device_io": {
                "metal_zero_copy": True,
                "upload_weights_once": True,
                "sync_only_at_commit": True,
                "keep_weights_in_gpu": True
            },

            # 4. 調度策略
            "scheduler": {
                "batch_size": 1,
                "continuous_batching": False,
                "prefix_caching": False,
                "paged_attention": False,
                "context_shift": True,
                "kv_cache_reuse": False
            }
        },

        # ---------------------------------------------------------------------
        # 後端 B：vLLM（高併發、工業級、伺服器、多使用者）
        # 🔥 你要的 vLLM 完整匯入
        # ---------------------------------------------------------------------
        "vllm_cuda": {
            "device": "cuda",
            "source": "vllm/src/scheduler + kernels",

            # 1. 計算策略
            "compute": {
                "tile_m": 128,
                "tile_n": 128,
                "tile_k": 32,
                "simd_width": 32,
                "unroll": 4,
                "fusion": [
                    "qkv_proj + rope + paged_attention",
                    "mlp_up + gelu + mlp_down"
                ]
            },

            # 2. 儲存策略
            "storage": {
                "weight_layout": "row-major",
                "kv_layout": "BSNH padded blocks",
                "mem_align": 128,
                "quant_block": 32,
                "paged_kv_cache": True,
                "block_size": 16,
                "memory_pool": True
            },

            # 3. 設備 IO 策略
            "device_io": {
                "pinned_memory": True,
                "cuda_memcpy_async": True,
                "overlap_data_transfer": True,
                "stream_queued": True,
                "host_device_coherence": False
            },

            # 4. 調度策略（vLLM 最強的部分）
            "scheduler": {
                "continuous_batching": True,
                "dynamic_batch_size": True,
                "prefix_caching": True,
                "paged_attention": True,
                "max_num_batched_tokens": 2048,
                "preemptive_swap": True,
                "eviction_policy": "lru",
                "kv_cache_reuse": True
            }
        },

        # ---------------------------------------------------------------------
        # 後端 C：Kimi KDA（遞歸式線性注意力）
        # ---------------------------------------------------------------------
        "kimi_kda": {
            "device": ["metal", "cuda"],
            "source": "Kimi 論文 + FlashLinearAttention",

            # 1. 計算策略
            "compute": {
                "tile_m": 64,
                "tile_n": 64,
                "tile_k": 64,
                "chunk_size": 128,
                "beta": 0.1,
                "fusion": [
                    "qkv_proj + kda_recurrent_update + kda_apply"
                ]
            },

            # 2. 儲存策略
            "storage": {
                "state_in_l1": True,
                "kv_layout": "chunked BSHN"
            },

            # 3. 設備 IO
            "device_io": {
                "zero_copy": True,
                "state_always_in_gpu": True
            },

            # 4. 調度策略
            "scheduler": {
                "chunked_recurrent": True,
                "no_attention_matrix": True,
                "state_reuse": True
            }
        }
    }
}

# -----------------------------------------------------------------------------
# 🔥 使用方法
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    # 儲存為 JSON
    with open("ground_truth_full.json", "w", encoding="utf-8") as f:
        json.dump(GROUND_TRUTH_FULL, f, indent=2, ensure_ascii=False)

    print("✅ 完整 Ground Truth 已儲存！")
    print(f"📊 包含 {len(GROUND_TRUTH_FULL['backends'])} 個後端策略")
    print(f"🔧 包含 {len(GROUND_TRUTH_FULL['ggml_legacy'])} 個舊版參數")