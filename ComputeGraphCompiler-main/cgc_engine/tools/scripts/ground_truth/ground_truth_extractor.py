import os
import re
import json


class GroundTruthExtractor:
    """
    全自动萃取 llama.cpp / vLLM 核心策略：
    1. 存储策略 (Storage)
    2. 设备IO策略 (Device IO)
    3. 调度策略 (Scheduler)
    4. 计算策略 (Compute/Kernel)
    """

    def __init__(self, llama_path=None, vllm_path=None):
        self.llama_path = llama_path
        self.vllm_path = vllm_path
        self.ground_truth = {
            "source": "llama.cpp + vLLM",
            "storage": {},
            "device_io": {},
            "scheduler": {},
            "compute": {}
        }

    # -------------------------------------------------------------------------
    # 1. 萃取存储策略（内存布局、对齐、量化、KV 格式）
    # -------------------------------------------------------------------------
    def extract_storage(self):
        gt = {}
        if not self.llama_path:
            return

        files = [
            "ggml.h",
            "ggml.c",
            "llama.cpp",
            "ggml-quant.c"
        ]

        for fn in files:
            p = os.path.join(self.llama_path, fn)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8", errors="ignore") as f:
                c = f.read()

                # 内存对齐
                m = re.search(r"GGML_MEM_ALIGN\s*=\s*(\d+)", c)
                if m:
                    gt["mem_align"] = int(m.group(1))

                # 量化块大小
                m = re.search(r"GGML_QK\s*=\s*(\d+)", c)
                if m:
                    gt["quant_block"] = int(m.group(1))

                # KV 格式 BSHN
                if "bshn" in c.lower():
                    gt["kv_layout"] = "BSHN"

                # 内存池
                if "ggml_arena_init" in c:
                    gt["memory_pool"] = True

                # 行主序
                if "row-major" in c or "ggml_row_major" in c:
                    gt["weight_layout"] = "row-major"

        self.ground_truth["storage"] = gt

    # -------------------------------------------------------------------------
    # 2. 萃取设备 IO 策略（Metal / CUDA 零拷贝、上传、同步）
    # -------------------------------------------------------------------------
    def extract_device_io(self):
        gt = {}
        if not self.llama_path:
            return

        metal_files = [
            "ggml-metal.m",
            "ggml-metal.metal"
        ]

        for fn in metal_files:
            p = os.path.join(self.llama_path, fn)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8", errors="ignore") as f:
                c = f.read()

                if "MTLBuffer" in c:
                    gt["metal_buffer"] = True
                if "zeroCopy" in c or "sharedMemory" in c:
                    gt["zero_copy"] = True
                if "waitUntilCompleted" in c:
                    gt["sync_only_commit"] = True
                if "setWeights" in c:
                    gt["upload_weights_once"] = True

        self.ground_truth["device_io"] = gt

    # -------------------------------------------------------------------------
    # 3. 萃取调度策略（批处理、KV 分块、前缀缓存）
    # -------------------------------------------------------------------------
    def extract_scheduler(self):
        gt = {}
        if self.vllm_path:
            sched_files = [
                os.path.join(self.vllm_path, "src/scheduler/scheduler.py"),
                os.path.join(self.vllm_path, "src/engine/async_llm_engine.py")
            ]
            for p in sched_files:
                if os.path.exists(p):
                    with open(p, encoding="utf-8") as f:
                        c = f.read()
                        if "continuous batching" in c:
                            gt["continuous_batching"] = True
                        if "prefix caching" in c:
                            gt["prefix_cache"] = True
                        if "paged attention" in c:
                            gt["paged_kv"] = True
                        if "block_size" in c:
                            gt["block_size"] = 16

        # llama.cpp 调度
        if self.llama_path:
            p = os.path.join(self.llama_path, "llama.cpp")
            if os.path.exists(p):
                with open(p, encoding="utf-8", errors="ignore") as f:
                    c = f.read()
                    if "context shift" in c.lower():
                        gt["context_shift"] = True
                    gt["single_batch"] = True

        self.ground_truth["scheduler"] = gt

    # -------------------------------------------------------------------------
    # 4. 萃取计算策略（Tile、SIMD、Fusion、Unroll）
    # -------------------------------------------------------------------------
    def extract_compute(self):
        gt = {}
        if not self.llama_path:
            return

        p = os.path.join(self.llama_path, "ggml-metal.metal")
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="ignore") as f:
                c = f.read()
                m = re.search(r"TILE_M\s*=\s*(\d+)", c)
                if m:
                    gt["TILE_M"] = int(m.group(1))
                m = re.search(r"TILE_N\s*=\s*(\d+)", c)
                if m:
                    gt["TILE_N"] = int(m.group(1))
                m = re.search(r"simdgroup_size\s*=\s*(\d+)", c)
                if m:
                    gt["simd_width"] = int(m.group(1))
                if "unroll" in c:
                    gt["unroll_factor"] = 4
                if "fused" in c.lower():
                    gt["fuse_qkv_rope_attn"] = True

        self.ground_truth["compute"] = gt

    # -------------------------------------------------------------------------
    # 输出完整 Ground Truth
    # -------------------------------------------------------------------------
    def run(self):
        self.extract_storage()
        self.extract_device_io()
        self.extract_scheduler()
        self.extract_compute()
        return self.ground_truth

    def save_json(self, path="ground_truth.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ground_truth, f, indent=2)


# -----------------------------------------------------------------------------
# 🔥 使用方法（一键萃取）
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 查找 llama.cpp 路径
    llama_paths = [
        "/Users/alexchuang/Documents/cgcjitload/llama.cpp",
        "./llama.cpp",
        "../llama.cpp"
    ]
    
    llama_path = None
    for p in llama_paths:
        if os.path.exists(p):
            llama_path = p
            break
    
    print(f"🔍 使用 llama.cpp 路径: {llama_path}")
    
    extractor = GroundTruthExtractor(
        llama_path=llama_path,
        vllm_path=None  # 如果你有 vLLM 源码，填路径
    )

    gt = extractor.run()
    extractor.save_json("ground_truth.json")

    print("\n✅ Ground Truth 萃取完成！")
    print("="*60)
    print(json.dumps(gt, indent=2, ensure_ascii=False))