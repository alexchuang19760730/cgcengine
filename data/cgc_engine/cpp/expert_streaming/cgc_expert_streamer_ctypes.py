import os
import sys
import ctypes
import ctypes.util
from ctypes import (
    c_void_p, c_char_p, c_int, c_uint32, c_uint64, c_bool,
    POINTER, Structure, Array, c_float
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 常量与 cgc_expert_streamer.h 对齐
CGC_MAX_PATH_LEN = 512
CGC_MAX_EXPERTS_PER_LAYER = 256
CGC_MAX_LAYERS = 256
CGC_MAX_EXPERT_REGIONS = 8
CGC_MAX_SLOT_COUNT = 1024

class CGCStreamLayout(Structure):
    """与 cgc_stream_layout_t 严格对齐（含 per-layer 多 region 字段）。"""
    _fields_ = [
        ("path", ctypes.c_char * CGC_MAX_PATH_LEN),
        ("stream_offset", c_uint64),
        ("stream_size", c_uint64),
        ("experts_per_layer", c_int),
        ("expert_stride", c_uint64),
        ("expert_offsets", c_uint64 * CGC_MAX_EXPERTS_PER_LAYER),
        ("has_explicit_offsets", c_int),
        # per-layer 多 region 布局 (Gemma4 per-layer GGUF)
        ("region_count", c_int),
        ("region_stride", c_uint64 * CGC_MAX_EXPERT_REGIONS),
        ("layer_offsets", (c_uint64 * CGC_MAX_EXPERT_REGIONS) * CGC_MAX_LAYERS),
        ("has_layer_offsets", c_int),
    ]

class CGCCacheResult(Structure):
    """与 cgc_cache_result_t 严格对齐（注意字段顺序：先 buffers/offsets/sizes）。"""
    _fields_ = [
        ("buffers", c_void_p * CGC_MAX_EXPERTS_PER_LAYER),
        ("offsets", c_uint64 * CGC_MAX_EXPERTS_PER_LAYER),
        ("sizes", c_uint64 * CGC_MAX_EXPERTS_PER_LAYER),
        ("count", c_int),
        ("hits", c_int),
        ("misses", c_int),
        ("read_wall_nanos", c_uint64),
        ("read_bytes", c_uint64),
    ]

class CGCCacheAccessCtx(Structure):
    """与 cgc_cache_access_ctx_t 严格对齐。"""
    _fields_ = [
        ("owner_phase", c_int),       # cgc_cache_slot_phase_t
        ("control_plane", c_int),     # cgc_cache_control_plane_t
        ("request_id", c_uint64),
        ("decode_step_index", c_int),
    ]

class CGCTelemetry(Structure):
    """与 cgc_cache_telemetry_t 严格对齐。"""
    _fields_ = [
        ("slot_count", c_int),
        ("occupied_slots", c_int),
        ("total_requests", c_uint64),
        ("total_hits", c_uint64),
        ("total_misses", c_uint64),
        ("total_loads", c_uint64),
        ("total_evictions", c_uint64),
        ("total_read_wall_nanos", c_uint64),
        ("total_read_bytes", c_uint64),
        ("total_prefetch_wall_nanos", c_uint64),
    ]

class CGCExpertTensorInfo(Structure):
    _fields_ = [
        ("expert_id", c_int),
        ("role", ctypes.c_char * 256),
        ("ggml_type", c_int),
        ("dims", c_uint64 * 4),
        ("n_dims", c_int),
        ("offset", c_uint64),
        ("size", c_uint64),
    ]

class CGCLayerGGUFMeta(Structure):
    _fields_ = [
        ("layer_index", c_int),
        ("expert_count", c_int),
        ("hidden_size", c_int),
        ("intermediate_size", c_int),
        ("ggml_type", c_int),
        ("quant_block_size", c_int),
    ]

class CGCLayerAssignment(Structure):
    _fields_ = [
        ("total_layers", c_int),
        ("prefill_count", c_int),
        ("decode_count", c_int),
        ("prefill_layers", c_int * 256),
        ("decode_layers", c_int * 256),
    ]

class CGCRouteEntry(Structure):
    _fields_ = [
        ("token_id", c_int),
        ("expert_ids", c_int * 256),
        ("expert_count", c_int),
        ("layer_index", c_int),
        ("timestamp_nanos", c_uint64),
    ]

class CGCSchedulerStats(Structure):
    _fields_ = [
        ("total_prefill_tokens", c_uint64),
        ("total_decode_tokens", c_uint64),
        ("prefill_switch_count", c_uint64),
        ("decode_switch_count", c_uint64),
        ("gpu0_cache_hits", c_uint64),
        ("gpu0_cache_misses", c_uint64),
        ("gpu1_cache_hits", c_uint64),
        ("gpu1_cache_misses", c_uint64),
    ]


def _find_c_library():
    # 0) 显式覆盖：CGC_EX_STREAM_LIB 指向具体库文件（调试 / 多版本共存用，优先级最高）
    override = os.environ.get("CGC_EX_STREAM_LIB", "")
    for cand in (override, override + ".dll", override + ".so", override + ".dylib"):
        if cand and os.path.exists(cand):
            try:
                return ctypes.CDLL(cand)
            except OSError:
                pass

    search_paths = [
        BASE_DIR,
        os.path.join(BASE_DIR, "build"),
        os.path.join(BASE_DIR, "build", "Release"),
        os.path.join(BASE_DIR, "build", "Debug"),
        os.path.join(os.environ.get("TEMP", ""), ""),
    ]

    # 已存在的实际 DLL 名（按优先级）：最新构建 cgc_stream_v3 排最前
    lib_names = [
        "cgc_stream_v3", "cgc_stream_v2", "cgc_stream", "cs6", "cs5", "cs4", "stream",
        "cgc_expert_streamer", "cgc_streamer",
    ]

    # 1) 显式路径（BASE_DIR / build / TEMP ...）：绝对路径直载，优先于 PATH 裸名
    for path in search_paths:
        for name in lib_names:
            for ext in [".dll", ".so", ".dylib", ".a", ".lib"]:
                full = os.path.join(path, f"{name}{ext}")
                if os.path.exists(full):
                    try:
                        lib = ctypes.CDLL(full)
                        return lib
                    except OSError:
                        pass

    # 2) 兜底：PATH / DLL 搜索路径上的裸名
    for name in lib_names:
        for ext in [".dll", ".so", ".dylib"]:
            try:
                lib = ctypes.CDLL(f"{name}{ext}")
                return lib
            except OSError:
                pass

    return None


def _build_shared_lib():
    import subprocess
    import shutil

    if not shutil.which("gcc"):
        raise RuntimeError("gcc not found. Please install MinGW-w64 or add gcc to PATH.")

    output_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(output_dir, exist_ok=True)
    output_lib = os.path.join(output_dir, "cgc_expert_streamer.dll")

    c_files = [
        "cgc_expert_streamer.c",
        "cgc_expert_streamer_gguf.c",
        "cgc_gguf_lite.c",
        "cgc_expert_compute.c",
        "cgc_pd_scheduler.c",
    ]

    src_files = [os.path.join(BASE_DIR, f) for f in c_files]
    for f in src_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Source file not found: {f}")

    cmd = [
        "gcc", "-std=c11", "-shared", "-O2", "-Wall",
        "-o", output_lib
    ] + src_files + ["-I", BASE_DIR, "-lws2_32"]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Build failed:\n{result.stderr}")

    return ctypes.CDLL(output_lib)


class CGCExpertStreamer:
    def __init__(self, lib_path=None, auto_build=True):
        self.lib = None
        self._load_lib(lib_path, auto_build)
        self._setup_functions()

    def _load_lib(self, lib_path, auto_build):
        if lib_path:
            self.lib = ctypes.CDLL(lib_path)
            return

        self.lib = _find_c_library()
        if self.lib:
            return

        if auto_build:
            try:
                self.lib = _build_shared_lib()
                return
            except Exception as e:
                print(f"[CGC] Auto-build failed: {e}")

        raise RuntimeError(
            "Cannot find cgc_expert_streamer library. "
            "Build with: gcc -shared -o cgc_expert_streamer.dll "
            "cgc_expert_streamer.c cgc_expert_streamer_gguf.c cgc_gguf_lite.c "
            "cgc_expert_compute.c cgc_pd_scheduler.c"
        )

    def _setup_functions(self):
        lib = self.lib

        # (C 导出名, restype, argtypes)。逐项 guard：磁盘上的 DLL 可能是不同构建
        # 版本，缺失的导出跳过即可，对应 Python 方法仅在调用时才报错。
        specs = [
            ("cgc_expert_streamer_create", c_void_p,
             [POINTER(CGCStreamLayout), c_int, c_bool, POINTER(c_int), c_int]),
            ("cgc_expert_streamer_destroy", None, [c_void_p]),
            ("cgc_expert_streamer_load_experts", CGCCacheResult,
             [c_void_p, POINTER(c_int), c_int, POINTER(CGCCacheAccessCtx)]),
            ("cgc_expert_streamer_prefetch", None, [c_void_p, POINTER(c_int), c_int]),
            ("cgc_expert_streamer_prefetch_load", c_int,
             [c_void_p, POINTER(c_int), c_int, POINTER(CGCCacheAccessCtx)]),
            ("cgc_expert_streamer_set_layer", None, [c_void_p, c_int]),
            ("cgc_expert_streamer_telemetry", CGCTelemetry, [c_void_p]),
            ("cgc_stream_layout_compute_offset", c_uint64, [POINTER(CGCStreamLayout), c_int, c_int]),
            ("cgc_load_stream_layout_from_gguf", CGCStreamLayout, [c_char_p]),
            ("cgc_pd_layer_assignment_by_ratio", CGCLayerAssignment, [c_int, c_float]),
            ("cgc_pd_layer_assignment_custom", CGCLayerAssignment,
             [POINTER(c_int), c_int, POINTER(c_int), c_int]),
            ("cgc_streamer_pool_create", c_void_p, []),
            ("cgc_streamer_pool_destroy", None, [c_void_p]),
            ("cgc_streamer_pool_add", c_bool, [c_void_p, c_int, c_void_p]),
            ("cgc_streamer_pool_get", c_void_p, [c_void_p, c_int]),
            ("cgc_streamer_pool_load_experts", CGCCacheResult,
             [c_void_p, c_int, POINTER(c_int), c_int, POINTER(CGCCacheAccessCtx)]),
            ("cgc_streamer_pool_prefetch", None, [c_void_p, c_int, POINTER(c_int), c_int]),
            ("cgc_pd_scheduler_create", c_void_p,
             [c_void_p, POINTER(CGCLayerAssignment), c_int, c_int]),
            ("cgc_pd_scheduler_destroy", None, [c_void_p]),
            ("cgc_pd_scheduler_enter_prefill", None, [c_void_p]),
            ("cgc_pd_scheduler_switch_to_decode", None, [c_void_p]),
            ("cgc_pd_scheduler_get_stats", CGCSchedulerStats, [c_void_p]),
            ("cgc_pd_scheduler_reset_stats", None, [c_void_p]),
        ]

        for name, restype, argtypes in specs:
            fn = getattr(lib, name, None)
            if fn is None:
                print(f"[CGC] warning: exported function {name} missing in loaded library")
                continue
            fn.restype = restype
            fn.argtypes = argtypes

    def create_streamer(self, path, stream_offset, experts_per_layer,
                         expert_stride, slot_count=8, use_mmap=False):
        layout = CGCStreamLayout()
        layout.path = path.encode("utf-8")
        layout.stream_offset = stream_offset
        layout.experts_per_layer = experts_per_layer
        layout.expert_stride = expert_stride
        layout.has_explicit_offsets = 0

        streamer = self.lib.cgc_expert_streamer_create(
            ctypes.byref(layout), slot_count, use_mmap, None, 0
        )
        return streamer

    def destroy_streamer(self, streamer):
        self.lib.cgc_expert_streamer_destroy(streamer)

    def load_experts(self, streamer, expert_ids, ctx=None):
        count = len(expert_ids)
        arr = (c_int * count)(*expert_ids)
        if ctx is None:
            ctx_ptr = None
        else:
            ctx_ptr = ctypes.byref(ctx)
        return self.lib.cgc_expert_streamer_load_experts(
            streamer, arr, count, ctx_ptr
        )

    def prefetch(self, streamer, expert_ids):
        count = len(expert_ids)
        arr = (c_int * count)(*expert_ids)
        self.lib.cgc_expert_streamer_prefetch(streamer, arr, count)

    def prefetch_load(self, streamer, expert_ids, ctx=None):
        """主动预取加载 (miss-only): 把未缓存的专家真正读入缓存槽。"""
        fn = getattr(self.lib, "cgc_expert_streamer_prefetch_load", None)
        if fn is None:
            return None
        count = len(expert_ids)
        arr = (c_int * count)(*expert_ids)
        if ctx is None:
            ctx_ptr = None
        else:
            ctx_ptr = ctypes.byref(ctx)
        return fn(streamer, arr, count, ctx_ptr)

    def set_layer(self, streamer, layer):
        fn = getattr(self.lib, "cgc_expert_streamer_set_layer", None)
        if fn is not None:
            fn(streamer, layer)

    def get_telemetry(self, streamer):
        return self.lib.cgc_expert_streamer_telemetry(streamer)

    def compute_offset(self, layout, layer, expert):
        return self.lib.cgc_stream_layout_compute_offset(
            ctypes.byref(layout), layer, expert
        )

    def load_layout_from_gguf(self, gguf_path):
        return self.lib.cgc_load_stream_layout_from_gguf(
            gguf_path.encode("utf-8")
        )

    def build_layout_per_layer(self, path, experts_per_layer,
                               region_count, region_stride, layer_offsets,
                               stream_size=0, stream_offset=0):
        """构造 per-layer 多 region 布局（Gemma4 ffn_down_exps + ffn_gate_up_exps）。"""
        layout = CGCStreamLayout()
        layout.path = path.encode("utf-8")
        layout.stream_offset = stream_offset
        layout.stream_size = stream_size
        layout.experts_per_layer = experts_per_layer
        # 单 expert 完整字节数 = 所有 region 合计 (C 侧 slot buffer 分配需要非 0 大小,
        # 否则 aligned_alloc(0) -> read_expert memcpy 越界 -> access violation)
        layout.expert_stride = sum(int(x) for x in region_stride)
        layout.has_explicit_offsets = 0
        layout.region_count = region_count
        for r in range(region_count):
            layout.region_stride[r] = region_stride[r]
            for l in range(len(layer_offsets)):
                layout.layer_offsets[l][r] = layer_offsets[l][r]
        layout.has_layer_offsets = 1
        return layout

    def create_streamer_for_layer(self, layout, layer, slot_count=16,
                                  hot_pool=None, use_mmap=True):
        """为 per-layer 布局的指定层创建 streamer。

        set_layer 是可选的（磁盘上的 DLL 可能是不含该导出的旧构建）：
        缺失时跳过 —— streamer 已经通过 per-layer layout 绑定到具体层，
        layer 语义由调用方在 pool_add / pool_load_experts 时显式给出。
        """
        hot_arr = None
        hot_count = 0
        if hot_pool:
            hot_ids = [int(i) for i in hot_pool if int(i) >= 0]
            if hot_ids:
                hot_arr = (c_int * len(hot_ids))(*hot_ids)
                hot_count = len(hot_ids)
        streamer = self.lib.cgc_expert_streamer_create(
            ctypes.byref(layout), slot_count, use_mmap, hot_arr, hot_count
        )
        if streamer:
            set_layer = getattr(self.lib, "cgc_expert_streamer_set_layer", None)
            if set_layer is not None:
                set_layer(streamer, layer)
        return streamer

    def pool_create(self):
        return self.lib.cgc_streamer_pool_create()

    def pool_add(self, pool, layer_idx, streamer):
        return self.lib.cgc_streamer_pool_add(pool, layer_idx, streamer)

    def pool_get(self, pool, layer_idx):
        return self.lib.cgc_streamer_pool_get(pool, layer_idx)

    def pool_load_experts(self, pool, layer_idx, expert_ids, ctx=None):
        count = len(expert_ids)
        arr = (c_int * count)(*expert_ids)
        if ctx is None:
            ctx_ptr = None
        else:
            ctx_ptr = ctypes.byref(ctx)
        return self.lib.cgc_streamer_pool_load_experts(pool, layer_idx, arr, count, ctx_ptr)

    def pool_prefetch(self, pool, layer_idx, expert_ids):
        count = len(expert_ids)
        arr = (c_int * count)(*expert_ids)
        self.lib.cgc_streamer_pool_prefetch(pool, layer_idx, arr, count)

    def pool_prefetch_load(self, pool, layer_idx, expert_ids, ctx=None):
        """池级主动预取加载 (miss-only 实际读入缓存槽)。

        相当于 pool_prefetch 但走 cgc_expert_streamer_prefetch_load:
        对未缓存且非热池的专家真正发起磁盘读, 而不是只发 mmap hint。
        #10 接线: on_token_routes 用真实 selected_experts 驱动实际预取。
        """
        streamer = self.pool_get(pool, layer_idx)
        if not streamer:
            return 0
        return self.prefetch_load(streamer, expert_ids, ctx)

    def create_pd_scheduler(self, total_layers, prefill_ratio=0.5,
                            max_experts_per_layer=8, tile_experts=8):
        pool = self.lib.cgc_streamer_pool_create()
        assignment = self.lib.cgc_pd_layer_assignment_by_ratio(
            total_layers, c_float(prefill_ratio)
        )
        scheduler = self.lib.cgc_pd_scheduler_create(
            pool, ctypes.byref(assignment),
            max_experts_per_layer, tile_experts
        )
        return scheduler, pool, assignment

    def destroy_pd_scheduler(self, scheduler, pool=None):
        self.lib.cgc_pd_scheduler_destroy(scheduler)
        if pool:
            self.lib.cgc_streamer_pool_destroy(pool)

    def scheduler_enter_prefill(self, scheduler):
        self.lib.cgc_pd_scheduler_enter_prefill(scheduler)

    def scheduler_switch_to_decode(self, scheduler):
        self.lib.cgc_pd_scheduler_switch_to_decode(scheduler)

    def scheduler_get_stats(self, scheduler):
        return self.lib.cgc_pd_scheduler_get_stats(scheduler)

    def scheduler_reset_stats(self, scheduler):
        self.lib.cgc_pd_scheduler_reset_stats(scheduler)


if __name__ == "__main__":
    print("Testing CGC Expert Streamer Python Bindings...")

    cgc = CGCExpertStreamer(auto_build=True)
    print("CGC library loaded successfully!")

    layout = cgc.load_layout_from_gguf("nonexistent.gguf")
    print(f"Layout from non-existent file: experts_per_layer={layout.experts_per_layer}")

    streamer = cgc.create_streamer(
        path="test.gguf",
        stream_offset=4096,
        experts_per_layer=8,
        expert_stride=256 * 1024,
        slot_count=4,
        use_mmap=False
    )

    if streamer:
        print(f"Streamer created: {streamer}")

        telemetry = cgc.get_telemetry(streamer)
        print(f"Initial telemetry: requests={telemetry.total_requests}")

        cgc.destroy_streamer(streamer)
        print("Streamer destroyed.")
    else:
        print("Streamer creation failed (expected - file doesn't exist)")

    scheduler, pool, assignment = cgc.create_pd_scheduler(8, 0.5)
    print(f"PD Scheduler: prefill={assignment.prefill_count}, decode={assignment.decode_count}")

    cgc.scheduler_enter_prefill(scheduler)
    print("Prefill phase entered.")

    stats = cgc.scheduler_get_stats(scheduler)
    print(f"Stats: prefill_tokens={stats.total_prefill_tokens}")

    cgc.scheduler_switch_to_decode(scheduler)
    print("Switched to decode phase.")

    cgc.destroy_pd_scheduler(scheduler, pool)
    print("PD Scheduler destroyed.")

    print("\nAll Python binding tests passed!")
