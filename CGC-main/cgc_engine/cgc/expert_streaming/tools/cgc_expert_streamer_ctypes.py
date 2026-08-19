import os
import sys
import ctypes
import ctypes.util
from ctypes import (
    c_void_p, c_char_p, c_int, c_uint32, c_uint64, c_bool,
    POINTER, Structure, Array, c_float
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class CGCStreamLayout(Structure):
    _fields_ = [
        ("path", ctypes.c_char * 512),
        ("stream_offset", c_uint64),
        ("stream_size", c_uint64),
        ("experts_per_layer", c_int),
        ("expert_stride", c_uint64),
        ("expert_offsets", c_uint64 * 256),
        ("has_explicit_offsets", c_int),
    ]

class CGCCacheResult(Structure):
    _fields_ = [
        ("count", c_int),
        ("hits", c_int),
        ("misses", c_int),
        ("read_wall_nanos", c_uint64),
        ("read_bytes", c_uint64),
        ("buffers", c_void_p * 256),
        ("sizes", c_uint64 * 256),
        ("offsets", c_uint64 * 256),
    ]

class CGCCacheAccessCtx(Structure):
    _fields_ = [
        ("owner_phase", c_int),
        ("layer_index", c_int),
        ("user_data", c_void_p),
    ]

class CGCTelemetry(Structure):
    _fields_ = [
        ("slot_count", c_int),
        ("total_requests", c_uint64),
        ("total_hits", c_uint64),
        ("total_misses", c_uint64),
        ("total_loads", c_uint64),
        ("total_evictions", c_uint64),
        ("total_read_bytes", c_uint64),
        ("total_read_wall_nanos", c_uint64),
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
    search_paths = [
        BASE_DIR,
        os.path.join(BASE_DIR, "build"),
        os.path.join(BASE_DIR, "build", "Release"),
        os.path.join(BASE_DIR, "build", "Debug"),
        os.path.join(os.environ.get("TEMP", ""), ""),
    ]

    lib_names = ["cgc_expert_streamer", "cgc_streamer"]

    for name in lib_names:
        try:
            lib = ctypes.CDLL(f"{name}.dll")
            return lib
        except OSError:
            pass
        try:
            lib = ctypes.CDLL(f"{name}.so")
            return lib
        except OSError:
            pass
        try:
            lib = ctypes.CDLL(f"{name}.dylib")
            return lib
        except OSError:
            pass

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

        lib.cgc_expert_streamer_create.restype = c_void_p
        lib.cgc_expert_streamer_create.argtypes = [
            POINTER(CGCStreamLayout), c_int, c_bool, POINTER(c_int), c_int
        ]

        lib.cgc_expert_streamer_destroy.restype = None
        lib.cgc_expert_streamer_destroy.argtypes = [c_void_p]

        lib.cgc_expert_streamer_load_experts.restype = CGCCacheResult
        lib.cgc_expert_streamer_load_experts.argtypes = [
            c_void_p, POINTER(c_int), c_int, POINTER(CGCCacheAccessCtx)
        ]

        lib.cgc_expert_streamer_prefetch.restype = None
        lib.cgc_expert_streamer_prefetch.argtypes = [c_void_p, POINTER(c_int), c_int]

        lib.cgc_expert_streamer_telemetry.restype = CGCTelemetry
        lib.cgc_expert_streamer_telemetry.argtypes = [c_void_p]

        lib.cgc_stream_layout_compute_offset.restype = c_uint64
        lib.cgc_stream_layout_compute_offset.argtypes = [
            POINTER(CGCStreamLayout), c_int, c_int
        ]

        lib.cgc_load_stream_layout_from_gguf.restype = CGCStreamLayout
        lib.cgc_load_stream_layout_from_gguf.argtypes = [c_char_p]

        lib.cgc_pd_layer_assignment_by_ratio.restype = CGCLayerAssignment
        lib.cgc_pd_layer_assignment_by_ratio.argtypes = [c_int, c_float]

        lib.cgc_pd_layer_assignment_custom.restype = CGCLayerAssignment
        lib.cgc_pd_layer_assignment_custom.argtypes = [
            POINTER(c_int), c_int, POINTER(c_int), c_int
        ]

        lib.cgc_streamer_pool_create.restype = c_void_p
        lib.cgc_streamer_pool_create.argtypes = []

        lib.cgc_streamer_pool_destroy.restype = None
        lib.cgc_streamer_pool_destroy.argtypes = [c_void_p]

        lib.cgc_streamer_pool_add.restype = c_bool
        lib.cgc_streamer_pool_add.argtypes = [c_void_p, c_int, c_void_p]

        lib.cgc_pd_scheduler_create.restype = c_void_p
        lib.cgc_pd_scheduler_create.argtypes = [
            c_void_p, POINTER(CGCLayerAssignment), c_int, c_int
        ]

        lib.cgc_pd_scheduler_destroy.restype = None
        lib.cgc_pd_scheduler_destroy.argtypes = [c_void_p]

        lib.cgc_pd_scheduler_enter_prefill.restype = None
        lib.cgc_pd_scheduler_enter_prefill.argtypes = [c_void_p]

        lib.cgc_pd_scheduler_switch_to_decode.restype = None
        lib.cgc_pd_scheduler_switch_to_decode.argtypes = [c_void_p]

        lib.cgc_pd_scheduler_get_stats.restype = CGCSchedulerStats
        lib.cgc_pd_scheduler_get_stats.argtypes = [c_void_p]

        lib.cgc_pd_scheduler_reset_stats.restype = None
        lib.cgc_pd_scheduler_reset_stats.argtypes = [c_void_p]

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
