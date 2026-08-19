import os
import sys
import ctypes
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DLL_PATH = os.path.join(BASE_DIR, "build", "cgc_expert_streamer.dll")
GGUF_PATH = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

from ctypes import (
    c_void_p, c_char_p, c_int, c_uint32, c_uint64, c_bool,
    POINTER, Structure, Array, c_float, c_double
)

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

class CGCCacheTelemetry(Structure):
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
    ]

class CGCLayerAssignment(Structure):
    _fields_ = [
        ("prefill_layers", c_int * 256),
        ("prefill_count", c_int),
        ("decode_layers", c_int * 256),
        ("decode_count", c_int),
        ("prefill_gpu", c_int),
        ("decode_gpu", c_int),
    ]

class CGCStats(Structure):
    _fields_ = [
        ("phase", c_int),
        ("gpu0_cache_count", c_int),
        ("gpu1_cache_count", c_int),
        ("gpu0_hit_rate", c_float),
        ("gpu1_hit_rate", c_float),
        ("prefill_tokens", c_uint64),
        ("decode_tokens", c_uint64),
        ("expert_loads", c_uint64),
        ("prefetch_hits", c_uint64),
        ("total_prefetch_time_nanos", c_uint64),
        ("total_load_time_nanos", c_uint64),
    ]

def setup_functions(lib):
    lib.cgc_expert_streamer_create.restype = c_void_p
    lib.cgc_expert_streamer_create.argtypes = [
        POINTER(CGCStreamLayout), c_int, c_bool, POINTER(c_int), c_int
    ]
    
    lib.cgc_expert_streamer_destroy.restype = None
    lib.cgc_expert_streamer_destroy.argtypes = [c_void_p]
    
    lib.cgc_expert_streamer_telemetry.restype = CGCCacheTelemetry
    lib.cgc_expert_streamer_telemetry.argtypes = [c_void_p]
    
    lib.cgc_stream_layout_compute_offset.restype = c_uint64
    lib.cgc_stream_layout_compute_offset.argtypes = [
        POINTER(CGCStreamLayout), c_int, c_int
    ]
    
    lib.cgc_streamer_pool_create.restype = c_void_p
    lib.cgc_streamer_pool_create.argtypes = []
    
    lib.cgc_streamer_pool_destroy.restype = None
    lib.cgc_streamer_pool_destroy.argtypes = [c_void_p]
    
    lib.cgc_streamer_pool_add.restype = c_bool
    lib.cgc_streamer_pool_add.argtypes = [c_void_p, c_int, c_void_p]
    
    lib.cgc_pd_layer_assignment_by_ratio.restype = CGCLayerAssignment
    lib.cgc_pd_layer_assignment_by_ratio.argtypes = [c_int, c_double]
    
    lib.cgc_pd_is_prefill_layer.restype = c_bool
    lib.cgc_pd_is_prefill_layer.argtypes = [POINTER(CGCLayerAssignment), c_int]
    
    lib.cgc_pd_is_decode_layer.restype = c_bool
    lib.cgc_pd_is_decode_layer.argtypes = [POINTER(CGCLayerAssignment), c_int]
    
    lib.cgc_pd_get_device_for_layer.restype = c_int
    lib.cgc_pd_get_device_for_layer.argtypes = [POINTER(CGCLayerAssignment), c_int]
    
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
    
    lib.cgc_pd_scheduler_get_stats.restype = CGCStats
    lib.cgc_pd_scheduler_get_stats.argtypes = [c_void_p]
    
    lib.cgc_pd_scheduler_reset_stats.restype = None
    lib.cgc_pd_scheduler_reset_stats.argtypes = [c_void_p]
    
    lib.cgc_ggml_type_bytes_per_elem.restype = c_double
    lib.cgc_ggml_type_bytes_per_elem.argtypes = [c_int]
    
    lib.cgc_gguf_lite_load.restype = c_void_p
    lib.cgc_gguf_lite_load.argtypes = [c_char_p]
    
    lib.cgc_gguf_lite_free.restype = None
    lib.cgc_gguf_lite_free.argtypes = [c_void_p]
    
    lib.cgc_gguf_lite_get_str.restype = c_char_p
    lib.cgc_gguf_lite_get_str.argtypes = [c_void_p, c_char_p]
    
    lib.cgc_gguf_lite_find_tensor.restype = c_int
    lib.cgc_gguf_lite_find_tensor.argtypes = [c_void_p, c_char_p]

def test_basic_streamer(lib):
    print("\n" + "="*60)
    print("TEST 1: Basic Streamer Creation/Destruction")
    print("="*60)
    
    test_file = os.path.join(BASE_DIR, "test_py_stream.bin")
    with open(test_file, "wb") as f:
        f.write(b"\x00" * 4096)
    
    layout = CGCStreamLayout()
    layout.path = test_file.encode("utf-8")
    layout.stream_offset = 0
    layout.stream_size = 4096
    layout.experts_per_layer = 4
    layout.expert_stride = 1024
    layout.has_explicit_offsets = 0
    
    streamer = lib.cgc_expert_streamer_create(
        ctypes.byref(layout), 2, False, None, 0
    )
    
    assert streamer is not None, "Streamer creation failed!"
    print(f"  ✅ Streamer created: {streamer}")
    
    telemetry = lib.cgc_expert_streamer_telemetry(streamer)
    print(f"  ✅ Telemetry: slot_count={telemetry.slot_count}")
    assert telemetry.slot_count == 2, f"Expected slot_count=2, got {telemetry.slot_count}"
    
    lib.cgc_expert_streamer_destroy(streamer)
    print(f"  ✅ Streamer destroyed")
    
    os.remove(test_file)
    print("  PASS\n")
    return True

def test_layout_offsets(lib):
    print("\n" + "="*60)
    print("TEST 2: Layout Offset Computation")
    print("="*60)
    
    layout = CGCStreamLayout()
    layout.stream_offset = 1000
    layout.expert_stride = 256
    layout.experts_per_layer = 4
    layout.has_explicit_offsets = 0
    
    off0 = lib.cgc_stream_layout_compute_offset(ctypes.byref(layout), 0, 0)
    off1 = lib.cgc_stream_layout_compute_offset(ctypes.byref(layout), 0, 1)
    off2 = lib.cgc_stream_layout_compute_offset(ctypes.byref(layout), 0, 2)
    
    assert off0 == 1000, f"Expected 1000, got {off0}"
    assert off1 == 1256, f"Expected 1256, got {off1}"
    assert off2 == 1512, f"Expected 1512, got {off2}"
    print(f"  ✅ Expert offsets: {off0}, {off1}, {off2}")
    
    off_l1_e0 = lib.cgc_stream_layout_compute_offset(ctypes.byref(layout), 1, 0)
    per_layer = 4 * 256
    assert off_l1_e0 == 1000 + per_layer, f"Expected {1000+per_layer}, got {off_l1_e0}"
    print(f"  ✅ Layer 1 expert 0 offset: {off_l1_e0}")
    
    print("  PASS\n")
    return True

def test_pool_management(lib):
    print("\n" + "="*60)
    print("TEST 3: Pool Management")
    print("="*60)
    
    pool = lib.cgc_streamer_pool_create()
    assert pool is not None, "Pool creation failed!"
    print(f"  ✅ Pool created: {pool}")
    
    test_file = os.path.join(BASE_DIR, "test_py_pool.bin")
    with open(test_file, "wb") as f:
        f.write(b"\x00" * 4096)
    
    layout = CGCStreamLayout()
    layout.path = test_file.encode("utf-8")
    layout.experts_per_layer = 4
    layout.expert_stride = 256
    
    streamer = lib.cgc_expert_streamer_create(ctypes.byref(layout), 2, False, None, 0)
    assert streamer is not None
    
    ok = lib.cgc_streamer_pool_add(pool, 5, streamer)
    assert ok == True, "Pool add failed!"
    print(f"  ✅ Added streamer for layer 5")
    
    lib.cgc_streamer_pool_destroy(pool)
    lib.cgc_expert_streamer_destroy(streamer)
    os.remove(test_file)
    print(f"  ✅ Pool and streamer destroyed")
    
    print("  PASS\n")
    return True

def test_pd_scheduler(lib):
    print("\n" + "="*60)
    print("TEST 4: PD Scheduler (Prefill/Decode)")
    print("="*60)
    
    pool = lib.cgc_streamer_pool_create()
    assert pool is not None
    
    assignment = lib.cgc_pd_layer_assignment_by_ratio(12, 0.5)
    print(f"  ✅ Assignment: prefill={assignment.prefill_count}, decode={assignment.decode_count}")
    assert assignment.prefill_count + assignment.decode_count == 12
    
    is_pf = lib.cgc_pd_is_prefill_layer(ctypes.byref(assignment), 0)
    is_dec = lib.cgc_pd_is_decode_layer(ctypes.byref(assignment), 0)
    assert is_pf == True, "Layer 0 should be prefill layer"
    assert is_dec == False, "Layer 0 should not be decode layer"
    print(f"  ✅ Layer 0: prefill={is_pf}, decode={is_dec}")
    
    device = lib.cgc_pd_get_device_for_layer(ctypes.byref(assignment), 0)
    print(f"  ✅ Layer 0 device: {device}")
    assert device == assignment.prefill_gpu
    
    scheduler = lib.cgc_pd_scheduler_create(pool, ctypes.byref(assignment), 8, 4)
    assert scheduler is not None
    print(f"  ✅ Scheduler created")
    
    lib.cgc_pd_scheduler_enter_prefill(scheduler)
    stats = lib.cgc_pd_scheduler_get_stats(scheduler)
    print(f"  ✅ Prefill phase entered: phase={stats.phase}")
    assert stats.phase == 1
    
    lib.cgc_pd_scheduler_switch_to_decode(scheduler)
    stats = lib.cgc_pd_scheduler_get_stats(scheduler)
    print(f"  ✅ Decode phase entered: phase={stats.phase}")
    assert stats.phase == 2
    
    lib.cgc_pd_scheduler_reset_stats(scheduler)
    stats = lib.cgc_pd_scheduler_get_stats(scheduler)
    print(f"  ✅ Stats reset: prefill_tokens={stats.prefill_tokens}")
    assert stats.prefill_tokens == 0
    
    lib.cgc_pd_scheduler_destroy(scheduler)
    lib.cgc_streamer_pool_destroy(pool)
    print(f"  ✅ Scheduler and pool destroyed")
    
    print("  PASS\n")
    return True

def test_ggml_type_bytes(lib):
    print("\n" + "="*60)
    print("TEST 5: GGML Type Bytes Per Element")
    print("="*60)
    
    F32 = 0
    F16 = 1
    BF16 = 30
    Q4_K = 12
    
    f32_bytes = lib.cgc_ggml_type_bytes_per_elem(F32)
    f16_bytes = lib.cgc_ggml_type_bytes_per_elem(F16)
    bf16_bytes = lib.cgc_ggml_type_bytes_per_elem(BF16)
    q4_bytes = lib.cgc_ggml_type_bytes_per_elem(Q4_K)
    
    print(f"  F32: {f32_bytes} bytes/elem")
    print(f"  F16: {f16_bytes} bytes/elem")
    print(f"  BF16: {bf16_bytes} bytes/elem")
    print(f"  Q4_K: {q4_bytes} bytes/elem")
    
    assert f32_bytes == 4.0, f"Expected 4.0, got {f32_bytes}"
    assert f16_bytes == 2.0, f"Expected 2.0, got {f16_bytes}"
    assert bf16_bytes == 2.0, f"Expected 2.0, got {bf16_bytes}"
    assert q4_bytes > 0, f"Expected > 0, got {q4_bytes}"
    
    print("  PASS\n")
    return True

def test_gguf_lite_parsing(lib, gguf_path):
    print("\n" + "="*60)
    print("TEST 6: GGUF Lite Header Parsing")
    print("="*60)
    
    if not os.path.exists(gguf_path):
        print(f"  ⚠️  GGUF file not found: {gguf_path}")
        print("  SKIP (no test file)\n")
        return True
    
    print(f"  Loading: {os.path.basename(gguf_path)}")
    t0 = time.time()
    ctx = lib.cgc_gguf_lite_load(gguf_path.encode("utf-8"))
    t1 = time.time()
    
    if ctx is None:
        print(f"  ❌ Failed to load GGUF file!")
        return False
    
    print(f"  ✅ GGUF header loaded in {t1-t0:.2f}s")
    
    arch = lib.cgc_gguf_lite_get_str(ctx, b"general.architecture")
    if arch:
        print(f"  ✅ Architecture: {arch.decode('utf-8')}")
    
    expert_count = lib.cgc_gguf_lite_get_str(ctx, b"gemma4.expert_count")
    if expert_count:
        print(f"  ✅ Expert count: {expert_count.decode('utf-8')}")
    
    block_count = lib.cgc_gguf_lite_get_str(ctx, b"gemma4.block_count")
    if block_count:
        print(f"  ✅ Block count: {block_count.decode('utf-8')}")
    
    hidden_size = lib.cgc_gguf_lite_get_str(ctx, b"gemma4.hidden_size")
    if hidden_size:
        print(f"  ✅ Hidden size: {hidden_size.decode('utf-8')}")
    
    lib.cgc_gguf_lite_free(ctx)
    print(f"  ✅ Context freed")
    print("  PASS\n")
    return True

def main():
    print("="*60)
    print("CGC Expert Streaming - Python ctypes Validation")
    print("="*60)
    
    if not os.path.exists(DLL_PATH):
        print(f"\n❌ DLL not found: {DLL_PATH}")
        print("Please build first: gcc -shared -o cgc_expert_streamer.dll ...")
        return 1
    
    print(f"\nLoading: {DLL_PATH}")
    lib = ctypes.CDLL(DLL_PATH)
    setup_functions(lib)
    print("✅ Library loaded and functions configured")
    
    tests = [
        ("Basic Streamer", test_basic_streamer, [lib]),
        ("Layout Offsets", test_layout_offsets, [lib]),
        ("Pool Management", test_pool_management, [lib]),
        ("PD Scheduler", test_pd_scheduler, [lib]),
        ("GGML Type Bytes", test_ggml_type_bytes, [lib]),
        ("GGUF Lite Parsing", test_gguf_lite_parsing, [lib, GGUF_PATH]),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_fn, args in tests:
        try:
            result = test_fn(*args)
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("="*60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())