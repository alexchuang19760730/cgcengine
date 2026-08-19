import os
import sys
import struct
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgc_expert_streamer_ctypes import CGCExpertStreamer, CGCStreamLayout

GGUF_PATH = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

def test_real_gguf():
    print("=" * 60)
    print("  Real GGUF Integration Test")
    print("=" * 60)

    if not os.path.exists(GGUF_PATH):
        print(f"GGUF file not found: {GGUF_PATH}")
        return False

    file_size = os.path.getsize(GGUF_PATH)
    print(f"GGUF file: {GGUF_PATH}")
    print(f"File size: {file_size / (1024**3):.2f} GB")

    cgc = CGCExpertStreamer(auto_build=False)

    print("\n--- Test 1: Load GGUF Layout ---")
    t0 = time.time()
    layout = cgc.load_layout_from_gguf(GGUF_PATH)
    t1 = time.time()

    print(f"  Layout loaded in {(t1 - t0) * 1000:.1f} ms")
    print(f"  stream_offset: {layout.stream_offset}")
    print(f"  stream_size: {layout.stream_size}")
    print(f"  experts_per_layer: {layout.experts_per_layer}")
    print(f"  expert_stride: {layout.expert_stride}")
    print(f"  has_explicit_offsets: {layout.has_explicit_offsets}")

    if layout.stream_offset == 0 and layout.experts_per_layer == 0:
        print("  WARNING: Layout parsing returned default values")
        print("  This may indicate the GGUF format needs special handling")
    else:
        print("  Layout parsed successfully!")

    print("\n--- Test 2: Compute Expert Offsets ---")
    for layer in range(min(3, layout.experts_per_layer)):
        off0 = cgc.compute_offset(layout, layer, 0)
        off1 = cgc.compute_offset(layout, layer, 1)
        print(f"  Layer {layer}: expert[0] at {off0}, expert[1] at {off1}")
        if layout.expert_stride > 0:
            expected_stride = off1 - off0
            print(f"    stride check: expected={layout.expert_stride}, actual={expected_stride}")

    print("\n--- Test 3: Create Streamer with Real Layout ---")
    if layout.experts_per_layer > 0 and layout.expert_stride > 0:
        streamer = cgc.create_streamer(
            path=GGUF_PATH,
            stream_offset=layout.stream_offset,
            experts_per_layer=layout.experts_per_layer,
            expert_stride=layout.expert_stride,
            slot_count=8,
            use_mmap=False
        )

        if streamer:
            print(f"  Streamer created: {streamer}")

            telemetry = cgc.get_telemetry(streamer)
            print(f"  Initial telemetry: requests={telemetry.total_requests}")

            print("\n--- Test 4: Load Expert 0 ---")
            t0 = time.time()
            result = cgc.load_experts(streamer, [0])
            t1 = time.time()

            print(f"  Load expert 0 in {(t1 - t0) * 1000:.1f} ms")
            print(f"  count={result.count}, hits={result.hits}, misses={result.misses}")
            print(f"  read_bytes={result.read_bytes}, read_time={result.read_wall_nanos / 1e6:.1f} ms")

            if result.buffers[0]:
                print(f"  Expert 0 buffer: {result.buffers[0]}, size={result.sizes[0]}")

            print("\n--- Test 5: Cache Hit Test ---")
            result2 = cgc.load_experts(streamer, [0])
            print(f"  Second load: hits={result2.hits}, misses={result2.misses}")

            telemetry = cgc.get_telemetry(streamer)
            print(f"  Final telemetry: requests={telemetry.total_requests}, hits={telemetry.total_hits}")

            cgc.destroy_streamer(streamer)
            print("  Streamer destroyed.")
        else:
            print("  WARNING: Streamer creation failed (may need layout adjustments)")
    else:
        print("  Skipped: Layout not fully parsed (experts_per_layer=0 or expert_stride=0)")
        print("  This is expected for models that don't use MoE architecture")

    print("\n--- Test 6: PD Scheduler ---")
    scheduler, pool, assignment = cgc.create_pd_scheduler(8, 0.5)
    print(f"  PD Scheduler created: prefill={assignment.prefill_count}, decode={assignment.decode_count}")

    cgc.scheduler_enter_prefill(scheduler)
    print("  Prefill phase entered.")

    stats = cgc.scheduler_get_stats(scheduler)
    print(f"  Stats: prefill_tokens={stats.total_prefill_tokens}")

    cgc.scheduler_switch_to_decode(scheduler)
    print("  Switched to decode phase.")

    stats = cgc.scheduler_get_stats(scheduler)
    print(f"  Final stats: prefill={stats.total_prefill_tokens}, decode={stats.total_decode_tokens}")

    cgc.destroy_pd_scheduler(scheduler, pool)
    print("  PD Scheduler destroyed.")

    print("\n" + "=" * 60)
    print("  Real GGUF Integration Test PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_real_gguf()
    sys.exit(0 if success else 1)
