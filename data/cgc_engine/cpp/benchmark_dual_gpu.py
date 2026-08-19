"""
PD Separation + Dual-GPU Load Balance Benchmark
Using Vulkan-enabled llama-bench from D:\\alex\\toolchains\\llama-build

Devices:
  Vulkan0: Intel(R) UHD Graphics (4138 MiB) - Prefill
  Vulkan1: NVIDIA MX250 (1983 MiB) - Decode

PD Strategy:
  - Prefill: First 20 layers on GPU 0 (Intel UHD)
  - Decode: Last 20 layers on GPU 1 (NVIDIA MX250)
"""

import subprocess
import json
import re
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
LLAMA_BENCH = r"D:\alex\toolchains\llama-build\bin\llama-bench.exe"
OUTPUT_DIR = Path(r"D:\alex\flashkv0516\bench_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set PATH for Vulkan runtime
os.environ['PATH'] = (
    r"D:\alex\toolchains\VulkanSDK\Bin;" +
    r"D:\alex\toolchains\winlibs-gcc162\mingw64\bin;" +
    os.environ.get('PATH', '')
)

def run_benchmark(n_prompt, n_gen, device='cpu', n_threads=8, extra_args=None):
    """Run llama-bench and return parsed metrics."""
    args = [
        LLAMA_BENCH,
        '-m', MODEL_PATH,
        '-p', str(n_prompt),
        '-n', str(n_gen),
        '-t', str(n_threads),
        '-r', '3',  # 3 repetitions
    ]
    
    if device == 'cpu':
        args.extend(['-ngl', '0'])
    elif device:
        args.extend(['-ngl', '99', '-dev', device])
    
    if extra_args:
        args.extend(extra_args)
    
    print(f"  Running on {device}: {' '.join(args[:8])}...")
    
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=900,
        cwd=os.path.dirname(LLAMA_BENCH)
    )
    
    return result.stdout + result.stderr

def parse_bench_output(output):
    """Parse benchmark output for key metrics."""
    metrics = {
        'prompt_tokens': None,
        'prompt_eval_ms': None,
        'prompt_eval_tps': None,
        'gen_tokens': None,
        'eval_ms': None,
        'eval_tps': None,
        'total_time': None,
    }
    
    # Parse markdown table output
    lines = output.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        if 'prompt eval' in line_lower:
            m = re.search(r'(\d+\.?\d*)\s+tokens/s', line)
            if m:
                metrics['prompt_eval_tps'] = float(m.group(1))
            m = re.search(r'(\d+\.?\d*)\s+ms', line)
            if m:
                metrics['prompt_eval_ms'] = float(m.group(1))
                
        if ('eval time' in line_lower or 
            ('generation' in line_lower and 'prompt' not in line_lower)):
            m = re.search(r'(\d+\.?\d*)\s+tokens/s', line)
            if m:
                metrics['eval_tps'] = float(m.group(1))
            m = re.search(r'(\d+\.?\d*)\s+ms', line)
            if m:
                metrics['eval_ms'] = float(m.group(1))
    
    return metrics

def main():
    print("=" * 70)
    print("PD SEPARATION + DUAL-GPU LOAD BALANCE BENCHMARK")
    print("Model: Qwen3.6-35B-A3B-UD-IQ3_XXS")
    print("=" * 70)
    
    # Phase 0: Verify devices
    print("\n[Phase 0] Device Verification")
    print("-" * 50)
    
    output = subprocess.run(
        [LLAMA_BENCH, '--list-devices'],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.dirname(LLAMA_BENCH),
        env=os.environ
    )
    print(output.stdout)
    
    # Phase 1: CPU baseline
    print("\n[Phase 1] CPU Baseline")
    print("-" * 50)
    
    cpu_results = {}
    for n_prompt, n_gen, label in [(512, 64, "512 prompt"), (1024, 64, "1024 prompt"), (2048, 32, "2048 prompt")]:
        print(f"\n  [{label}]")
        try:
            output = run_benchmark(n_prompt, n_gen, 'cpu')
            metrics = parse_bench_output(output)
            cpu_results[label] = metrics
            print(f"    Prompt eval: {metrics.get('prompt_eval_tps', '?')} tps")
            print(f"    Token gen:   {metrics.get('eval_tps', '?')} tps")
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT")
            cpu_results[label] = {'error': 'timeout'}
        except Exception as e:
            print(f"    ERROR: {e}")
            cpu_results[label] = {'error': str(e)}
    
    # Phase 2: GPU 0 (Intel UHD) benchmark
    print("\n[Phase 2] GPU 0 - Intel UHD Graphics")
    print("-" * 50)
    
    gpu0_results = {}
    for n_prompt, n_gen, label in [(512, 64, "512 prompt"), (1024, 64, "1024 prompt"), (2048, 32, "2048 prompt")]:
        print(f"\n  [{label}]")
        try:
            output = run_benchmark(n_prompt, n_gen, 'Vulkan0')
            metrics = parse_bench_output(output)
            gpu0_results[label] = metrics
            print(f"    Prompt eval: {metrics.get('prompt_eval_tps', '?')} tps")
            print(f"    Token gen:   {metrics.get('eval_tps', '?')} tps")
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT")
            gpu0_results[label] = {'error': 'timeout'}
        except Exception as e:
            print(f"    ERROR: {e}")
            gpu0_results[label] = {'error': str(e)}
    
    # Phase 3: GPU 1 (NVIDIA MX250) benchmark
    print("\n[Phase 3] GPU 1 - NVIDIA MX250")
    print("-" * 50)
    
    gpu1_results = {}
    for n_prompt, n_gen, label in [(512, 64, "512 prompt"), (1024, 64, "1024 prompt"), (2048, 32, "2048 prompt")]:
        print(f"\n  [{label}]")
        try:
            output = run_benchmark(n_prompt, n_gen, 'Vulkan1')
            metrics = parse_bench_output(output)
            gpu1_results[label] = metrics
            print(f"    Prompt eval: {metrics.get('prompt_eval_tps', '?')} tps")
            print(f"    Token gen:   {metrics.get('eval_tps', '?')} tps")
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT")
            gpu1_results[label] = {'error': 'timeout'}
        except Exception as e:
            print(f"    ERROR: {e}")
            gpu1_results[label] = {'error': str(e)}
    
    # Phase 4: PD Split benchmark (all layers to each GPU separately)
    print("\n[Phase 4] Full Model on Single GPU (Baseline)")
    print("-" * 50)
    
    full_gpu_results = {}
    for dev in ['Vulkan0', 'Vulkan1']:
        print(f"\n  Full model on {dev}:")
        try:
            output = run_benchmark(512, 64, dev)
            metrics = parse_bench_output(output)
            full_gpu_results[dev] = metrics
            print(f"    Prompt eval: {metrics.get('prompt_eval_tps', '?')} tps")
            print(f"    Token gen:   {metrics.get('eval_tps', '?')} tps")
        except Exception as e:
            print(f"    ERROR: {e}")
            full_gpu_results[dev] = {'error': str(e)}
    
    # Phase 5: Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n  CPU Baseline:")
    for k, v in cpu_results.items():
        if 'error' not in v:
            print(f"    {k}: prompt={v.get('prompt_eval_tps', '?')} tps, gen={v.get('eval_tps', '?')} tps")
    
    print("\n  GPU 0 (Intel UHD):")
    for k, v in gpu0_results.items():
        if 'error' not in v:
            print(f"    {k}: prompt={v.get('prompt_eval_tps', '?')} tps, gen={v.get('eval_tps', '?')} tps")
    
    print("\n  GPU 1 (NVIDIA MX250):")
    for k, v in gpu1_results.items():
        if 'error' not in v:
            print(f"    {k}: prompt={v.get('prompt_eval_tps', '?')} tps, gen={v.get('eval_tps', '?')} tps")
    
    print("\n  Full Model on Single GPU:")
    for k, v in full_gpu_results.items():
        if 'error' not in v:
            print(f"    {k}: prompt={v.get('prompt_eval_tps', '?')} tps, gen={v.get('eval_tps', '?')} tps")
    
    # Save all results
    results = {
        'model': 'Qwen3.6-35B-A3B-UD-IQ3_XXS',
        'devices': {
            'Vulkan0': 'Intel(R) UHD Graphics (4138 MiB)',
            'Vulkan1': 'NVIDIA MX250 (1983 MiB)',
        },
        'cpu_results': cpu_results,
        'gpu0_results': gpu0_results,
        'gpu1_results': gpu1_results,
        'full_gpu_results': full_gpu_results,
        'pd_architecture': {
            'prefill_device': 'Vulkan0 (Intel UHD)',
            'decode_device': 'Vulkan1 (NVIDIA MX250)',
            'prefill_layers': 20,
            'decode_layers': 20,
            'notes': 'PD separation requires custom kernel to split layers between devices',
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    output_file = OUTPUT_DIR / 'qwen36_pd_dual_gpu_bench.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {output_file}")
    print(f"\n  ✅ Benchmark complete!")

if __name__ == '__main__':
    main()
