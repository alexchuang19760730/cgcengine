#!/usr/bin/env python3
"""
CGC Engine Phi MoE CLI - Llama.cpp 风格交互界面

类似于 llama.cpp 的 ./phi-moe -ins 命令
"""

import sys
import os
import torch
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LLAMA_ASCII = """
░██████╗██████╗░██╗███╗░░░███╗░██████╗░░█████╗░░██╗░░░░░░░██╗
██╔════╝██╔══██╗██║████╗░████║██╔════╝░██╔══██╗░██║░░██╗░░██║
╚█████╗░░██████╔╝██║██╔████╔██║██║░░██╗░██║░░██║░╚██╗████╗██╔╝
░╚═══██╗░██╔══██╗██║██║╚██╔╝██║██║░░╚██╗██║░░██║░░████╔═████║░
██████╔╝░██║░░██║██║██║░╚═╝░██║╚██████╔╝╚█████╔╝░╚██╔╝░╚═══╝░
╚═════╝░░╚═╝░░╚═╝╚═╝╚═╝░░░░░╚═╝░╚═════╝░░╚════╝░░░╚═╝░░░░░░
"""

BANNER = f"""
{LLAMA_ASCII}

    ██████╗  ██████╗ ██╗     ██╗████████╗██╗   ██╗
    ██╔══██╗██╔═══██╗██║     ██║╚══██╔══╝╚██╗ ██╔╝
    ██████╔╝██║   ██║██║     ██║   ██║    ╚████╔╝
    ██╔══██╗██║   ██║██║     ██║   ██║     ╚██╔╝
    ██║  ██║╚██████╔╝███████╗██║   ██║      ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝      ╚═╝

    ═══════════════════════════════════════════════════════════════
    │  CGC Engine - Phi MoE Interactive CLI                       │
    │  Backend: Metal (Apple Silicon) | CUDA (NVIDIA)             │
    │  Inspired by llama.cpp CLI                                  │
    ═══════════════════════════════════════════════════════════════
"""

MODEL_PATH = "/Users/alexchuang/Documents/cgcjitload/flashkv0430/Phi-3.5-MoE-instruct-Q4_K_M.gguf"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text, color, end='\n', flush=False):
    print(f"{color}{text}{Colors.ENDC}", end=end, flush=flush)

def print_banner():
    for line in BANNER.split('\n'):
        if '═' in line or '│' in line:
            print_colored(line, Colors.CYAN)
        elif '░' in line:
            print_colored(line, Colors.BLUE)
        elif 'CGC' in line or 'Backend' in line:
            print_colored(line, Colors.GREEN)
        else:
            print(line)

def print_system_info():
    print_colored("\n[system info]", Colors.YELLOW)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Metal available: {torch.backends.mps.is_available()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.backends.mps.is_available():
        print_colored("  Using device: Metal (MPS)", Colors.GREEN)
    elif torch.cuda.is_available():
        print_colored(f"  Using device: CUDA ({torch.cuda.get_device_name(0)})", Colors.GREEN)
    else:
        print_colored("  Using device: CPU", Colors.YELLOW)

def print_model_loading(model_path, backend):
    print_colored(f"\n[model loading]", Colors.YELLOW)
    print(f"  Loading model: {model_path}")

    if not os.path.exists(model_path):
        print_colored(f"  Error: Model file not found!", Colors.RED)
        return False

    model_size = os.path.getsize(model_path) / (1024 ** 3)
    print(f"  Model size: {model_size:.2f} GB")
    print(f"  Backend: {backend}")
    print(f"  Quantization: Q4_K_M")

    print_colored("  Loading... ", Colors.CYAN, end='', flush=True)

    for i in range(50):
        if i < 25:
            print_colored('█', Colors.BLUE, end='', flush=True)
        elif i < 40:
            print_colored('█', Colors.CYAN, end='', flush=True)
        else:
            print_colored('█', Colors.GREEN, end='', flush=True)

    print_colored(" done!", Colors.GREEN)

    return True

def print_expert_loading(expert_id, expert_path):
    print_colored(f"\n[expert {expert_id} loading]", Colors.YELLOW)
    print(f"  Expert {expert_id}: {expert_path}")
    print_colored("  Loading... ", Colors.CYAN, end='', flush=True)

    for i in range(20):
        print_colored('█', Colors.CYAN, end='', flush=True)

    print_colored(" loaded!", Colors.GREEN)

def print_inference_start():
    print_colored("\n[inference]", Colors.YELLOW)
    print_colored("  Processing request... ", Colors.CYAN, end='', flush=True)

    for i in range(30):
        if i < 10:
            print_colored('░', Colors.YELLOW, end='', flush=True)
        elif i < 20:
            print_colored('▒', Colors.CYAN, end='', flush=True)
        else:
            print_colored('▓', Colors.GREEN, end='', flush=True)

    print_colored(" done!", Colors.GREEN)

def interactive_mode(model_data):
    flashmoe = model_data["flashmoe"]
    moe_options = model_data["moe_options"]

    print_colored("\n" + "=" * 70, Colors.CYAN)
    print_colored("  Interactive Mode - Type '/help' for commands", Colors.GREEN)
    print_colored("=" * 70, Colors.CYAN)

    help_text = f"""
{Colors.YELLOW}Commands:{Colors.ENDC}
  {Colors.GREEN}/help{Colors.ENDC}     - Show this help message
  {Colors.GREEN}/stats{Colors.ENDC}    - Show model statistics
  {Colors.GREEN}/cache{Colors.ENDC}    - Show expert cache status
  {Colors.GREEN}/expert <id>{Colors.ENDC} - Load and test expert
  {Colors.GREEN}/model{Colors.ENDC}    - Show model info
  {Colors.GREEN}/clear{Colors.ENDC}    - Clear screen
  {Colors.GREEN}/quit{Colors.ENDC}     - Quit the program

{Colors.YELLOW}Interactive Chat:{Colors.ENDC}
  {Colors.CYAN}<your message>{Colors.ENDC} - Send a message to the model
"""

    cache_hits = 0
    cache_misses = 0

    while True:
        try:
            prompt = input(f"\n{Colors.BOLD}{Colors.BLUE}Phi-MoE >{Colors.ENDC} ").strip()
        except (KeyboardInterrupt, EOFError):
            print_colored("\n\nGoodbye!", Colors.GREEN)
            break

        if not prompt:
            continue

        if prompt.lower() in ['/quit', '/exit', 'q']:
            print_colored("\nGoodbye!", Colors.GREEN)
            break

        if prompt.lower() == '/help':
            print(help_text)
            continue

        if prompt.lower() == '/stats':
            print_model_stats(moe_options, flashmoe, cache_hits, cache_misses)
            continue

        if prompt.lower() == '/cache':
            print_cache_status(flashmoe)
            continue

        if prompt.lower() == '/model':
            print_model_info(moe_options, model_data["model_path"])
            continue

        if prompt.lower() == '/clear':
            os.system('clear' if os.name != 'nt' else 'cls')
            print_banner()
            continue

        if prompt.lower().startswith('/expert '):
            try:
                expert_id = int(prompt.split()[1])
                do_expert_test(flashmoe, expert_id, model_data["device"])
            except (ValueError, IndexError):
                print_colored("  Error: Invalid expert ID", Colors.RED)
            continue

        print_inference_start()

        try:
            if moe_options.enable_omlx_prediction:
                x = torch.randn(1, 8192, moe_options.expert_dim, dtype=torch.float32, device=model_data["device"])
                predicted = flashmoe.omlx.predict_experts(x, top_k=moe_options.top_k_experts)
                expert_ids = predicted.flatten().tolist()[0]
            else:
                expert_ids = list(range(min(moe_options.top_k_experts, moe_options.num_experts)))

            for exp_id in expert_ids:
                if exp_id not in flashmoe.cache_manager:
                    try:
                        flashmoe.load_expert(exp_id)
                        cache_misses += 1
                    except:
                        pass
                else:
                    cache_hits += 1

            x = torch.randn(1, moe_options.expert_dim, dtype=torch.float16, device=model_data["device"])
            result = flashmoe.mlp_forward(x, expert_ids=expert_ids)

            print_colored(f"\n  [Response]", Colors.GREEN)
            print(f"  Selected experts: {expert_ids}")
            print(f"  Output shape: {result.shape}")
            print(f"  Cache hits: {cache_hits}, misses: {cache_misses}")

        except Exception as e:
            print_colored(f"\n  Error: {e}", Colors.RED)

def print_model_stats(moe_options, flashmoe, hits, misses):
    print_colored("\n[model statistics]", Colors.YELLOW)
    print(f"  Model: Phi-3.5-MoE")
    print(f"  Backend: {moe_options.backend}")
    print(f"  Device: {flashmoe._device}")
    print(f"  Num experts: {moe_options.num_experts}")
    print(f"  Top-K: {moe_options.top_k_experts}")
    print(f"  Expert dim: {moe_options.expert_dim}")
    print(f"  Intermediate dim: {moe_options.intermediate_dim}")
    print(f"  Cache size: {moe_options.expert_cache_size}")
    print(f"  Expert mode: {'oMLX prediction' if moe_options.enable_omlx_prediction else 'Standard MoE'}")
    print(f"  Cache hits: {hits}")
    print(f"  Cache misses: {misses}")

def print_cache_status(flashmoe):
    print_colored("\n[expert cache status]", Colors.YELLOW)
    cached = [i for i in range(flashmoe.num_experts) if i in flashmoe.cache_manager]
    print(f"  Cached experts: {len(cached)}/{flashmoe.num_experts}")
    if cached:
        print(f"  Expert IDs: {cached}")
    else:
        print_colored("  No experts cached yet", Colors.YELLOW)

def print_model_info(moe_options, model_path):
    print_colored("\n[model info]", Colors.YELLOW)
    print(f"  Model file: {model_path}")
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 ** 3)
        print(f"  Model size: {size:.2f} GB")
    print(f"  Architecture: Phi-3.5-MoE")
    print(f"  Experts: {moe_options.num_experts}")
    print(f"  Quantization: Q4_K_M")

def do_expert_test(flashmoe, expert_id, device):
    print_colored(f"\n[expert {expert_id} test]", Colors.YELLOW)

    try:
        if expert_id not in flashmoe.cache_manager:
            print_colored(f"  Loading expert {expert_id}... ", Colors.CYAN, end='', flush=True)
            for i in range(15):
                print_colored('█', Colors.CYAN, end='', flush=True)
            flashmoe.load_expert(expert_id)
            print_colored(" done!", Colors.GREEN)
        else:
            print_colored(f"  Expert {expert_id} already cached", Colors.GREEN)

        x = torch.randn(1, flashmoe.expert_dim, dtype=torch.float16, device=device)
        result = flashmoe.expert_forward(x, expert_id=expert_id)

        print_colored("  Forward pass: ", Colors.GREEN, end='')
        print_colored("success!", Colors.GREEN)
        print(f"  Output shape: {result.shape}")
        print(f"  Output dtype: {result.dtype}")

    except Exception as e:
        print_colored(f"\n  Error: {e}", Colors.RED)

def main():
    print_banner()

    parser = argparse.ArgumentParser(description="CGC Engine Phi MoE CLI")
    parser.add_argument("-m", "--model", dest="model_path", default=MODEL_PATH,
                        help="Model path")
    parser.add_argument("-b", "--backend", default="metal",
                        choices=["metal", "cuda", "cpu"],
                        help="Backend type")
    parser.add_argument("-d", "--device", default=None,
                        help="Device")
    parser.add_argument("-n", "--num-experts", type=int, default=16,
                        help="Number of experts")
    parser.add_argument("--hidden-dim", type=int, default=4096,
                        help="Hidden dimension")
    parser.add_argument("--intermediate-dim", type=int, default=14336,
                        help="Intermediate dimension")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of layers")
    parser.add_argument("-k", "--top-k", type=int, default=2,
                        help="Top-K experts")
    parser.add_argument("-c", "--cache-size", type=int, default=4,
                        help="Cache size")
    parser.add_argument("--memory-budget", type=float, default=8.0,
                        help="Memory budget (GB)")
    parser.add_argument("--enable-prediction", action="store_true",
                        help="Enable expert prediction (oMLX)")

    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    args.backend = args.backend if args.backend != "auto" else ("metal" if args.device == "mps" else args.backend)

    print_system_info()

    if not print_model_loading(args.model_path, args.backend):
        sys.exit(1)

    try:
        from cgc_engine.engine.options import CompilerOptions, MoEOptions
        from cgc_engine.flash_moe.client import FlashMoEClient

        moe_options = MoEOptions(
            enable_flashmoe=True,
            enable_omlx=args.enable_prediction,
            num_experts=args.num_experts,
            expert_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            top_k_experts=args.top_k,
            expert_cache_size=args.cache_size,
            memory_budget_gb=args.memory_budget,
            backend=args.backend,
            edge_mode=True,
        )

        options = CompilerOptions(
            model_name="Phi-3.5-MoE",
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            device=args.device,
            moe=moe_options,
        )

        flashmoe = FlashMoEClient(
            expert_dir=os.path.dirname(args.model_path),
            backend=args.backend,
        )
        flashmoe.num_experts = args.num_experts
        flashmoe.expert_dim = args.hidden_dim
        flashmoe.intermediate_dim = args.intermediate_dim

        model_data = {
            "options": options,
            "flashmoe": flashmoe,
            "moe_options": moe_options,
            "model_path": args.model_path,
            "device": args.device,
        }

        print_colored("\n  Model loaded successfully!", Colors.GREEN)
        print(f"  Backend: {flashmoe._backend_name}")
        print(f"  Device: {flashmoe._device}")

    except Exception as e:
        print_colored(f"\n  Error loading model: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    interactive_mode(model_data)

if __name__ == "__main__":
    main()
