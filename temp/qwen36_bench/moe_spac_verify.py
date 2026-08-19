#!/usr/bin/env python3
"""
MoE-SpAc verification (simplified).

Core question: how well does the CURRENT step's routing predict the NEXT
step's routing? This is the MoE-SpAc utility estimator's lower bound
(actual MTP draft tokens would have their own routing, but current hidden
state is a conservative proxy).

Metrics:
- step-to-step routing Jaccard (overlap fraction)
- MTP utility recall: |current_routing ∩ next_routing| / |next_routing|
- Static top-64 recall:  |top64_profile ∩ next_routing| / |next_routing|
- Step-to-step expert set size stability
"""
import json, sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE = torch.bfloat16
DEVICE = "cpu"
HF_DIR = "/Volumes/AlexZhuang/qwen36-hf"
PROFILE_FILE = os.path.join(os.path.dirname(__file__), 'qwen36_top64.json')
OUTPUT = os.path.join(os.path.dirname(__file__), 'moe_spac_result.json')
PROMPTS = {
    "code":  "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n\n# Test\narr = [3, 6, 8, 10, 1, 2, 1]\nprint(quicksort(arr))",
    "prose": "The transformer architecture has become the foundation of modern natural language processing. Its key innovation is the self-attention mechanism, which allows each token to attend to every other token in the sequence. This enables the model to capture long-range dependencies that were difficult for previous recurrent architectures.",
}

def load_static_profile():
    return json.load(open(PROFILE_FILE))

def get_experts(layer, hidden):
    """Get top-8 expert IDs for hidden state at this layer."""
    with torch.no_grad():
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            router_logits = layer.mlp.gate(hidden)  # [1, 1, 256]
        else:
            return None
        return router_logits[0, 0].topk(8).indices.tolist()

def run_decode(model, prompt, max_new=200):
    """Run decode, collect per-step hidden states and routing."""
    tokenizer = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True)
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_ids = tokens.clone()
    seq_len = input_ids.shape[1]
    print(f"  Prompt: {seq_len} tok, decode: {max_new} tok")

    # Per-step data
    per_step = []  # list of {step, hidden[index], experts_per_layer, token}

    with torch.no_grad():
        for step in range(max_new):
            if step % 20 == 0:
                print(f"    step {step}/{max_new}...", flush=True)

            # Forward: get hidden states for all layers at the last position
            outputs = model(input_ids, output_hidden_states=True, use_cache=False)
            hiddens = outputs.hidden_states  # [num_layers+1, B, S, D]

            # Collect routing for each layer at the last position
            experts = {}
            for L, layer in enumerate(model.model.layers):
                hs = hiddens[L + 1]  # output of layer L (post-attn+FFN)
                e = get_experts(layer, hs[:, -1:, :])  # [1, 1, D]
                if e is not None:
                    experts[L] = e

            # Greedy token
            logits = outputs.logits[0, -1]
            next_token = logits.argmax().item()

            per_step.append({
                "step": step,
                "token": next_token,
                "experts": experts,
            })

            # Append and continue
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

    return per_step

def compute_metrics(per_step, static_profile, max_layers=40):
    """Compute step-to-step routing recall vs static profile recall."""
    n = len(per_step)
    if n < 2:
        return {"error": "need at least 2 steps"}

    # Per-layer accumulators
    mtp_hits = {L: 0 for L in range(max_layers)}
    mtp_total = {L: 0 for L in range(max_layers)}
    static_hits = {L: 0 for L in range(max_layers)}
    static_total = {L: 0 for L in range(max_layers)}

    # Step-level union sizes
    step_union_sizes = []
    jaccard_scores = []
    mtp_recall_steps = []
    static_recall_steps = []

    for i in range(1, n):
        curr = per_step[i - 1]
        next_ = per_step[i]

        for L in range(max_layers):
            curr_experts = set(curr["experts"].get(L, []))
            next_experts = set(next_["experts"].get(L, []))

            if not next_experts:
                continue

            # MTP utility: current routing predicts next routing
            # (conservative: current step's routing, not actual MTP draft routing)
            mtp_total[L] += len(next_experts)
            if curr_experts:
                mtp_hits[L] += len(next_experts & curr_experts)

            # Static profile
            static_set = set(static_profile[L][:64])
            static_total[L] += len(next_experts)
            static_hits[L] += len(next_experts & static_set)

        # Step-level metrics
        curr_all = set()
        next_all = set()
        static_all = set()
        for L in range(max_layers):
            curr_all.update(per_step[i - 1]["experts"].get(L, []))
            next_all.update(per_step[i]["experts"].get(L, []))
            static_all.update(static_profile[L][:64])

        if curr_all and next_all:
            jaccard = len(curr_all & next_all) / len(curr_all | next_all)
            jaccard_scores.append(jaccard)
            mtp_recall = len(curr_all & next_all) / len(next_all) if next_all else 1.0
            mtp_recall_steps.append(mtp_recall)
            static_recall = len(next_all & static_all) / len(next_all) if next_all else 1.0
            static_recall_steps.append(static_recall)
            step_union_sizes.append(len(curr_all))

    # Per-layer recall
    mtp_recall_per_layer = {}
    static_recall_per_layer = {}
    for L in range(max_layers):
        mtp_recall_per_layer[L] = mtp_hits[L] / mtp_total[L] if mtp_total[L] > 0 else 0
        static_recall_per_layer[L] = static_hits[L] / static_total[L] if static_total[L] > 0 else 0

    result = {
        "mtp_recall_per_layer": mtp_recall_per_layer,
        "static_recall_per_layer": static_recall_per_layer,
        "mtp_avg_recall": sum(mtp_recall_per_layer.values()) / max_layers,
        "static_avg_recall": sum(static_recall_per_layer.values()) / max_layers,
        "mtp_step_avg_recall": sum(mtp_recall_steps) / len(mtp_recall_steps) if mtp_recall_steps else 0,
        "static_step_avg_recall": sum(static_recall_steps) / len(static_recall_steps) if static_recall_steps else 0,
        "avg_jaccard": sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0,
        "avg_union_size": sum(step_union_sizes) / len(step_union_sizes) if step_union_sizes else 0,
        "num_steps": n,
        "steps_analyzed": n - 1,
    }
    return result

if __name__ == "__main__":
    print("MoE-SpAc Verification (simplified)")
    print("=" * 60)

    print("\n[1/4] Loading static profile...")
    static_profile = load_static_profile()
    print(f"  40 layers × top-64 ✓")

    print("\n[2/4] Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        HF_DIR, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True, low_cpu_mem_usage=True)
    print(f"  {time.time()-t0:.1f}s, {model.config.num_hidden_layers} layers")

    print("\n[3/4] Running decode + routing traces...")
    all_results = {}
    for domain, prompt in PROMPTS.items():
        print(f"\n  --- {domain} ---")
        per_step = run_decode(model, prompt, max_new=200)
        metrics = compute_metrics(per_step, static_profile)
        all_results[domain] = metrics

        print(f"  {domain}:")
        print(f"    MTP (step-to-step) recall: {metrics['mtp_avg_recall']:.3f}")
        print(f"    Static top-64 recall:      {metrics['static_avg_recall']:.3f}")
        print(f"    Step Jaccard:              {metrics['avg_jaccard']:.3f}")
        print(f"    Avg union size:              {metrics['avg_union_size']:.0f}")
        print(f"    Steps:                       {metrics['num_steps']}")

    json.dump(all_results, open(OUTPUT, "w"), indent=1)
    print(f"\nResults saved to {OUTPUT}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    for domain, m in all_results.items():
        gain = m['mtp_avg_recall'] - m['static_avg_recall']
        print(f"  {domain}: MTP={m['mtp_avg_recall']:.1%} vs static={m['static_avg_recall']:.1%} (Δ={gain:+.1%})")
        print(f"          Jaccard={m['avg_jaccard']:.3f} union={m['avg_union_size']:.0f}")