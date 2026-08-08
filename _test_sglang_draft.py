#!/usr/bin/env python3
"""Test sglang CgcMtpModel forward with EAGLE-style inputs on Host2."""

import torch
import torch.nn.functional as F
import sys, os

# Add sglang to path
sys.path.insert(0, "/data2/venv_gemma4/lib/python3.12/site-packages")

TARGET_PATH = "/data2/models/Qwen3-VL-2B-Instruct"
DRAFT_PATH = "/data/eagle_drafts/qwen3vl"

def main():
    device = torch.device("cuda:0")
    
    # 1. Load target model
    print("Loading target model...")
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
    target = Qwen3VLForConditionalGeneration.from_pretrained(
        TARGET_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True)
    
    # 2. Load sglang draft model
    print("Loading sglang draft model...")
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.model_loader import get_model
    from sglang.srt.models.cgc_mtp_eagle import CgcMtpForCausalLMEagle
    
    # Create model config
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(DRAFT_PATH, trust_remote_code=True)
    
    draft = CgcMtpForCausalLMEagle(
        config=hf_config,
    )
    
    # Load weights from safetensors
    from safetensors.torch import load_file
    draft_state = load_file("/data/eagle_drafts/qwen3vl/model.safetensors")
    
    # Load with prefix handling
    draft_sd = draft.state_dict()
    loaded = {}
    for k, v in draft_state.items():
        if k in draft_sd:
            loaded[k] = v
        else:
            print(f"  Skip {k} (not in model)")
    
    missing = set(draft_sd.keys()) - set(loaded.keys())
    if missing:
        print(f"  Missing keys: {list(missing)[:10]}")
    
    draft.load_state_dict(loaded, strict=False)
    
    # Share embed and lm_head from target model
    print("Sharing embed and lm_head from target...")
    target_embed = target.model.language_model.embed_tokens.weight.data
    target_lm_head = target.lm_head.weight.data
    
    draft.model.embed_tokens.weight.data.copy_(target_embed)
    draft.lm_head.weight.data.copy_(target_lm_head)
    
    draft = draft.to(device=device, dtype=torch.bfloat16).eval()
    
    # 3. Test with real prompt
    print("\n=== Test with code prompt ===")
    prompt = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x =="
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    L = input_ids.shape[1]
    
    # Get target hidden states
    with torch.no_grad():
        lang_out = target.model.language_model(input_ids=input_ids, output_hidden_states=True)
        all_hidden = lang_out.last_hidden_state  # [1, L, 2048]
    
    # Create a fake ForwardBatch with spec_info
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    
    print("\nPer-position test (using sglang draft model forward):")
    print(f"{'Pos':>4} {'Input':>25} {'Target':>25} {'Draft':>25} {'Match':>6}")
    print("-" * 90)
    
    correct = 0
    for i in range(L - 1):
        hidden_i = all_hidden[:, i:i+1, :]  # [1, 1, hidden]
        token_i = input_ids[:, i:i+1]       # [1, 1]
        
        # Create fake batch
        class FakeSpecInfo:
            def __init__(self, hidden):
                self.hidden_states = hidden
                self.input_ids = None
                self.topk_probs = None
                self.topk_indices = None
        
        class FakeBatch:
            def __init__(self, ids, hidden):
                self.spec_info = FakeSpecInfo(hidden)
                self.input_ids = ids
        
        fake_batch = FakeBatch(token_i, hidden_i)
        positions = torch.zeros_like(token_i, device=device)
        
        with torch.no_grad():
            draft_hidden = draft.model(
                input_ids=token_i,
                positions=positions,
                forward_batch=fake_batch,
            )
            # draft.logits_processor is for final logits; let's use lm_head directly
            draft_logits = draft.lm_head(draft_hidden)
        
        draft_pred = draft_logits.argmax().item()
        
        target_logits_i = target.lm_head(all_hidden[:, i:i+1, :])
        target_pred = target_logits_i.argmax().item()
        
        actual_next = input_ids[0, i+1].item()
        
        match = draft_pred == target_pred
        if match:
            correct += 1
        
        input_text = tokenizer.decode([input_ids[0, i].item()])
        target_text = tokenizer.decode([target_pred])
        draft_text = tokenizer.decode([draft_pred])
        marker = "✓" if match else "✗"
        
        print(f"{i:>4} {input_text[:25]:>25} {target_text[:25]:>25} "
              f"{draft_text[:25]:>25} {marker:>6}")
    
    print("-" * 90)
    print(f"\nDraft matches target: {correct}/{L-1} = {correct/(L-1)*100:.0f}%")
    
    # Also compare with training-style MTPHead for sanity
    print("\n=== Compare with training MTPHead ===")
    sys.path.insert(0, "/root/flashkv0516/CGC_Phase2/mtp_head")
    from model import create_mtp_head_for_qwen3vl_2b
    
    ckpt = torch.load("/data/mtp_output/qwen3vl/mtp_head_qwen3vl_decode.pt", 
                      map_location="cpu", weights_only=False)
    mtp_train = create_mtp_head_for_qwen3vl_2b()
    mtp_train.load_state_dict(ckpt["model_state_dict"], strict=True)
    mtp_train.set_shared_lm_head(target_lm_head)
    mtp_train = mtp_train.to(device=device, dtype=torch.bfloat16).eval()
    
    train_correct = 0
    for i in range(L - 1):
        hidden_i = all_hidden[:, i:i+1, :]
        token_i = input_ids[:, i:i+1]
        token_embed = target.model.language_model.embed_tokens(token_i)
        
        with torch.no_grad():
            train_logits = mtp_train(hidden_i, token_embed)
        train_pred = train_logits.argmax().item()
        
        target_pred = target.lm_head(all_hidden[:, i:i+1, :]).argmax().item()
        if train_pred == target_pred:
            train_correct += 1
    
    print(f"Training MTPHead matches target: {train_correct}/{L-1} = {train_correct/(L-1)*100:.0f}%")
    
    # If sglang draft matches training MTPHead, the forward is correct
    if correct == train_correct:
        print("✓ Sglang draft model output matches training MTPHead output")
    else:
        print(f"✗ MISMATCH: sglang({correct}) vs training({train_correct})")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
