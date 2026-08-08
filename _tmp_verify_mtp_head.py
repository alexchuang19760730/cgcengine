#!/usr/bin/env python3
"""End-to-end MTP head verification: manual forward vs training code path."""
import torch
import os, sys

os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

# Load target model (Qwen3-VL is multimodal)
print("Loading target model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/data2/models/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).cuda()
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

tokenizer = AutoTokenizer.from_pretrained(
    "/data2/models/Qwen3-VL-2B-Instruct", trust_remote_code=True
)

# Test prompt
prompt = (
    'def fibonacci(n: int) -> int:\n'
    '    """Compute the nth Fibonacci number recursively."""\n'
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"Prompt: {prompt[:60]!r}...")
print(f"Input ids: {inputs.input_ids.shape}")

# Forward target model
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden]
logits = outputs.logits[:, -1, :]
target_next_id = int(logits.argmax(dim=-1).item())
target_next = tokenizer.decode([target_next_id])
print(f"Target next token: {target_next_id} ({target_next!r})")
print(f"Target hidden: mean={hidden.float().mean():.6f}, std={hidden.float().std():.6f}")

# Load MTP head checkpoint
print("\nLoading MTP head...")
ckpt = torch.load(
    "/data/mtp_output/qwen3vl/mtp_head_qwen3vl_decode.pt",
    map_location="cuda", weights_only=False,
)
mtp_sd = ckpt.get("model_state_dict", ckpt)

# Build exact MTP head (same as training architecture)
hidden_size = 2048
intermediate = 5632


class MTPHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm1 = torch.nn.RMSNorm(hidden_size, eps=1e-6)
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm2 = torch.nn.RMSNorm(hidden_size, eps=1e-6)
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden_size, bias=False)
        self.norm_out = torch.nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(self, target_hidden, token_embed):
        # EXACTLY matches training: concat [target_hidden, token_embed]
        x = self.proj(torch.cat([target_hidden, token_embed], dim=-1))
        # Manual attention on seq_len=1: o_proj(v_proj(norm1(x)))
        normed = self.norm1(x)
        v = self.v_proj(normed)
        attn_out = self.o_proj(v)
        x = x + attn_out
        # MLP: gate(up * silu(gate))
        normed2 = self.norm2(x)
        gate = torch.nn.functional.silu(self.gate_proj(normed2))
        up = self.up_proj(normed2)
        x = x + self.down_proj(gate * up)
        # norm_out
        return self.norm_out(x)


mtp = MTPHead().cuda().to(torch.bfloat16)

weight_map = {
    "proj.weight":          "proj.weight",
    "norm1.weight":         "norm1.weight",
    "attn.q_proj.weight":   "q_proj.weight",
    "attn.k_proj.weight":   "k_proj.weight",
    "attn.v_proj.weight":   "v_proj.weight",
    "attn.o_proj.weight":   "o_proj.weight",
    "norm2.weight":         "norm2.weight",
    "mlp.gate_proj.weight": "gate_proj.weight",
    "mlp.up_proj.weight":   "up_proj.weight",
    "mlp.down_proj.weight": "down_proj.weight",
    "norm_out.weight":      "norm_out.weight",
}

for pt_name, mtp_name in weight_map.items():
    if pt_name in mtp_sd:
        w = mtp_sd[pt_name].to(torch.bfloat16)
        dict(mtp.named_parameters())[mtp_name].data.copy_(w)
    else:
        print(f"WARNING: {pt_name} not found in checkpoint")

print(f"MTP head params: {sum(p.numel() for p in mtp.parameters())/1e6:.1f}M")

# Embeddings and lm_head access
# Qwen3VLForConditionalGeneration: .model (vision+language), .model.language_model, .lm_head
lang_model = model.model.language_model  # Qwen3LLMModel
embed_weight = lang_model.embed_tokens.weight
lm_head = model.lm_head.weight
with torch.no_grad():
    last_token_id = int(inputs.input_ids[0, -1].item())
    last_token_embed = embed_weight[last_token_id].unsqueeze(0)
    print(f"Last input token: {last_token_id} ({tokenizer.decode([last_token_id])!r})")

    mtp_hidden = mtp(target_hidden=hidden, token_embed=last_token_embed)

    mtp_logits = torch.nn.functional.linear(mtp_hidden, lm_head)
    target_logits = torch.nn.functional.linear(hidden, lm_head)

    mtp_next_id = int(mtp_logits.argmax(dim=-1).item())
    target_next_id2 = int(target_logits.argmax(dim=-1).item())

    print(f"\n=== Single-step Results ===")
    print(f"Target next: {target_next_id2} ({tokenizer.decode([target_next_id2])!r})")
    print(f"MTP predicts: {mtp_next_id} ({tokenizer.decode([mtp_next_id])!r})")
    print(f"Match: {mtp_next_id == target_next_id2}")

    # Also test: multi-step chain
    print(f"\n=== Multi-step chain test (5 steps) ===")
    cur_token_id = last_token_id
    cur_hidden = hidden.clone()
    chain_matches = 0
    for step in range(5):
        # Get next real token
        with torch.no_grad():
            next_out = model(
                torch.tensor([[cur_token_id]], device="cuda"),
                output_hidden_states=True,
            )
        real_next_id = int(next_out.logits[0, -1].argmax().item())
        real_next_hidden = next_out.hidden_states[-1][0, 0:1]

        # MTP prediction
        cur_embed = embed_weight[cur_token_id].unsqueeze(0)
        mtp_h = mtp(target_hidden=cur_hidden, token_embed=cur_embed)
        mtp_pred_id = int(torch.nn.functional.linear(mtp_h, lm_head).argmax().item())

        match = mtp_pred_id == real_next_id
        chain_matches += int(match)
        print(
            f"  Step {step}: target={tokenizer.decode([real_next_id])!r:15s} "
            f"MTP={tokenizer.decode([mtp_pred_id])!r:15s} {'MATCH' if match else 'MISMATCH'}"
        )

        # Advance
        cur_token_id = real_next_id
        cur_hidden = real_next_hidden

    print(f"Chain match: {chain_matches}/5 ({chain_matches/5*100:.0f}%)")
