"""Capture sglang target model hidden states and run MTP head on them."""
import torch, os, sys, json
sys.path.insert(0, "/data2/venv_gemma4/lib/python3.12/site-packages")
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, CaptureHiddenMode

# Config
model_path = "/data2/models/Qwen3-VL-2B-Instruct"
draft_path = "/data/eagle_drafts/qwen3vl"

# Server args
server_args = ServerArgs(
    model_path=model_path,
    tp_size=1,
    mem_fraction_static=0.80,
    trust_remote_code=True,
    attention_backend="triton",
    sampling_backend="pytorch",
    disable_cuda_graph=True,
    skip_server_warmup=True,
)

# Init model runner
print("Initializing target model runner...")
runner = ModelRunner(
    tp_rank=0,
    tp_size=1,
    server_args=server_args,
)
runner.init_tp_groups()
runner.load_model()
print("Target model loaded.")

# Tokenize a prompt
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = "def fibonacci(n: int) -> int:\n"
input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
print("Input: %r -> %d tokens" % (prompt, len(input_ids)))

# Create a request
sampling_params = SamplingParams(max_new_tokens=1, temperature=0)
req = Req(
    rid="test1",
    origin_input_text=prompt,
    origin_input_ids=tuple(input_ids),
    sampling_params=sampling_params,
)
req.fill_ids = input_ids
req.sampling_params.max_new_tokens = 1

# Add req to batch
batch = ScheduleBatch(reqs=[req], max_prefill_tokens=8192, current_forward_mode=None)
batch.prepare_for_extend()

# Run forward with hidden states capture
with torch.no_grad():
    forward_batch = ForwardBatch.init_new(
        batch, runner, capture_hidden_mode=CaptureHiddenMode.FULL
    )
    output = runner.forward(forward_batch)

# Extract hidden states
hidden = output.logits_output.hidden_states
print("Hidden states shape: %s" % str(hidden.shape))

# Get last token hidden state
last_hidden = hidden[-1:].clone()
print("Last hidden: mean=%.6f, std=%.6f" % (
    last_hidden.float().mean().item(), last_hidden.float().std().item()
))

# Get token embedding and lm_head
embed_weight = runner.model.get_input_embeddings().weight
lm_head_weight = runner.model.get_output_embeddings().weight

# Load MTP head
ckpt = torch.load(
    "/data/mtp_output/qwen3vl/mtp_head_qwen3vl_decode.pt",
    map_location="cuda", weights_only=False
)
mtp_sd = ckpt.get("model_state_dict", ckpt)


class MTPHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4096, 2048, bias=False)
        self.norm1 = torch.nn.RMSNorm(2048, eps=1e-6)
        self.q_proj = torch.nn.Linear(2048, 2048, bias=False)
        self.k_proj = torch.nn.Linear(2048, 2048, bias=False)
        self.v_proj = torch.nn.Linear(2048, 2048, bias=False)
        self.o_proj = torch.nn.Linear(2048, 2048, bias=False)
        self.norm2 = torch.nn.RMSNorm(2048, eps=1e-6)
        self.gate_proj = torch.nn.Linear(2048, 5632, bias=False)
        self.up_proj = torch.nn.Linear(2048, 5632, bias=False)
        self.down_proj = torch.nn.Linear(5632, 2048, bias=False)
        self.norm_out = torch.nn.RMSNorm(2048, eps=1e-6)

    def forward(self, target_hidden, token_embed):
        x = self.proj(torch.cat([target_hidden, token_embed], dim=-1))
        normed = self.norm1(x)
        v = self.v_proj(normed)
        attn_out = self.o_proj(v)
        x = x + attn_out
        normed2 = self.norm2(x)
        gate = torch.nn.functional.silu(self.gate_proj(normed2))
        up = self.up_proj(normed2)
        x = x + self.down_proj(gate * up)
        return self.norm_out(x)


mtp = MTPHead().cuda().to(torch.bfloat16)

weight_map = {
    "proj.weight": "proj.weight",
    "norm1.weight": "norm1.weight",
    "attn.q_proj.weight": "q_proj.weight",
    "attn.k_proj.weight": "k_proj.weight",
    "attn.v_proj.weight": "v_proj.weight",
    "attn.o_proj.weight": "o_proj.weight",
    "norm2.weight": "norm2.weight",
    "mlp.gate_proj.weight": "gate_proj.weight",
    "mlp.up_proj.weight": "up_proj.weight",
    "mlp.down_proj.weight": "down_proj.weight",
    "norm_out.weight": "norm_out.weight",
}
for pt_name, mtp_name in weight_map.items():
    if pt_name in mtp_sd:
        dict(mtp.named_parameters())[mtp_name].data.copy_(
            mtp_sd[pt_name].to(torch.bfloat16)
        )

# Run MTP
last_token_id = input_ids[-1]
token_embed = embed_weight[last_token_id].unsqueeze(0)
print("Last input token: %d (%r)" % (last_token_id, tokenizer.decode([last_token_id])))

mtp_hidden = mtp(target_hidden=last_hidden, token_embed=token_embed)
mtp_logits = torch.nn.functional.linear(mtp_hidden, lm_head_weight)
target_logits = torch.nn.functional.linear(last_hidden, lm_head_weight)

mtp_pred = int(mtp_logits.argmax().item())
target_pred = int(target_logits.argmax().item())

print("\n=== SGLLang hidden state test ===")
print("Target pred: %d (%r)" % (target_pred, tokenizer.decode([target_pred])))
print("MTP pred:   %d (%r)" % (mtp_pred, tokenizer.decode([mtp_pred])))
print("Match: %s" % (mtp_pred == target_pred))
cos_sim = torch.nn.functional.cosine_similarity(
    mtp_logits.float(), target_logits.float(), dim=-1
)
print("Logits cosine similarity: %.4f" % cos_sim.item())

# Also test with embeds from sglang's internal embedding
print("\n=== Testing with sglang embed_tokens ===")
sglang_embed = runner.model.get_input_embeddings()
sglang_token_embed = sglang_embed(
    torch.tensor([last_token_id], device="cuda")
)
print("sglang embed shape: %s" % str(sglang_token_embed.shape))
mtp_hidden2 = mtp(target_hidden=last_hidden, token_embed=sglang_token_embed)
mtp_logits2 = torch.nn.functional.linear(mtp_hidden2, lm_head_weight)
mtp_pred2 = int(mtp_logits2.argmax().item())
print("MTP pred (sglang embed): %d (%r)" % (mtp_pred2, tokenizer.decode([mtp_pred2])))
print("Match (sglang embed): %s" % (mtp_pred2 == target_pred))
