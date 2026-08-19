#!/usr/bin/env python3
"""Export golden test vectors for the Metal Qwen36MTPHeadRunner.

For 3 deterministic (hidden, embedding) pairs, run the framework head and
save:
  - mtp_head_in_hidden_N.bin  (fp16 [2048])
  - mtp_head_in_embed_N.bin   (fp16 [2048])
  - mtp_head_ref_top1_N.json  (argmax token id)
The Metal test feeds the same vectors through Qwen36MTPHeadRunner.forwardToken
and asserts the argmax matches.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "/Users/alexchuang/Documents/flashkv0516"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/CGC_Phase2/mtp_head")
from model import create_mtp_head_for_qwen36
from safetensors import safe_open

CKPT = f"{REPO}/temp/qwen36_mtp_data/mtp_head_qwen36_official.pt"
OUT = Path(f"{REPO}/prime-agent-worktrees/turbo-fieldfare/Tests/TurboFieldfare/Core/Kernels/metal_out")
OUT.mkdir(parents=True, exist_ok=True)
HIDDEN = 2048

mtp = create_mtp_head_for_qwen36()
sd = torch.load(CKPT, map_location="cpu", weights_only=True)["model_state_dict"]
mtp.load_state_dict(sd, strict=True)
sf = safe_open("/Volumes/AlexZhuang/qwen36-hf/model-00026-of-00026.safetensors", framework="pt")
mtp.set_shared_lm_head(sf.get_tensor("lm_head.weight"), trainable=False)
mtp.eval()

torch.manual_seed(11)
for i in range(3):
    hidden = torch.randn(1, HIDDEN) * 0.02
    embed = torch.randn(1, HIDDEN) * 0.02
    with torch.no_grad():
        logits = mtp(hidden.unsqueeze(0), embed.unsqueeze(0))  # [B=1,S=1,H]
    top1 = int(logits.argmax(-1))
    hidden.numpy().astype(np.float16).tofile(OUT / f"mtp_head_in_hidden_{i}.bin")
    embed.numpy().astype(np.float16).tofile(OUT / f"mtp_head_in_embed_{i}.bin")
    (OUT / f"mtp_head_ref_top1_{i}.json").write_text(json.dumps({"top1": top1}))
    print(f"case {i}: top1={top1} (hidden/embed written)")
print("DONE")
