import torch
from typing import Any, Dict, Optional
from pathlib import Path


def run_moe_entrypoint(
    *,
    device: torch.device,
    expert_dir: str,
    num_experts: int,
    expert_dim: int,
    intermediate_dim: int,
    batch_size: int = 2,
    seq_len: int = 8,
    top_k: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    from cgc_engine.flash_moe.client import FlashMoEClient
    from cgc_engine.omlx.client import OMLXClient

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    x = torch.randn(batch_size, seq_len, expert_dim, device=device, dtype=dtype)

    flash = FlashMoEClient(expert_dir=expert_dir, backend="auto")
    omlx = OMLXClient(
        model_dir=str(Path(expert_dir) / "omlx_model"),
        num_experts=int(num_experts),
        expert_dim=int(expert_dim),
        intermediate_dim=int(intermediate_dim),
        ssd_cache_dir=str(Path(expert_dir) / "omlx_ssd_cache"),
    )

    flash.expert_dim = int(expert_dim)
    flash.intermediate_dim = int(intermediate_dim)
    flash.num_experts = int(num_experts)

    pred = omlx.predict_experts(x.mean(dim=1), top_k=int(top_k))
    pred_ids = pred.detach().to("cpu").flatten().tolist()

    unique = []
    for i in pred_ids:
        i = int(i)
        if i not in unique:
            unique.append(i)
    expert_ids = unique[: int(top_k)]

    for eid in expert_ids:
        flash.load_expert(
            int(eid),
            expert_dim=int(expert_dim),
            intermediate_dim=int(intermediate_dim),
        )

    y = flash.moe_forward(x, expert_ids=expert_ids, top_k=int(top_k))

    return {
        "device": str(device),
        "dtype": str(dtype),
        "input_shape": tuple(x.shape),
        "output_shape": tuple(y.shape),
        "expert_ids": expert_ids,
        "output": y,
    }
