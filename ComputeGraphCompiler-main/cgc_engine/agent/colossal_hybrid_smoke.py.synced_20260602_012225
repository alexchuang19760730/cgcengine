import os
import torch
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_int(name: str, default: int) -> int:
    v = os.environ.get(name, str(default)).strip()
    try:
        return int(v)
    except Exception:
        return default


def main() -> None:
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model  # type: ignore

        if not hasattr(GPT2Model, "get_head_mask"):
            def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):  # type: ignore
                if head_mask is None:
                    return [None] * int(num_hidden_layers)
                return head_mask

            GPT2Model.get_head_mask = _get_head_mask  # type: ignore[attr-defined]
    except Exception:
        pass

    model_id = os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2").strip()
    microbatch_size = _get_int("MICROBATCH_SIZE", 1)
    seq_len = _get_int("SEQ_LEN", 256)
    tp_size = _get_int("TP_SIZE", 1)
    pp_size = _get_int("PP_SIZE", 2)
    zero_stage = _get_int("ZERO_STAGE", 0)
    steps = _get_int("STEPS", 2)
    precision = os.environ.get("PRECISION", "bf16").strip()

    colossalai.launch_from_torch(seed=42)
    coordinator = DistCoordinator()

    plugin = HybridParallelPlugin(
        tp_size=tp_size,
        pp_size=pp_size,
        microbatch_size=microbatch_size,
        zero_stage=zero_stage,
        precision=precision,
    )
    booster = Booster(plugin=plugin)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if precision.lower() == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def criterion(outputs, inputs):
        return outputs.loss

    class Dummy(torch.utils.data.Dataset):
        def __len__(self):
            return 32

        def __getitem__(self, idx):
            t = tokenizer(
                "hello %d" % idx,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            )
            batch = {k: v.squeeze(0) for k, v in t.items()}
            batch["labels"] = batch["input_ids"].clone()
            return batch

    effective_batch_size = microbatch_size * pp_size if pp_size > 1 else microbatch_size
    dataloader = plugin.prepare_dataloader(
        Dummy(),
        batch_size=effective_batch_size,
        shuffle=False,
        drop_last=True,
    )

    model, optimizer, criterion, dataloader, _lr_scheduler = booster.boost(model, optimizer, criterion, dataloader)

    if coordinator.is_master():
        world = coordinator.world_size
        dp = world // max(1, (tp_size * pp_size))
        print(
            "SMOKE_OK world=%d dp~=%d pp=%d tp=%d zero=%d precision=%s cuda=%s ngpu=%d model=%s"
            % (
                world,
                dp,
                pp_size,
                tp_size,
                zero_stage,
                precision,
                str(torch.cuda.is_available()),
                int(torch.cuda.device_count()),
                model_id,
            )
        )

    it = iter(dataloader)
    for step in range(steps):
        batch = next(it)
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        if pp_size > 1:
            result = booster.execute_pipeline(
                data_iter=iter([batch]),
                model=model,
                criterion=lambda outputs, inputs: criterion(outputs, inputs),
                optimizer=optimizer,
                return_loss=True,
                return_outputs=False,
            )
            loss = result.get("loss")
        else:
            outputs = model(**batch)
            loss = criterion(outputs, batch)
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if coordinator.is_master():
            if loss is None:
                print("step=%d loss=None" % step)
            else:
                loss_val = float(loss.detach().cpu())
                print("step=%d loss=%f" % (step, loss_val))


if __name__ == "__main__":
    main()
