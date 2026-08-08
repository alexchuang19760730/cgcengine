from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.jit_kernel.dsv4 import (
    fused_norm_rope_inplace,
    fused_q_norm_rope,
    fused_rope_inplace,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.communicator_dsa_cp import (
    dsa_cp_gather_hidden_states,
    dsa_cp_reduce_scatter_hidden_states,
)
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    attn_tp_all_gather,
    dp_gather_partial,
    dp_scatter,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.mhc import mhc_fused_post_pre
from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_round_robin_input_ids,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_runner import (
    compile_in_capture_mode,
    get_is_capture_mode,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_common.utils import awq_dequantize_func
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
    try_fused_hc_post_pre,
)
from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu

if not _is_hip:
    from sglang.srt.layers.utils.cp_utils import (
        prepare_context_parallel_metadata,
    )

from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    get_bool_env_var,
    is_gfx95_supported,
    log_info_on_rank0,
    make_layers,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)


def _parse_debug_layer_id_set(env_name: str) -> Set[int]:
    raw = str(os.environ.get(env_name, "") or "").strip()
    if not raw:
        return set()
    values: Set[int] = set()
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            values.add(int(text))
        except ValueError:
            logger.warning(
                "[DeepseekV4LayerTrace] event=%r env_name=%r invalid_value=%r",
                "debug_layer_set_parse_failed",
                env_name,
                text,
            )
    return values


_DEBUG_DISABLE_INDEXER_LAYERS = _parse_debug_layer_id_set(
    "CGC_DEBUG_DISABLE_INDEXER_LAYERS"
)
_DEBUG_DISABLE_COMPRESSOR_LAYERS = _parse_debug_layer_id_set(
    "CGC_DEBUG_DISABLE_COMPRESSOR_LAYERS"
)


def _log_deepseek_v4_layer_trace(event: str, **fields: object) -> None:
    parts = [f"{key}={value!r}" for key, value in fields.items()]
    suffix = f" {' '.join(parts)}" if parts else ""
    logger.info("[DeepseekV4LayerTrace] event=%r%s", event, suffix)

_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()
_MHC_POST_MULT_VALUE = 2.0


def _format_unresolved_wqkv_a_contract_error(
    cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]],
    weight_names: list[str],
) -> str:
    unresolved = ", ".join(
        f"{param_name}[{','.join(sorted(bucket.keys()))}]"
        for param_name, bucket in sorted(cache_wqkv_a_weight.items())
    )
    has_legacy_kv = any(".self_attn.legacy_kv_" in name for name in weight_names)
    has_legacy_o_proj = any(".self_attn.legacy_o_proj." in name for name in weight_names)
    has_wkv = any(".self_attn.wkv." in name for name in weight_names)
    return (
        "DeepSeek-V4 weight mapping contract mismatch: unresolved fused wqkv_a "
        f"shards={unresolved}. checkpoint_has_legacy_kv={has_legacy_kv} "
        f"checkpoint_has_legacy_o_proj={has_legacy_o_proj} checkpoint_has_wkv={has_wkv}. "
        "This checkpoint family remaps attention weights into "
        "`wq_a + legacy_kv + legacy_o_proj`, but the current runtime only "
        "consumes `legacy_o_proj` and still expects `wq_a + wkv -> wqkv_a` "
        "for the KV/cache path. A legacy_kv bridge into the DSV4 cache path is "
        "required before these weights can be materialized safely."
    )


def _build_q_only_fused_wqkv_tensor(
    *,
    param: torch.Tensor,
    q_tensor: torch.Tensor,
    param_name: str,
) -> torch.Tensor:
    """Fill the missing fused KV shard when a checkpoint only provides q_a.

    DeepSeek-V4-Flash checkpoints in the legacy_kv family expose `q_a` plus
    `legacy_kv_*`, not `wkv`. The fused runtime still instantiates `wqkv_a`, so
    we materialize a q-only fused tensor here and let the request-time path make
    the legacy_kv decision explicitly.
    """
    target_shape = tuple(param.shape)
    q_shape = tuple(q_tensor.shape)
    if len(target_shape) != len(q_shape) or target_shape[1:] != q_shape[1:]:
        raise RuntimeError(
            "Cannot synthesize q-only fused wqkv_a shard because the q shard "
            f"shape does not match the fused parameter suffix: param={param_name} "
            f"target_shape={target_shape} q_shape={q_shape}"
        )
    missing_rows = int(target_shape[0]) - int(q_shape[0])
    if missing_rows < 0:
        raise RuntimeError(
            "Cannot synthesize q-only fused wqkv_a shard because the q shard is "
            f"larger than the fused parameter: param={param_name} "
            f"target_shape={target_shape} q_shape={q_shape}"
        )
    if missing_rows == 0:
        return q_tensor
    filler_factory = q_tensor.new_ones if param_name.endswith(".weight_scale_inv") else q_tensor.new_zeros
    filler = filler_factory((missing_rows, *q_shape[1:]))
    return torch.cat([q_tensor, filler], dim=0)


# === DUMP PATCH (CGC diagnostic): capture per-layer post-block residual ===
import os as _os

_FORK_DUMP_HS = _os.environ.get("SGLANG_DSV4_DUMP_HS", "") == "1"
_FORK_DUMP_IDS_S = _os.environ.get("SGLANG_DSV4_DUMP_HS_IDS", "")
_FORK_DUMP_IDS = None
if _FORK_DUMP_IDS_S:
    try:
        _FORK_DUMP_IDS = tuple(int(_x) for _x in _FORK_DUMP_IDS_S.split(",") if _x != "")
    except Exception:
        _FORK_DUMP_IDS = None
_FORK_HS_DICT = {}
_FORK_HS_SAVED = [False]
# === END DUMP PATCH ===


def _is_fused_mhc_post_pre_enabled() -> bool:
    # The fused path directly reuses TileLang mhc_post/mhc_pre kernels and their
    # tensor layout assumptions, so keep it disabled when either dependency is off.
    return (
        envs.SGLANG_OPT_FUSE_MHC_POST_PRE.get()
        and envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()
        and envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
    )


_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_gfx95_supported = is_gfx95_supported()

if _use_aiter:
    if _is_gfx95_supported:
        from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant


def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps):
    x_quant, x_bf16, _, _ = fused_rms_fp8_group_quant(
        hidden_states,
        weight,
        eps,
        inp2=None,
        inp2_weight=None,
        inp2_epsilon=None,
        group_size=128,
        dtype_quant=torch.float8_e4m3fn,
        res1=None,
        output_unquantized_inp1=True,
    )
    return x_quant, x_bf16


_FREQS_CIS_TO_COS_SIN: dict[
    Tuple[int, torch.dtype, torch.device], Tuple[torch.Tensor, torch.Tensor]
] = {}


def _freqs_cis_to_cos_sin(
    freqs_cis: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive (cos, sin) bf16 contiguous tables from a complex64 `freqs_cis`,
    cached by `(id(freqs_cis), dtype, device)` so that all layers sharing the
    same `freqs_cis` (via `precompute_freqs_cis`'s lru_cache) reuse one pair."""
    key = (id(freqs_cis), dtype, device)
    cached = _FREQS_CIS_TO_COS_SIN.get(key)
    if cached is not None:
        return cached
    fr = torch.view_as_real(freqs_cis)
    cos = fr[..., 0].to(device=device, dtype=dtype).contiguous()
    sin = fr[..., 1].to(device=device, dtype=dtype).contiguous()
    _FREQS_CIS_TO_COS_SIN[key] = (cos, sin)
    return cos, sin


if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
    )
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@triton.jit
def _rms_normalize_kernel(
    x_ptr,
    weight_ptr,
    eps,
    stride_row,
    dim,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    base = pid * stride_row
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / dim
    rms_inv = tl.rsqrt(mean_sq + eps)
    out = x * rms_inv

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = out * weight

    tl.store(x_ptr + base + offs, out, mask=mask)


def rms_normalize_triton(
    x: torch.Tensor, eps: float, weight: torch.Tensor = None
) -> torch.Tensor:
    dim = x.shape[-1]
    x_flat = x.view(-1, dim)
    num_rows = x_flat.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(dim)
    grid = (num_rows,)

    _rms_normalize_kernel[grid](
        x_flat,
        weight,
        eps,
        x_flat.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
    )
    return x


class MQALayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.tp_rank = attn_tp_rank = get_attention_tp_rank()
        self.tp_size = attn_tp_size = get_attention_tp_size()
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()
            self.tp_rank = attn_tp_rank = 0
            self.tp_size = attn_tp_size = 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // attn_tp_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        compress_ratio = (
            compress_ratio_override
            if compress_ratio_override is not None
            else config.compress_ratios[layer_id]
        )
        assert compress_ratio in [0, 4, 128]
        self.compress_ratio: Literal[0, 4, 128] = compress_ratio

        assert self.head_dim == config.head_dim
        assert config.num_key_value_heads == 1

        rope_theta, rope_scaling = get_rope_config(config)
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        rope_base = config.compress_rope_theta if self.compress_ratio else rope_theta

        self.rotary_emb = get_rope_wrapper(
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_base,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=get_global_server_args().device,
        )

        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        assert self.compress_ratio in {0, 4, 128}
        if self.compress_ratio:
            original_seq_len = rope_scaling["original_max_position_embeddings"]
        else:
            original_seq_len = 0

        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=rope_base,
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        if _is_hip:
            cos_cache = freqs_cis.real.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            sin_cache = freqs_cis.imag.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            self.register_buffer("cos_cache", cos_cache, persistent=False)
            self.register_buffer("sin_cache", sin_cache, persistent=False)

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]
            self.alt_streams_indexer = alt_streams[-2:]
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                freqs_cis=freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
                rotary_emb=getattr(self, "rotary_emb", None),
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    freqs_cis=freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                    rotary_emb=getattr(self, "rotary_emb", None),
                )

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        if self.fuse_wqa_wkv:
            self.wqkv_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wqkv_a", prefix),
            )
        else:
            self.wq_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wq_a", prefix),
            )
            self.wkv = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wkv", prefix),
            )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # Stage-1 legacy MLA compat: add loadable targets for DeepSeek-V2-style
        # KV tensors without wiring them into the V4 runtime path yet.
        self.legacy_kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("legacy_kv_a_proj_with_mqa", prefix),
        )
        self.legacy_kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=self.eps)
        self.legacy_kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("legacy_kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # Stage-1 legacy output compat: add a real load target for V2-style
        # single-stage o_proj weights before introducing a forward branch.
        self.legacy_o_proj = RowParallelLinear(
            self.n_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size > 1,
            prefix=add_prefix("legacy_o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        self.legacy_attn_mqa = RadixAttention(
            self.n_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("legacy_attn_mqa", prefix),
        )
        self.register_buffer("legacy_w_kc", None, persistent=False)
        self.register_buffer("legacy_w_vc", None, persistent=False)
        self.use_legacy_kv = False
        self.use_legacy_o_proj = False
        self._logged_legacy_kv_contract = False
        self._logged_legacy_o_proj_contract = False
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config if _FP8_WO_A_GEMM else None,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),
        )
        if _FP8_WO_A_GEMM:
            assert hasattr(
                self.wo_a, "weight_scale_inv"
            ), "FP8 quant_config must create weight_scale_inv"
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size > 1,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.use_fused_qk_norm_rope = (
            _is_hip and envs.SGLANG_OPT_USE_FUSED_QK_NORM_ROPE.get()
        )

        # KV cache write is always fused into the K kernel
        # (`_compute_kv_to_cache`), so the legacy "overlap store cache" flag
        # has no effect here -- the fused path is on by default.

    def _compute_q_a(
        self,
        x: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            q = qkv_a[..., : self.q_lora_rank]
        else:
            q, _ = self.wq_a(x)
        return self.q_norm(q)

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if q_out is None:
            q_out = torch.empty_like(q)
        # Fused warp-per-(token, head) rmsnorm-self + RoPE + write to q_out.
        fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
        return q_out

    def _materialize_legacy_kv_absorb_weights(self) -> None:
        """Split legacy_kv_b_proj into absorbed-MLA K/V weights.

        The legacy_kv checkpoint family follows the DeepSeek-V2 MLA contract:
        cache the normalized latent KV plus RoPE part, then apply the K/V slices
        of kv_b_proj as separate per-head BMMs at request time.
        """
        if self.legacy_w_kc is not None and self.legacy_w_vc is not None:
            return

        if hasattr(self.legacy_kv_b_proj, "qweight"):
            awq_dequantize_f = awq_dequantize_func()
            if awq_dequantize_f is None:
                raise RuntimeError(
                    "legacy_kv runtime branch requires AWQ dequantization support "
                    "for legacy_kv_b_proj, but no device-specific AWQ dequant "
                    "function is available."
                )
            w = awq_dequantize_f(
                self.legacy_kv_b_proj.qweight,
                self.legacy_kv_b_proj.scales,
                self.legacy_kv_b_proj.qzeros,
            ).T
        else:
            w = self.legacy_kv_b_proj.weight

        if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            selected_quant_config = getattr(self.quant_config, "linear_fp8_config", None)
            if selected_quant_config is None:
                selected_quant_config = self.quant_config
            weight_block_size = getattr(selected_quant_config, "weight_block_size", None)
            weight_scale = (
                self.legacy_kv_b_proj.weight_scale
                if hasattr(self.legacy_kv_b_proj, "weight_scale")
                else getattr(self.legacy_kv_b_proj, "weight_scale_inv", None)
            )
            if weight_scale is None:
                raise RuntimeError(
                    "legacy_kv runtime branch expected legacy_kv_b_proj to expose "
                    "weight_scale/weight_scale_inv for fp8 weights."
                )
            if weight_block_size is not None:
                w = block_quant_dequant(
                    w,
                    weight_scale,
                    weight_block_size,
                    torch.bfloat16,
                )
            else:
                scale = weight_scale.to(torch.float32)
                while scale.ndim < w.ndim:
                    scale = scale.unsqueeze(-1)
                w = (w.to(torch.float32) * scale).to(torch.bfloat16)
        elif w.dtype == torch.int8:
            weight_block_size = getattr(self.quant_config, "weight_block_size", None)
            if weight_block_size is not None:
                weight_scale = getattr(self.legacy_kv_b_proj, "weight_scale_inv", None)
                if weight_scale is None:
                    raise RuntimeError(
                        "legacy_kv runtime branch expected legacy_kv_b_proj."
                        "weight_scale_inv for blockwise int8 weights."
                    )
                w = int8_block_dequant(w, weight_scale, weight_block_size).to(
                    torch.bfloat16
                )
            else:
                weight_scale = getattr(self.legacy_kv_b_proj, "weight_scale", None)
                if weight_scale is None:
                    raise RuntimeError(
                        "legacy_kv runtime branch expected legacy_kv_b_proj."
                        "weight_scale for channelwise int8 weights."
                    )
                scale = weight_scale.to(torch.float32)
                while scale.ndim < w.ndim:
                    scale = scale.unsqueeze(-1)
                w = (w.to(torch.float32) * scale).to(torch.bfloat16)
        else:
            w = w.to(torch.bfloat16)

        w_kc, w_vc = w.unflatten(
            0, (-1, self.qk_nope_head_dim + self.v_head_dim)
        ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.legacy_w_kc = w_kc.transpose(1, 2).contiguous()
        self.legacy_w_vc = w_vc.transpose(1, 2).contiguous()

    def _forward_legacy_kv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.use_legacy_o_proj:
            raise RuntimeError(
                "DeepSeek-V4 legacy_kv runtime branch requires legacy_o_proj "
                "weights as well, but use_legacy_o_proj is false."
            )

        self._materialize_legacy_kv_absorb_weights()
        assert self.legacy_w_kc is not None and self.legacy_w_vc is not None

        q_lora = self._compute_q_a(x, qkv_a=qkv_a)
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        q_nope, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        latent, _ = self.legacy_kv_a_proj_with_mqa(x)
        k_nope, k_pe = torch.split(
            latent,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        k_nope = self.legacy_kv_a_layernorm(k_nope).unsqueeze(1)
        k_pe = k_pe.unsqueeze(1)

        q_nope_out = torch.bmm(
            q_nope.to(torch.bfloat16).transpose(0, 1),
            self.legacy_w_kc,
        ).transpose(0, 1)

        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        attn_output = self.legacy_attn_mqa(
            q_nope_out,
            k_nope,
            k_nope,
            forward_batch,
            q_rope=q_pe,
            k_rope=k_pe,
        )
        attn_output = attn_output.view(-1, self.n_local_heads, self.kv_lora_rank)
        legacy_v = torch.bmm(
            attn_output.to(torch.bfloat16).transpose(0, 1),
            self.legacy_w_vc,
        ).transpose(0, 1)
        o, _ = self.legacy_o_proj(
            legacy_v.flatten(1).contiguous(),
            forward_batch=forward_batch,
        )
        return o

    def _compute_kv_to_cache(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> None:
        """Fused: rmsnorm + RoPE + write directly to FlashMLA paged cache.

        Replaces the bf16-kv-intermediate path. Used everywhere except the DSA
        prefill-CP case (which needs bf16 kv for the cross-rank all-gather).
        """
        if self.use_legacy_kv:
            raise RuntimeError(
                "DeepSeek-V4 legacy_kv runtime branch reached the DSV4 fused "
                "cache-write path. The loader can now materialize q-only "
                "`wqkv_a` for legacy checkpoints, but the request-time "
                "`legacy_kv` bridge is still required before these weights can "
                "run through the fused DSV4 cache path safely."
            )
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        token_to_kv_pool = get_token_to_kv_pool()
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )

    def _compute_kv_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bf16-kv path used by the DSA prefill-CP case (needs all-gather)."""
        if self.use_legacy_kv:
            raise RuntimeError(
                "DeepSeek-V4 legacy_kv runtime branch reached the bf16 DSV4 KV "
                "path, but only the loader-side q-only `wqkv_a` compatibility "
                "has been enabled so far. A request-time legacy_kv bridge is "
                "still required before prefill/decode can proceed."
            )
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        kv = kv.contiguous()
        fused_norm_rope_inplace(
            kv,
            self.kv_norm.weight.data,
            self.eps,
            self.freqs_cis,
            positions,
        )
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        x_linear = x_quant if x_quant is not None else x
        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready: Optional[torch.cuda.Event] = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            qkv_a_ready = current_stream.record_event()

        q_lora = self._compute_q_a(x_linear, qkv_a=qkv_a)
        q_lora_ready = current_stream.record_event()

        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    attn_backend=attn_backend,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        with torch.cuda.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)
            # Fused norm + rope + cache write -- no bf16 KV intermediate.
            self._compute_kv_to_cache(
                x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
            )

        del qkv_a

        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        q = self._compute_q_b(q_lora, positions, q_out)
        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)

        return q

    def _forward_prepare_multi_stream_hip(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:
        """ATOM-style ROCm path: overlap compressors, keep Q/KV on main stream."""
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 1

        current_stream = torch.cuda.current_stream()
        stream_compressor = self.alt_streams[0]
        stream_indexer_compressor = (
            self.alt_streams[1] if len(self.alt_streams) > 1 else None
        )

        if self.compressor is not None:
            stream_compressor.wait_stream(current_stream)
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        if self.indexer is not None and stream_indexer_compressor is not None:
            stream_indexer_compressor.wait_stream(current_stream)
            with torch.cuda.stream(stream_indexer_compressor):
                attn_backend.forward_indexer_compressor(
                    x=x,
                    forward_batch=forward_batch,
                    layer_id=self.indexer.layer_id,
                    compressor=self.indexer.compressor,
                )

        x_linear = x_quant if x_quant is not None else x
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
            qkv_a = None

        if self.use_fused_qk_norm_rope:
            if _is_gfx95_supported:
                q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(
                    q_lora,
                    self.q_norm.weight,
                    self.q_norm.variance_epsilon,
                )
                q, _ = self.wq_b(q_for_wqb)
            else:
                q_lora = self.q_norm(q_lora)
                q, _ = self.wq_b(q_lora)

            kv = (
                qkv_a[..., self.q_lora_rank :]
                if qkv_a is not None
                else self.wkv(x_linear)[0]
            )

            from sglang.srt.layers.fused_qk_norm_rope_store import (
                fused_qk_norm_rope_swa_store,
            )

            token_to_kv_pool = get_token_to_kv_pool()
            swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
            swa_cache = token_to_kv_pool.swa_kv_pool.kv_buffer[self.layer_id]
            swa_page_size = token_to_kv_pool.swa_kv_pool.page_size

            q = fused_qk_norm_rope_swa_store(
                q=q,
                kv=kv,
                q_norm_weight=None,
                kv_norm_weight=self.kv_norm.weight,
                q_rms_eps=self.eps,
                kv_rms_eps=self.eps,
                rope_head_dim=self.qk_rope_head_dim,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                positions=positions,
                swa_cache=swa_cache,
                swa_loc=swa_loc,
                swa_page_size=swa_page_size,
                q_out=q_out,
                dtype=x.dtype,
            )
        else:
            q_lora = self.q_norm(q_lora)
            q = self._compute_q_b(q_lora, positions, q_out)
            self._compute_kv_to_cache(
                x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
            )

        del qkv_a

        if self.indexer is not None:
            current_stream.wait_stream(stream_compressor)
            if stream_indexer_compressor is not None:
                current_stream.wait_stream(stream_indexer_compressor)
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
                skip_compressor=True,
            )
        elif self.compressor is not None:
            current_stream.wait_stream(stream_compressor)

        return q

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        prepare_started_at = time.monotonic()
        x_linear = x_quant if x_quant is not None else x
        _log_deepseek_v4_layer_trace(
            "self_attn_prepare_enter",
            layer_id=self.layer_id,
            tp_rank=self.tp_rank,
            num_tokens=x.shape[0],
            x_shape=tuple(x.shape),
            x_linear_shape=tuple(x_linear.shape),
            compress_ratio=self.compress_ratio,
            use_fused_qk_norm_rope=self.use_fused_qk_norm_rope,
            elapsed_s=round(time.monotonic() - prepare_started_at, 3),
        )
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
            qkv_a = None
        _log_deepseek_v4_layer_trace(
            "self_attn_prepare_q_ready",
            layer_id=self.layer_id,
            tp_rank=self.tp_rank,
            num_tokens=x.shape[0],
            q_lora_shape=tuple(q_lora.shape),
            has_qkv_a=qkv_a is not None,
            elapsed_s=round(time.monotonic() - prepare_started_at, 3),
        )

        use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)
        kv: Optional[torch.Tensor]
        _log_deepseek_v4_layer_trace(
            "self_attn_prepare_branch",
            layer_id=self.layer_id,
            tp_rank=self.tp_rank,
            num_tokens=x.shape[0],
            use_cp=use_cp,
            use_fused_qk_norm_rope=self.use_fused_qk_norm_rope,
            elapsed_s=round(time.monotonic() - prepare_started_at, 3),
        )

        if self.use_fused_qk_norm_rope:

            if _is_gfx95_supported:
                q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(
                    q_lora,
                    self.q_norm.weight,
                    self.q_norm.variance_epsilon,
                )
                q, _ = self.wq_b(q_for_wqb)
            else:
                q_lora = self.q_norm(q_lora)
                q, _ = self.wq_b(q_lora)

            kv = (
                qkv_a[..., self.q_lora_rank :]
                if qkv_a is not None
                else self.wkv(x_linear)[0]
            )

            from sglang.srt.layers.fused_qk_norm_rope_store import (
                fused_qk_norm_rope_swa_store,
            )

            token_to_kv_pool = get_token_to_kv_pool()
            swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
            swa_cache = token_to_kv_pool.swa_kv_pool.kv_buffer[self.layer_id]
            swa_page_size = token_to_kv_pool.swa_kv_pool.page_size

            q = fused_qk_norm_rope_swa_store(
                q=q,
                kv=kv,
                q_norm_weight=None,
                kv_norm_weight=self.kv_norm.weight,
                q_rms_eps=self.eps,
                kv_rms_eps=self.eps,
                rope_head_dim=self.qk_rope_head_dim,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                positions=positions,
                swa_cache=swa_cache,
                swa_loc=swa_loc,
                swa_page_size=swa_page_size,
                q_out=q_out,
                dtype=x.dtype,
            )
            _log_deepseek_v4_layer_trace(
                "self_attn_prepare_fused_qk_done",
                layer_id=self.layer_id,
                tp_rank=self.tp_rank,
                num_tokens=x.shape[0],
                q_shape=tuple(q.shape),
                kv_shape=tuple(kv.shape) if kv is not None else None,
                elapsed_s=round(time.monotonic() - prepare_started_at, 3),
            )

            if use_cp:
                # DSA CP: keep bf16 kv around for the cross-rank all-gather, then
                # write to the FlashMLA cache after gather.
                kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
                kv = cp_all_gather_rerange_output(
                    kv.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                _log_deepseek_v4_layer_trace(
                    "self_attn_prepare_cp_gather_done",
                    layer_id=self.layer_id,
                    tp_rank=self.tp_rank,
                    num_tokens=x.shape[0],
                    kv_shape=tuple(kv.shape),
                    elapsed_s=round(time.monotonic() - prepare_started_at, 3),
                )
        else:
            q_lora = self.q_norm(q_lora)
            q = self._compute_q_b(q_lora, positions, q_out)
            _log_deepseek_v4_layer_trace(
                "self_attn_prepare_q_compute_done",
                layer_id=self.layer_id,
                tp_rank=self.tp_rank,
                num_tokens=x.shape[0],
                q_shape=tuple(q.shape),
                elapsed_s=round(time.monotonic() - prepare_started_at, 3),
            )
            if use_cp:
                # NSA CP: keep bf16 kv around for the cross-rank all-gather, then
                # write to the FlashMLA cache after gather.
                kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)
                kv = cp_all_gather_rerange_output(
                    kv.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                _log_deepseek_v4_layer_trace(
                    "self_attn_prepare_cp_gather_done",
                    layer_id=self.layer_id,
                    tp_rank=self.tp_rank,
                    num_tokens=x.shape[0],
                    kv_shape=tuple(kv.shape),
                    elapsed_s=round(time.monotonic() - prepare_started_at, 3),
                )
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )
                _log_deepseek_v4_layer_trace(
                    "self_attn_prepare_store_cache_done",
                    layer_id=self.layer_id,
                    tp_rank=self.tp_rank,
                    num_tokens=x.shape[0],
                    kv_shape=tuple(kv.shape),
                    elapsed_s=round(time.monotonic() - prepare_started_at, 3),
                )
            else:
                self._compute_kv_to_cache(
                    x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
                )
                kv = None
                _log_deepseek_v4_layer_trace(
                    "self_attn_prepare_kv_to_cache_done",
                    layer_id=self.layer_id,
                    tp_rank=self.tp_rank,
                    num_tokens=x.shape[0],
                    elapsed_s=round(time.monotonic() - prepare_started_at, 3),
                )

        del qkv_a
        _log_deepseek_v4_layer_trace(
            "self_attn_prepare_post_kv",
            layer_id=self.layer_id,
            tp_rank=self.tp_rank,
            num_tokens=x.shape[0],
            q_shape=tuple(q.shape),
            kv_shape=tuple(kv.shape) if kv is not None else None,
            has_indexer=self.indexer is not None,
            has_compressor=self.compressor is not None,
            elapsed_s=round(time.monotonic() - prepare_started_at, 3),
        )

        if self.indexer is not None and self.layer_id not in _DEBUG_DISABLE_INDEXER_LAYERS:
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
            )
            _log_deepseek_v4_layer_trace(
                "self_attn_prepare_indexer_done",
                layer_id=self.layer_id,
                tp_rank=self.tp_rank,
                num_tokens=x.shape[0],
                elapsed_s=round(time.monotonic() - prepare_started_at, 3),
            )
        elif self.indexer is not None:
            _log_deepseek_v4_layer_trace(
                "self_attn_prepare_indexer_skipped_by_debug_env",
                layer_id=self.layer_id,
                tp_rank=self.tp_rank,
                num_tokens=x.shape[0],
                elapsed_s=round(time.monotonic() - prepare_started_at, 3),
            )
        if (
            self.compressor is not None
            and self.layer_id not in _DEBUG_DISABLE_COMPRESSOR_LAYERS
        ):
            attn_backend.forward_core_compressor(
                x,
                forward_batch,
                self.layer_id,
                self.compressor,
            )
            _log_deepseek_v4_layer_trace(
                "self_attn_prepare_compressor_done",
                layer_id=self.layer_id,
                tp_rank=self.tp_rank,
                num_tokens=x.shape[0],
                elapsed_s=round(time.monotonic() - prepare_started_at, 3),
            )
        elif self.compressor is not None:
            _log_deepseek_v4_layer_trace(
                "self_attn_prepare_compressor_skipped_by_debug_env",
                layer_id=self.layer_id,
                tp_rank=self.tp_rank,
                num_tokens=x.shape[0],
                elapsed_s=round(time.monotonic() - prepare_started_at, 3),
            )

        _log_deepseek_v4_layer_trace(
            "self_attn_prepare_return",
            layer_id=self.layer_id,
            tp_rank=self.tp_rank,
            num_tokens=x.shape[0],
            q_shape=tuple(q.shape),
            kv_shape=tuple(kv.shape) if kv is not None else None,
            elapsed_s=round(time.monotonic() - prepare_started_at, 3),
        )

        return q, kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        x_quant=None,
    ) -> torch.Tensor:
        started_at = time.monotonic()
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            # Idle scheduler ticks can legitimately arrive with empty local
            # tensors on every rank under multi-node TP+DP. In that case there
            # is no pending collective to match, so returning early avoids
            # crashing the worker on an otherwise harmless no-op forward.
            if forward_batch.forward_mode.is_idle():
                return x
            assert (
                not self.wo_b.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return x

        if self.use_legacy_kv:
            x_linear = x_quant if x_quant is not None else x
            qkv_a = None
            if self.fuse_wqa_wkv:
                qkv_a, _ = self.wqkv_a(x_linear)
            return self._forward_legacy_kv(
                x_linear,
                positions,
                forward_batch,
                qkv_a=qkv_a,
            )

        attn_backend = get_attn_backend()
        if TYPE_CHECKING:
            assert isinstance(
                attn_backend,
                (DeepseekV4AttnBackend, DeepseekV4HipRadixBackend),
            )

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch))
            and not (_is_hip and self.compressor is None)
        )
        _log_deepseek_v4_layer_trace(
            "self_attn_core_begin",
            layer_id=self.layer_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            num_tokens=x.shape[0],
            x_shape=tuple(x.shape),
            compress_ratio=self.compress_ratio,
            enable_multi_stream=enable_multi_stream,
            use_legacy_kv=self.use_legacy_kv,
            elapsed_s=round(time.monotonic() - started_at, 3),
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.tp_size > 1:
            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
            rank = self.tp_rank
            tp_slice = slice(rank * self.n_local_heads, (rank + 1) * self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]

        if enable_multi_stream:
            # Multi-stream path always fuses cache write into the K kernel,
            # so the bf16 KV intermediate is gone.
            if _is_hip:
                q = self._forward_prepare_multi_stream_hip(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            else:
                q = self._forward_prepare_multi_stream(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            kv = None
        else:
            q, kv = self._forward_prepare(
                x,
                positions,
                forward_batch,
                attn_backend,
                q_out,
                x_quant=x_quant,
            )
        _log_deepseek_v4_layer_trace(
            "self_attn_prepare_done",
            layer_id=self.layer_id,
            tp_rank=self.tp_rank,
            num_tokens=x.shape[0],
            q_shape=tuple(q.shape),
            q_padded_shape=tuple(q_padded.shape) if q_padded is not None else None,
            kv_shape=tuple(kv.shape) if kv is not None else None,
            elapsed_s=round(time.monotonic() - started_at, 3),
        )

        # The cache write is always fused / already done by _forward_prepare* --
        # tell the backend to skip its own store_cache. When `kv is None`
        # (no DSA-CP), pass `q` as a sentinel for the `k is v` assert; the
        # attention path doesn't read it once `save_kv_cache=False`.
        attn_k = kv if kv is not None else q
        _log_deepseek_v4_layer_trace(
            "self_attn_backend_begin",
            layer_id=self.layer_id,
            tp_rank=self.tp_rank,
            num_tokens=x.shape[0],
            backend_type=type(attn_backend).__name__,
            attn_q_shape=(
                tuple(q_padded.shape) if q_padded is not None else tuple(q.shape)
            ),
            attn_k_shape=tuple(attn_k.shape),
            elapsed_s=round(time.monotonic() - started_at, 3),
        )
        o = attn_backend.forward(
            q=q_padded if q_padded is not None else q,
            k=attn_k,
            v=attn_k,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
            save_kv_cache=False,
        )
        _log_deepseek_v4_layer_trace(
            "self_attn_backend_done",
            layer_id=self.layer_id,
            tp_rank=self.tp_rank,
            num_tokens=x.shape[0],
            o_shape=tuple(o.shape),
            elapsed_s=round(time.monotonic() - started_at, 3),
        )
        o = o[:, tp_slice, :]
        fused_rope_inplace(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )

        legacy_o_input = o.reshape(o.shape[0], -1).contiguous()
        if self.use_legacy_o_proj:
            expected_local_input = (
                self.legacy_o_proj.input_size_per_partition
                if self.legacy_o_proj.input_is_parallel
                else self.legacy_o_proj.input_size
            )
            if not self._logged_legacy_o_proj_contract:
                logger.info(
                    "legacy_o_proj forward contract: layer=%s attn_out_shape=%s "
                    "legacy_input_shape=%s expected_local_input=%s "
                    "expected_global_input=%s tp_size=%s",
                    self.layer_id,
                    tuple(o.shape),
                    tuple(legacy_o_input.shape),
                    expected_local_input,
                    self.legacy_o_proj.input_size,
                    self.tp_size,
                )
                self._logged_legacy_o_proj_contract = True
            if legacy_o_input.shape[-1] != expected_local_input:
                raise RuntimeError(
                    "legacy_o_proj input contract mismatch: "
                    f"layer={self.layer_id} got={tuple(legacy_o_input.shape)} "
                    f"expected_last_dim={expected_local_input} "
                    f"global_input={self.legacy_o_proj.input_size} "
                    f"tp_size={self.tp_size}"
                )
            o, _ = self.legacy_o_proj(legacy_o_input, forward_batch=forward_batch)
            return o

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm

            T, G, D = o.shape
            R = self.o_lora_rank
            o_fp8, o_s = sglang_per_token_group_quant_fp8(
                o.reshape(T * G, D).contiguous(),
                group_size=128,
            )
            o_s = deep_gemm.ceil_to_ue8m0(o_s)
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=(1, 1, 128),
            )
            o = output
        else:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("tgd,grd->tgr", o, wo_a)

        o, _ = self.wo_b(o.flatten(1))

        return o


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.self_attn = MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )
        moe_alt_stream = (
            alt_streams[0]
            if (
                alt_streams is not None
                and (_is_cuda or envs.SGLANG_ROCM_USE_MULTI_STREAM.get())
            )
            else None
        )
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=moe_alt_stream,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.rms_norm_eps = config.rms_norm_eps
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.use_fused_mhc_post_pre = _is_fused_mhc_post_pre_enabled()
        self._input_layernorm_weight_bf16 = None
        self._post_attention_layernorm_weight_bf16 = None

    def refresh_mhc_norm_weight_cache(self):
        # Cache bf16 norm weights so the fused path does not allocate/cast per forward.
        self._input_layernorm_weight_bf16 = (
            self.input_layernorm.weight.data.bfloat16().contiguous()
        )
        self._post_attention_layernorm_weight_bf16 = (
            self.post_attention_layernorm.weight.data.bfloat16().contiguous()
        )

    def prewarm_mhc_token_counts(
        self, token_counts: Tuple[int, ...], device: torch.device
    ) -> None:
        paths = (
            (
                "attn",
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm,
            ),
            (
                "ffn",
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.post_attention_layernorm,
            ),
        )

        with torch.inference_mode():
            for num_tokens in token_counts:
                for path_name, hc_fn, hc_scale, hc_base, norm in paths:
                    tic = time.perf_counter()
                    residual = torch.empty(
                        (num_tokens, self.hc_mult, self.hidden_size),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    y, post, comb, _ = self.hc_pre(
                        residual,
                        hc_fn,
                        hc_scale,
                        hc_base,
                        norm=norm,
                    )
                    del residual, y, post, comb
                    torch.cuda.synchronize()
                    logger.info(
                        "DeepSeek V4 MHC prewarm path=%s num_tokens=%s completed in %.3fs",
                        path_name,
                        num_tokens,
                        time.perf_counter() - tic,
                    )

            if self.use_fused_mhc_post_pre:
                for num_tokens in token_counts:
                    for path_name, hc_fn, hc_scale, hc_base, norm in paths:
                        tic = time.perf_counter()
                        # Dummy inputs matching the fused kernel's expected shapes.
                        x = torch.empty(
                            (num_tokens, self.hidden_size),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        residual = torch.empty(
                            (num_tokens, self.hc_mult, self.hidden_size),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        post_mix = torch.empty(
                            (num_tokens, self.hc_mult, 1),
                            dtype=torch.float32,
                            device=device,
                        )
                        comb_mix = torch.empty(
                            (num_tokens, self.hc_mult, self.hc_mult),
                            dtype=torch.float32,
                            device=device,
                        )
                        norm_weight = norm.weight.data.bfloat16().contiguous()
                        mhc_fused_post_pre(
                            x,
                            residual,
                            post_mix,
                            comb_mix,
                            hc_fn,
                            hc_scale,
                            hc_base,
                            self.rms_norm_eps,
                            self.hc_eps,
                            self.hc_eps,
                            _MHC_POST_MULT_VALUE,
                            self.hc_sinkhorn_iters,
                            norm_weight=norm_weight,
                            norm_eps=norm.variance_epsilon,
                        )
                        del x, residual, post_mix, comb_mix, norm_weight
                        torch.cuda.synchronize()
                        logger.info(
                            "DeepSeek V4 MHC fused prewarm path=%s num_tokens=%s completed in %.3fs",
                            path_name,
                            num_tokens,
                            time.perf_counter() - tic,
                        )

    def prewarm_mhc_token_count_buckets(
        self, max_num_tokens: int, device: torch.device
    ) -> Tuple[int, ...]:
        from sglang.srt.layers.mhc import get_mhc_pre_token_count_representatives

        token_counts = get_mhc_pre_token_count_representatives(
            max_num_tokens, self.hc_mult * self.hidden_size
        )
        if not token_counts:
            return token_counts

        logger.info(
            "DeepSeek V4 MHC prewarm max_num_tokens=%s representative token counts: %s",
            max_num_tokens,
            token_counts,
        )
        self.prewarm_mhc_token_counts(token_counts, device)
        return token_counts

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: Optional[nn.Module] = None,
    ):
        """If *norm* is given and the TileLang path is active, the returned
        hidden_states are already post-norm (the norm is fused into the kernel)."""

        @compile_in_capture_mode
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        shape, dtype = x.size(), x.dtype

        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=torch.float32, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=torch.float32, device=x.device
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.srt.layers.mhc import mhc_pre

            norm_kwargs = {}
            if norm is not None:
                norm_kwargs["norm_weight"] = norm.weight.data
                norm_kwargs["norm_eps"] = norm.variance_epsilon

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=_MHC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                **norm_kwargs,
            )
            return y, post.squeeze(-1), comb, norm is not None

        if _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_PRE.get():
            from aiter.ops.mhc import mhc_pre

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=_MHC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            return y, post.squeeze(-1), comb, False

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
                tf32_hc_prenorm_gemm,
            )

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        from sglang.srt.layers.mhc import hc_split_sinkhorn

        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1), False

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):

        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            from sglang.srt.layers.mhc import mhc_post

            return mhc_post(x, residual, post, comb)

        elif _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_POST.get():
            from aiter.ops.mhc import mhc_post

            result = torch.empty_like(residual)
            mhc_post(result, x, residual, post, comb)
            return result

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        @compile_in_capture_mode
        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)

        return hc_post_torch_impl(x, residual, post, comb)

    def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
        prev_residual: Optional[torch.Tensor] = None,
        prev_post: Optional[torch.Tensor] = None,
        prev_comb: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        started_at = time.monotonic()
        if self.layer_id in (43, 44):
            _log_deepseek_v4_layer_trace(
                "layer_forward_entry",
                layer_id=self.layer_id,
                num_tokens=hidden_states.shape[0],
                has_prev_residual=prev_residual is not None,
                has_prev_post=prev_post is not None,
                has_prev_comb=prev_comb is not None,
            )
        use_fused = self.use_fused_mhc_post_pre

        if prev_residual is not None and use_fused:
            residual, post, comb, hidden_states = mhc_fused_post_pre(
                hidden_states,
                prev_residual,
                prev_post,
                prev_comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                _MHC_POST_MULT_VALUE,
                self.hc_sinkhorn_iters,
                norm_weight=(
                    self._input_layernorm_weight_bf16
                    if self._input_layernorm_weight_bf16 is not None
                    else self.input_layernorm.weight.data
                ),
                norm_eps=self.input_layernorm.variance_epsilon,
            )
            x_quant = None
        else:
            residual = hidden_states
            hidden_states, post, comb, norm_fused = self.hc_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm=self.input_layernorm,
            )
            if not norm_fused:
                if _use_aiter and _is_gfx95_supported:
                    x_quant, hidden_states = _fused_rmsnorm_fp8_quant(
                        hidden_states,
                        self.input_layernorm.weight,
                        self.rms_norm_eps,
                    )
                else:
                    hidden_states = self.input_layernorm(hidden_states)
                    x_quant = None
            else:
                x_quant = None

        _log_deepseek_v4_layer_trace(
            "self_attn_begin",
            layer_id=self.layer_id,
            num_tokens=hidden_states.shape[0],
            elapsed_s=round(time.monotonic() - started_at, 3),
        )
        hidden_states = self.self_attn(
            x=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            x_quant=x_quant,
        )
        _log_deepseek_v4_layer_trace(
            "self_attn_done",
            layer_id=self.layer_id,
            num_tokens=hidden_states.shape[0],
            elapsed_s=round(time.monotonic() - started_at, 3),
        )

        if use_fused:
            fused_mhc = try_fused_hc_post_pre(
                hidden_states,
                residual,
                post,
                comb,
                self.hc_ffn_fn.T,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.hc_mult,
                self.rms_norm_eps,
                self.hc_eps,
                _MHC_POST_MULT_VALUE,
                self.hc_sinkhorn_iters,
                _is_gfx95_supported,
            )
            if fused_mhc is not None:
                residual, hidden_states, post, comb, norm_fused = fused_mhc
            else:
                residual, post, comb, hidden_states = mhc_fused_post_pre(
                    hidden_states,
                    residual,
                    post.unsqueeze(-1) if post.ndim == 2 else post,
                    comb,
                    self.hc_ffn_fn,
                    self.hc_ffn_scale,
                    self.hc_ffn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    _MHC_POST_MULT_VALUE,
                    self.hc_sinkhorn_iters,
                    norm_weight=(
                        self._post_attention_layernorm_weight_bf16
                        if self._post_attention_layernorm_weight_bf16 is not None
                        else self.post_attention_layernorm.weight.data
                    ),
                    norm_eps=self.post_attention_layernorm.variance_epsilon,
                )
                norm_fused = True
        else:
            hidden_states = self.hc_post(hidden_states, residual, post, comb)
            residual = hidden_states
            hidden_states, post, comb, norm_fused = self.hc_pre(
                hidden_states,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                norm=self.post_attention_layernorm,
            )
            if not norm_fused:
                hidden_states = self.post_attention_layernorm(hidden_states)

        _use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_attention_dp_size() > 1
            and get_moe_a2a_backend().is_none()
        )
        _use_tp_attn_a2a_scatter = (
            not _use_cp
            and envs.SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER.get()
            and get_attention_tp_size() > 1
            and not get_moe_a2a_backend().is_none()
        )
        if _use_cp:
            if get_moe_a2a_backend().is_none():
                _log_deepseek_v4_layer_trace(
                    "cp_hidden_states_gather_begin",
                    layer_id=self.layer_id,
                    num_tokens=hidden_states.shape[0],
                    elapsed_s=round(time.monotonic() - started_at, 3),
                )
                hidden_states = dsa_cp_gather_hidden_states(hidden_states)
                _log_deepseek_v4_layer_trace(
                    "cp_hidden_states_gather_done",
                    layer_id=self.layer_id,
                    num_tokens=hidden_states.shape[0],
                    elapsed_s=round(time.monotonic() - started_at, 3),
                )
            else:
                assert get_moe_a2a_backend().is_deepep(), (
                    "CP requires DeepEP (moe_a2a_backend == deepep). "
                    "Only DeepEP is tested with CP's per-rank token split."
                )
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
        _a2a_scatter_chunks: Optional[List[torch.Tensor]] = None
        if _use_tp_attn_a2a_scatter:
            s, r = get_attention_tp_size(), get_attention_tp_rank()
            _a2a_scatter_chunks = list(hidden_states.tensor_split(s))
            hidden_states = _a2a_scatter_chunks[r].contiguous()
            input_ids = input_ids.tensor_split(s)[r].contiguous()
            input_ids_global = input_ids_global.tensor_split(s)[r].contiguous()
        _log_deepseek_v4_layer_trace(
            "mlp_begin",
            layer_id=self.layer_id,
            num_tokens=hidden_states.shape[0],
            elapsed_s=round(time.monotonic() - started_at, 3),
        )
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
            use_reduce_scatter=_use_cp,
        )
        _log_deepseek_v4_layer_trace(
            "mlp_done",
            layer_id=self.layer_id,
            num_tokens=hidden_states.shape[0],
            elapsed_s=round(time.monotonic() - started_at, 3),
        )
        if _use_cp and get_moe_a2a_backend().is_none():
            _log_deepseek_v4_layer_trace(
                "cp_hidden_states_reduce_scatter_begin",
                layer_id=self.layer_id,
                num_tokens=hidden_states.shape[0],
                elapsed_s=round(time.monotonic() - started_at, 3),
            )
            hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
            _log_deepseek_v4_layer_trace(
                "cp_hidden_states_reduce_scatter_done",
                layer_id=self.layer_id,
                num_tokens=hidden_states.shape[0],
                elapsed_s=round(time.monotonic() - started_at, 3),
            )
        elif _use_tp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )
            if should_use_dp_reduce_scatterv():
                _log_deepseek_v4_layer_trace(
                    "tp_moe_reduce_scatterv_begin",
                    layer_id=self.layer_id,
                    num_tokens=global_hidden_states.shape[0],
                    elapsed_s=round(time.monotonic() - started_at, 3),
                )
                get_tp_group().reduce_scatterv(
                    global_hidden_states,
                    output=hidden_states,
                    sizes=get_dp_global_num_tokens(),
                )
                _log_deepseek_v4_layer_trace(
                    "tp_moe_reduce_scatterv_done",
                    layer_id=self.layer_id,
                    num_tokens=hidden_states.shape[0],
                    elapsed_s=round(time.monotonic() - started_at, 3),
                )
            else:
                dp_scatter(hidden_states, global_hidden_states, forward_batch)
        if _use_tp_attn_a2a_scatter:
            assert _a2a_scatter_chunks is not None
            gathered = [torch.empty_like(t) for t in _a2a_scatter_chunks]
            _log_deepseek_v4_layer_trace(
                "attn_tp_all_gather_begin",
                layer_id=self.layer_id,
                num_tokens=hidden_states.shape[0],
                elapsed_s=round(time.monotonic() - started_at, 3),
            )
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            _log_deepseek_v4_layer_trace(
                "attn_tp_all_gather_done",
                layer_id=self.layer_id,
                num_tokens=sum(t.shape[0] for t in gathered),
                elapsed_s=round(time.monotonic() - started_at, 3),
            )
            hidden_states = torch.cat(gathered)

        if not use_fused:
            hidden_states = self.hc_post(hidden_states, residual, post, comb)
            # === DUMP PATCH ===
            if _FORK_DUMP_HS and _FORK_DUMP_IDS is not None and hidden_states.shape[0] > 1:
                try:
                    _ids = tuple(input_ids.detach().cpu().tolist())
                    if _ids == _FORK_DUMP_IDS:
                        _FORK_HS_DICT.setdefault(_ids, []).append(
                            hidden_states[-1].detach().float().cpu().clone().unsqueeze(0)
                        )
                except Exception:
                    pass
            # === END DUMP PATCH ===
            _log_deepseek_v4_layer_trace(
                "layer_return",
                layer_id=self.layer_id,
                num_tokens=hidden_states.shape[0],
                elapsed_s=round(time.monotonic() - started_at, 3),
            )
            return hidden_states, None, None, None

        # Return the deferred FFN hc_post state; the next layer consumes it with
        # cross-layer fusion, and the final layer is completed in DeepseekV4Model.
        _log_deepseek_v4_layer_trace(
            "layer_return",
            layer_id=self.layer_id,
            num_tokens=hidden_states.shape[0],
            elapsed_s=round(time.monotonic() - started_at, 3),
        )
        return hidden_states, residual, post, comb


# === CGC cloud-edge handoff: real resume loaders ===
# These replace the dead `from sglang.srt.pipeline_parallel import
# load_hidden_states_from_ref / load_partial_kv_into_forward_batch` calls that
# never existed in this environment. They are the CONSUMER side: resolve a
# "reference" into a usable tensor / KV cache. The producer + transfer channel
# (serialize / zero-copy VRAM / host-side fallback) lives in
# cgc_engine/pd/cgc_handoff_transport.py (Phase 2 M1).
def load_hidden_states_from_ref(ref, device, dtype):
    """Resolve a hidden-states reference into a tensor on `device` with `dtype`.

    `ref` may be:
      - torch.Tensor        -> returned as-is (moved to device/dtype). Used by the
                               M0 in-process self-check (captured tensor) and by the
                               zero-copy VRAM transport (tensor already resident on GPU).
      - str (path)          -> torch.load(path). Host-side fallback transport.
      - bytes / bytearray   -> torch.load(BytesIO(ref)). Serialized fallback transport.
    """
    import io
    import torch

    if torch.is_tensor(ref):
        return ref.to(device=device, dtype=dtype)
    if isinstance(ref, (bytes, bytearray)):
        return torch.load(io.BytesIO(ref), map_location=device, weights_only=True)
    if isinstance(ref, str):
        return torch.load(ref, map_location=device, weights_only=True)
    raise TypeError(
        f"load_hidden_states_from_ref: unsupported ref type {type(ref)!r}"
    )


def load_partial_kv_into_forward_batch(
    partial_kv_ref, forward_batch, finished_layer, model_layers, device
):
    """Load the partial KV cache (layers 0..finished_layer) produced by the peer
    into this instance's KV cache so that subsequent decode steps can attend to
    it.

    M0 (in-process self-check) never sets `partial_kv_ref`, so this is a no-op
    there. The real cross-instance loader (host-side fallback + zero-copy VRAM)
    is implemented in cgc_engine/pd/cgc_handoff_transport.py and wired in for
    Phase 2 M1. Until then, supplying a real ref without the transport backend
    raises a clear error instead of silently producing wrong KV.
    """
    if partial_kv_ref is None:
        return
    # TODO(M1): deserialize `partial_kv_ref` (tensor / path / bytes) and copy the
    # KV for layers 0..finished_layer into forward_batch.token_to_kv_pool.
    raise NotImplementedError(
        "load_partial_kv_into_forward_batch requires the CGC handoff transport "
        "(Phase 2 M1); cross-instance KV is not yet wired."
    )


# Streaming handoff (M1v2): the cloud emits the layer-`cut` hidden_states on
# EVERY forward, not just the prefill. Prefill = step 0; each decode step =
# 1..N. The edge loads the matching step before each of its own forwards. Both
# counters reset when a new request's EXTEND forward is observed
# (forward_mode != 2). This is what makes true layer-split multi-token decode
# correct: each decoded token's layer-(cut+1) input must be the cloud's residual
# stream AFTER layers 0..cut for THAT token, which the edge cannot reproduce by
# embedding the token locally (that would skip layers 0..cut). For deterministic
# greedy decoding the edge and cloud sample identical tokens, so edge step k ==
# cloud step k and the per-step hidden_states line up by induction.
_CGC_EMIT_STEP = 0
_CGC_RESUME_STEP = 0
# Module-level (NOT per-forward) cache + lock for the handoff transport.
# Defined here so it persists across forward() calls: a per-forward local
# would be reset every call and re-bind the same 31000+rank port -> EADDRINUSE.
import threading as _CGC_THREADING

_CGC_TRANSPORTS = {}
_CGC_TRANSPORT_LOCK = _CGC_THREADING.Lock()


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.hidden_size = config.hidden_size
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.rms_norm_eps = config.rms_norm_eps
        use_stream_pool = _is_cuda or (
            _is_hip
            and (
                envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
                or envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            )
        )
        num_alt_streams = 5 if _is_cuda else 2
        self.alt_streams = (
            [torch.cuda.Stream() for _ in range(num_alt_streams)]
            if use_stream_pool
            else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.gemm_output_zero_allocator_size = 0
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        if self.pp_group.is_last_rank:
            hc_dim = hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.use_fused_mhc_post_pre = _is_fused_mhc_post_pre_enabled()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        if x.numel() > 0:
            from sglang.srt.layers.mhc_head import fused_hc_head

            return fused_hc_head(
                x.contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        # --- Gate 2.0 stage 3b: layer-wise resume (finished_layer 接续) ---
        finished_layer: Optional[int] = None,
        hidden_states_ref: Optional[str] = None,
        partial_kv_ref: Optional[str] = None,
        layer_kv_callback: Optional[Callable] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        # --- Gate 2.0 stage 3b: 层级接续模式 ---
        # 当 finished_layer is not None 时，跳过 embedding，从 hidden_states_ref 恢复中间态，
        # 层循环从 finished_layer+1 开始（而非 self.start_layer）。
        layer_resume_mode = finished_layer is not None
        if layer_resume_mode and hidden_states_ref is not None:
            # 从端侧产出的 hidden_states_ref 恢复中间态
            hidden_states = load_hidden_states_from_ref(
                hidden_states_ref, device=input_ids.device, dtype=self.embed_tokens.weight.dtype
            )
            # 注入 hc_mult 维度以匹配后续 mHC 层结构
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
            elif hidden_states.ndim == 3 and hidden_states.shape[1] != self.hc_mult:
                hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        elif self.pp_group.is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            # Unflatten 2D PP IPC tensor back to 3D mHC shape.
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.view(
                    hidden_states.shape[0], self.hc_mult, self.hidden_size
                )

        # --- Gate 2.0 stage 3b: 恢复 partial KV cache（如提供）---
        if layer_resume_mode and partial_kv_ref is not None:
            load_partial_kv_into_forward_batch(
                partial_kv_ref=partial_kv_ref,
                forward_batch=forward_batch,
                finished_layer=int(finished_layer),
                model_layers=self.layers,
                device=input_ids.device,
            )

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        _log_deepseek_v4_layer_trace(
            "input_ids_global_ready",
            layer_id=self.start_layer,
            local_input_ids_shape=getattr(input_ids, "shape", None),
            global_input_ids_shape=getattr(input_ids_global, "shape", None),
            local_input_ids_min=(
                int(input_ids.min().item()) if input_ids is not None and input_ids.numel() else None
            ),
            local_input_ids_max=(
                int(input_ids.max().item()) if input_ids is not None and input_ids.numel() else None
            ),
            global_input_ids_min=(
                int(input_ids_global.min().item())
                if input_ids_global is not None and input_ids_global.numel()
                else None
            ),
            global_input_ids_max=(
                int(input_ids_global.max().item())
                if input_ids_global is not None and input_ids_global.numel()
                else None
            ),
            local_input_ids_window=(
                input_ids[:16].detach().cpu().tolist()
                if input_ids is not None and input_ids.numel()
                else None
            ),
            global_input_ids_window=(
                input_ids_global[:16].detach().cpu().tolist()
                if input_ids_global is not None and input_ids_global.numel()
                else None
            ),
            global_num_tokens_cpu=getattr(forward_batch, "global_num_tokens_cpu", None),
        )

        if dsa_use_prefill_cp(forward_batch):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
            input_ids = cp_round_robin_input_ids(input_ids)
            input_ids_global = input_ids

        # Reset Compressor's per-step freqs_cis cache from any previous step.
        for _attr in ("freqs_cis_c4", "freqs_cis_c128"):
            if hasattr(forward_batch, _attr):
                delattr(forward_batch, _attr)

        use_fused = self.use_fused_mhc_post_pre
        prev_residual, prev_post, prev_comb = None, None, None
        last_layer = None
        setattr(forward_batch, "_cgc_debug_model_current_layer", None)
        setattr(forward_batch, "_cgc_debug_model_last_done_layer", None)
        # --- Gate 2.0 stage 3b: 层循环起始支持 finished_layer 接续 ---
        loop_start = self.start_layer
        if layer_resume_mode and finished_layer is not None:
            loop_start = max(self.start_layer, int(finished_layer) + 1)
        for i in range(loop_start, self.end_layer):
            layer = self.layers[i]
            last_layer = layer
            if i in (43, 44):
                _log_deepseek_v4_layer_trace(
                    "model_layer_handoff_begin",
                    layer_id=i,
                    prev_layer_id=(i - 1),
                    prev_last_done_layer=getattr(
                        forward_batch, "_cgc_debug_model_last_done_layer", None
                    ),
                    num_tokens=hidden_states.shape[0],
                )
            setattr(forward_batch, "_cgc_debug_model_current_layer", i)
            if i in (43, 44):
                _log_deepseek_v4_layer_trace(
                    "model_layer_handoff_ready",
                    layer_id=i,
                    current_layer=getattr(
                        forward_batch, "_cgc_debug_model_current_layer", None
                    ),
                    prev_last_done_layer=getattr(
                        forward_batch, "_cgc_debug_model_last_done_layer", None
                    ),
                    num_tokens=hidden_states.shape[0],
                )
            ctx = (
                nullcontext()
                if not get_global_server_args().disable_piecewise_cuda_graph
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                _log_deepseek_v4_layer_trace(
                    "model_layer_call_begin",
                    layer_id=i,
                    num_tokens=hidden_states.shape[0],
                )
                hidden_states, prev_residual, prev_post, prev_comb = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    input_ids=input_ids,
                    input_ids_global=input_ids_global,
                    prev_residual=prev_residual,
                    prev_post=prev_post,
                    prev_comb=prev_comb,
                )
                _log_deepseek_v4_layer_trace(
                    "model_layer_call_done",
                    layer_id=i,
                    num_tokens=hidden_states.shape[0],
                )
                setattr(forward_batch, "_cgc_debug_model_last_done_layer", i)
                if i == 42:
                    _log_deepseek_v4_layer_trace(
                        "model_layer_handoff_emit",
                        layer_id=i,
                        next_layer_id=43,
                        num_tokens=hidden_states.shape[0],
                    )
            # --- Gate 2.0 stage 4: 逐层 KV 流式推送回调 ---
            if layer_kv_callback is not None:
                layer_kv_callback(i, hidden_states, forward_batch)
        # === DUMP PATCH ===
        if _FORK_DUMP_HS and _FORK_HS_DICT and not _FORK_HS_SAVED[0]:
            try:
                import torch.distributed as _td
                _rk = _td.get_rank() if _td.is_initialized() else 0
                if _rk == 0:
                    import torch as _torch
                    _torch.save(dict(_FORK_HS_DICT), "/data/fork_hs.pt")
                    _FORK_HS_SAVED[0] = True
                    print(
                        "[DUMP PATCH] saved /data/fork_hs.pt layers=",
                        {len(v) for v in _FORK_HS_DICT.values()},
                        flush=True,
                    )
            except Exception as _e:
                print("[DUMP PATCH] save error:", repr(_e), flush=True)
        # === END DUMP PATCH ===
        if use_fused and last_layer is not None:
            hidden_states = last_layer.hc_post(
                hidden_states, prev_residual, prev_post, prev_comb
            )

        # CP all-gather only on the last PP rank; PP IPC carries CP-split tensors.
        if self.pp_group.is_last_rank and dsa_use_prefill_cp(forward_batch):
            _log_deepseek_v4_layer_trace(
                "model_cp_all_gather_begin",
                layer_id=self.num_hidden_layers - 1,
                num_tokens=hidden_states.shape[0],
            )
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
            _log_deepseek_v4_layer_trace(
                "model_cp_all_gather_done",
                layer_id=self.num_hidden_layers - 1,
                num_tokens=hidden_states.shape[0],
            )

        if not self.pp_group.is_last_rank:
            # Flatten 3D mHC tensor for PP IPC.
            return PPProxyTensors({"hidden_states": hidden_states.flatten(1)})

        pre_hc_head = hidden_states.flatten(1)

        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.norm(hidden_states)

        return hidden_states, pre_hc_head


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.model.start_layer, self.model.end_layer)
                if isinstance(
                    self.model.layers[layer_id].mlp, deepseek_v2.DeepseekV2MoE
                )
            }
        )

        # Expose start_layer/end_layer for model_runner PP support
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_rank = get_attention_cp_rank()
            self.cp_size = get_attention_cp_size()

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Waterfill needs shared-experts fusion so it can dispatch shared
        # expert tokens to least-loaded EP ranks.
        if get_global_server_args().enable_deepep_waterfill:
            if self.config.n_shared_experts != 1:
                raise ValueError(
                    "DeepEP Waterfill for DeepSeek V4 expects exactly one shared "
                    f"expert, but got n_shared_experts={self.config.n_shared_experts}."
                )
            self.num_fused_shared_experts = self.config.n_shared_experts
            log_info_on_rank0(
                logger,
                "DeepSeek V4: --enable-deepep-waterfill set; KEEP shared-experts "
                "fusion enabled so waterfill can rebalance shared expert dispatch.",
            )
            return

        get_global_server_args().disable_shared_experts_fusion = True
        log_info_on_rank0(
            logger,
            "DeepSeek V4 requires different clamping for shared and routed experts. "
            "Shared experts fusion optimization is disabled.",
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
                if is_dsa_prefill_cp_round_robin_split():
                    attn_backend = get_attn_backend()
                    metadata = attn_backend.forward_metadata
                    core_meta = metadata.core_attn_metadata
                    core_meta.apply_cp_reindex()
                    core_meta.init_flashmla_related(is_prefill=True)
                    if metadata.indexer_metadata is not None:
                        metadata.indexer_metadata = (
                            attn_backend.init_forward_metadata_indexer(core_meta)
                        )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            # === CGC Phase 2: handoff modes (mutually exclusive) ===
            #  - CGC_SELFCHECK_CUT : in-process resume self-check (M0)
            #  - CGC_EMIT_CUT      : cloud side — capture hidden_states at cut
            #                        layer and persist it for the edge (M1 emit)
            #  - CGC_RESUME_FROM   : edge side — load persisted hidden_states
            #                        and resume the model from that layer (M1 resume)
            import os

            def _cgc_rank():
                # Per-rank handoff filenames so tp>1 is correct whether the
                # residual stream is replicated (all ranks identical) or sharded.
                try:
                    import torch.distributed as _dist

                    if _dist.is_available() and _dist.is_initialized():
                        return int(_dist.get_rank())
                except Exception:
                    pass
                return 0

            # --- CGC handoff transport (M1v2) ---------------------------------
            # Default ("file") keeps the proven per-step torch.save/load path.
            # "tcp" / "nixl" activate the zero-copy / host-side transport defined
            # in cgc_handoff_transport.py (deployed next to this file). Each rank
            # gets its OWN transport endpoint (port = base + rank) so the 8 tp
            # ranks never collide and there is no cross-rank hop.
            # NOTE: _CGC_TRANSPORTS and _CGC_TRANSPORT_LOCK are module-level
            # (see top of file) so the cache persists across forward() calls;
            # a per-forward local would reset every call and re-bind the same
            # 31000+rank port -> EADDRINUSE. Double-checked locking makes
            # creation atomic per (role,rank,mode) under overlap scheduling.

            def _cgc_get_transport(role, rank, mode, host=None):
                _key = (role, rank, mode)
                if _key not in _CGC_TRANSPORTS:
                    with _CGC_TRANSPORT_LOCK:
                        if _key not in _CGC_TRANSPORTS:
                            _port = (
                                int(os.environ.get("CGC_TRANSPORT_TCP_PORT", "31000"))
                                + rank
                            )
                            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                            from cgc_handoff_transport import HandoffTransport as _HT

                            _CGC_TRANSPORTS[_key] = _HT.make(
                                mode,
                                role=role,
                                host="0.0.0.0" if role == "server" else "127.0.0.1",
                                port=_port,
                                connect_host=host
                                or os.environ.get("CGC_TRANSPORT_TCP_HOST", "127.0.0.1"),
                                rank=rank,
                            )
                return _CGC_TRANSPORTS[_key]

            import sys as _sys

            _selfcheck_raw = os.environ.get("CGC_SELFCHECK_CUT", "") or ""
            try:
                _sc_all = [int(x) for x in str(_selfcheck_raw).split(",") if x.strip()]
            except ValueError:
                _sc_all = []
            _emit_cut = int(os.environ.get("CGC_EMIT_CUT", "-1") or -1)
            _resume_cut = int(os.environ.get("CGC_RESUME_FROM", "-1") or -1)
            _end = getattr(self.model, "end_layer", 10**9)

            print(
                f"[CGC_DBG] sc_all={_sc_all} emit_cut={_emit_cut} "
                f"resume_cut={_resume_cut} end={_end} "
                f"start={getattr(self.model, 'start_layer', None)} "
                f"n_layers={len(getattr(self.model, 'layers', []))} "
                f"forward_mode={getattr(forward_batch, 'forward_mode', None)} "
                f"num_tokens={int(input_ids.shape[0])} "
                f"extend_seq_lens={getattr(forward_batch, 'extend_seq_lens', None)} "
                f"seq_lens={getattr(forward_batch, 'seq_lens', None)}",
                flush=True,
            )

            if _sc_all:
                # ---- M0 self-check ----
                if not hasattr(self, "_cgc_sc_remaining"):
                    self._cgc_sc_remaining = [c for c in _sc_all if 1 <= c < _end]
                _sc_cut = (
                    self._cgc_sc_remaining.pop(0)
                    if getattr(self, "_cgc_sc_remaining", None)
                    else 0
                )
                if _sc_cut:
                    _captured = {}

                    def _sc_cb(i, hs, fb):
                        if i == _sc_cut:
                            _captured["hs"] = hs.detach().clone()

                    _base = self.model.forward(
                        input_ids, positions, forward_batch, input_embeds,
                        pp_proxy_tensors, layer_kv_callback=_sc_cb,
                    )
                    _res = self.model.forward(
                        input_ids, positions, forward_batch, input_embeds,
                        pp_proxy_tensors,
                        finished_layer=_sc_cut,
                        hidden_states_ref=_captured.get("hs"),
                    )
                    _base_norm = _base[0] if isinstance(_base, tuple) else _base
                    _res_norm = _res[0] if isinstance(_res, tuple) else _res
                    _d = (
                        _base_norm.float() - _res_norm.float()
                    ).abs().max().item()
                    print(
                        f"[CGC_SELFCHECK] cut={_sc_cut} maxdiff={_d:.6e} "
                        f"-> {'PASS' if _d < 1e-2 else 'FAIL'} "
                        f"(remaining={getattr(self, '_cgc_sc_remaining', [])})",
                        flush=True,
                    )
                    hidden_states = _base
                else:
                    hidden_states = self.model.forward(
                        input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
                    )
            elif 0 <= _emit_cut < _end:
                # ---- M1 emit (cloud side) ----
                # Stream the layer-`cut` hidden_states on EVERY forward. Prefill
                # (EXTEND, forward_mode != 2) resets the step counter to 0; each
                # subsequent decode forward increments it. The edge replays the
                # same step indices so per-token hidden_states stay aligned.
                _emit_base = os.environ.get("CGC_HANDOFF_PATH", "/data/cgc_handoff.pt")
                global _CGC_EMIT_STEP
                _mode = (
                    int(getattr(forward_batch, "forward_mode", 1))
                    if forward_batch is not None
                    else 1
                )
                if _mode != 2:  # EXTEND = start of a new request
                    _CGC_EMIT_STEP = 0
                _step = _CGC_EMIT_STEP
                _CGC_EMIT_STEP += 1
                _transport_mode = os.environ.get("CGC_TRANSPORT", "file").lower()
                _transport = (
                    _cgc_get_transport("server", _cgc_rank(), _transport_mode)
                    if _transport_mode != "file"
                    else None
                )

                def _emit_cb(i, hs, fb):
                    import torch as _t

                    if i == _emit_cut:
                        _ntok = int(hs.shape[0])
                        # NIXL zero-copy needs the tensor to stay on VRAM and be
                        # frozen: clone it so the registered buffer is immutable
                        # while the model keeps mutating the live layer output.
                        # file/tcp serialize off-device, so drop to CPU there.
                        if _transport_mode == "nixl":
                            _hs = hs.detach().clone()
                        else:
                            _hs = hs.detach().cpu().contiguous()
                        _payload = {
                            "finished_layer": int(_emit_cut),
                            "hidden_states": _hs,
                            "step": _step,
                        }
                        if _transport is not None:
                            _transport.send(_cgc_rank(), _step, _payload)
                            _p = f"<{_transport_mode} transport>"
                        else:
                            _p = f"{_emit_base}.rank{_cgc_rank()}.step{_step}"
                            _t.save(_payload, _p)
                        print(
                            f"[CGC_EMIT] cut={_emit_cut} step={_step} "
                            f"rank={_cgc_rank()} tok={_ntok} "
                            f"hidden_states [{tuple(hs.shape)}] -> {_p}",
                            flush=True,
                        )

                hidden_states = self.model.forward(
                    input_ids, positions, forward_batch, input_embeds,
                    pp_proxy_tensors, layer_kv_callback=_emit_cb,
                )
            elif 0 <= _resume_cut < _end:
                # ---- M1 resume (edge side) ----
                # True layer-split streaming (M1v2): the edge NEVER embeds tokens
                # locally for the residual stream. Every forward (prefill AND each
                # decode step) is fed the cloud's layer-`cut` hidden_states for the
                # exact same token. SGLang runs:
                #   EXTEND (forward_mode != 2, full prompt tokens) -> step 0
                #   DECODE (forward_mode == 2, one new token)     -> step k
                # The cloud emits matching step files, so edge step k == cloud step k.
                # The edge's KV cache for layers finished_layer+1..end is primed
                # during EXTEND and appended to during DECODE, exactly as a normal
                # decoder would.
                _resume_base = os.environ.get("CGC_HANDOFF_PATH", "/data/cgc_handoff.pt")
                global _CGC_RESUME_STEP
                _mode = (
                    int(getattr(forward_batch, "forward_mode", 1))
                    if forward_batch is not None
                    else 1
                )
                if _mode != 2:  # EXTEND = start of a new request
                    _CGC_RESUME_STEP = 0
                _step = _CGC_RESUME_STEP
                _CGC_RESUME_STEP += 1
                _transport_mode = os.environ.get("CGC_TRANSPORT", "file").lower()
                _transport = (
                    _cgc_get_transport(
                        "client", _cgc_rank(), _transport_mode,
                        host=os.environ.get("CGC_TRANSPORT_TCP_HOST", "127.0.0.1"),
                    )
                    if _transport_mode != "file"
                    else None
                )
                _hs = None
                if _transport is not None:
                    # Pull the cloud's layer-`cut` hidden_states for THIS step.
                    try:
                        _d = _transport.recv(_cgc_rank(), _step, timeout=120.0)
                        _hs = _d["hidden_states"].to(
                            device=input_ids.device,
                            dtype=self.model.embed_tokens.weight.dtype,
                        )
                        print(
                            f"[CGC_RESUME] from={_resume_cut} step={_step} "
                            f"rank={_cgc_rank()} mode={_mode} "
                            f"loaded hidden_states [{tuple(_hs.shape)}] "
                            f"<- {_transport_mode} transport",
                            flush=True,
                        )
                    except Exception as _e:
                        print(
                            f"[CGC_RESUME] WARN from={_resume_cut} step={_step} "
                            f"rank={_cgc_rank()} mode={_mode} transport recv "
                            f"failed ({_e!r}), full forward fallback",
                            flush=True,
                        )
                else:
                    _p = f"{_resume_base}.rank{_cgc_rank()}.step{_step}"
                    if os.path.exists(_p):
                        import torch as _t

                        _d = _t.load(_p, map_location=input_ids.device, weights_only=True)
                        _hs = _d["hidden_states"].to(
                            device=input_ids.device,
                            dtype=self.model.embed_tokens.weight.dtype,
                        )
                        print(
                            f"[CGC_RESUME] from={_resume_cut} step={_step} "
                            f"rank={_cgc_rank()} mode={_mode} "
                            f"loaded hidden_states [{tuple(_hs.shape)}] <- {_p}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[CGC_RESUME] WARN from={_resume_cut} step={_step} "
                            f"rank={_cgc_rank()} mode={_mode} step file missing, "
                            f"full forward fallback",
                            flush=True,
                        )
                if _hs is not None:
                    hidden_states = self.model.forward(
                        input_ids, positions, forward_batch, input_embeds,
                        pp_proxy_tensors,
                        finished_layer=int(_resume_cut),
                        hidden_states_ref=_hs,
                    )
                else:
                    hidden_states = self.model.forward(
                        input_ids, positions, forward_batch, input_embeds,
                        pp_proxy_tensors,
                        finished_layer=int(_resume_cut),
                    )
                # Slice guard: only triggers if token counts somehow mismatch.
                if isinstance(hidden_states, tuple) and hidden_states[0].shape[0] > int(input_ids.shape[0]):
                    _n = int(input_ids.shape[0])
                    _hs_out, _pre = hidden_states
                    _pre = _pre[-_n:] if _pre is not None else _pre
                    hidden_states = (_hs_out[-_n:], _pre)
            else:
                hidden_states = self.model.forward(
                    input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
                )
        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        hidden_states, pre_hc_head = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=pre_hc_head,
        )

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        from deep_gemm import transform_sf_into_required_layout

        if is_nextn:
            layers = [self.model.decoder]
        else:
            layers = [
                self.model.layers[layer_id]
                for layer_id in range(self.model.start_layer, self.model.end_layer)
            ]
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            if layer.layer_id == self.model.start_layer:
                logger.info(
                    "wo_a scale transform input: layer=%s weight_shape=%s "
                    "scale_shape=%s raw_scale_shape=%s scale_dtype=%s "
                    "format_ue8m0=%s G=%s R=%s D=%s",
                    layer.layer_id,
                    tuple(attn.wo_a.weight.shape),
                    tuple(attn.wo_a.weight_scale_inv.shape),
                    tuple(raw_scale.shape),
                    attn.wo_a.weight_scale_inv.dtype,
                    getattr(attn.wo_a.weight_scale_inv, "format_ue8m0", None),
                    G,
                    R,
                    D,
                )
            attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                raw_scale,
                mn=R,
                k=D,
                recipe=(1, 128, 128),
                num_groups=G,
                is_sfa=False,
            )

    def post_load_weights(self, is_nextn=False, weight_names=None):
        uses_legacy_kv = False
        uses_legacy_o_proj = False
        if weight_names is not None:
            has_wkv = any(".self_attn.wkv." in name for name in weight_names)
            has_kv_a = any(".self_attn.kv_a_proj_with_mqa." in name for name in weight_names)
            has_kv_b = any(".self_attn.kv_b_proj." in name for name in weight_names)
            uses_legacy_kv = any(
                ".self_attn.legacy_kv_" in name for name in weight_names
            )
            has_wo_a = any(".self_attn.wo_a." in name for name in weight_names)
            has_o_proj = any(".self_attn.o_proj." in name for name in weight_names)
            has_wo_b = any(".self_attn.wo_b." in name for name in weight_names)
            uses_legacy_o_proj = any(
                ".self_attn.legacy_o_proj." in name for name in weight_names
            )
            logger.info(
                "post_load_weights attention mapping: has_wkv=%s has_kv_a=%s "
                "has_kv_b=%s has_legacy_kv=%s has_wo_a=%s has_o_proj=%s "
                "has_wo_b=%s has_legacy_o_proj=%s",
                has_wkv,
                has_kv_a,
                has_kv_b,
                uses_legacy_kv,
                has_wo_a,
                has_o_proj,
                has_wo_b,
                uses_legacy_o_proj,
            )
        self.use_legacy_kv = uses_legacy_kv
        self.use_legacy_o_proj = uses_legacy_o_proj
        if _FP8_WO_A_GEMM:
            if uses_legacy_o_proj:
                logger.info(
                    "Skip wo_a FP8 scale setup because legacy_o_proj weights were loaded"
                )
            else:
                self._setup_fp8_wo_a_scales(is_nextn)

        if is_nextn:
            return
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_id]
            self_attn = layer.self_attn
            self_attn.use_legacy_kv = uses_legacy_kv
            self_attn.use_legacy_o_proj = uses_legacy_o_proj
            if uses_legacy_kv:
                self_attn._materialize_legacy_kv_absorb_weights()
            if self_attn.compress_ratio != 0 and not self_attn.compressor.ape_converted:
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()
            layer.refresh_mhc_norm_weight_cache()

    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str, is_nextn: bool = False, num_hidden_layers: Optional[int] = None
    ) -> str:
        if name == "embed.weight":
            return "model.embed_tokens.weight"
        if name == "head.weight":
            return "lm_head.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj.weight_scale_inv"
                    elif rest == "h_proj.scale":
                        rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        if "self_attn" in name:
            name = name.replace(".scale", ".weight_scale_inv")
            # DeepSeek-V4-Flash checkpoints keep DeepSeek-V2 MLA naming for these
            # attention projections/norms, while the local V4 module uses wq*/wkv/wo*
            # and q_norm/kv_norm.
            name = name.replace(".q_a_proj.", ".wq_a.")
            name = name.replace(".q_a_layernorm.", ".q_norm.")
            name = name.replace(".kv_a_layernorm.", ".legacy_kv_a_layernorm.")
            name = name.replace(".kv_a_proj_with_mqa.", ".legacy_kv_a_proj_with_mqa.")
            name = name.replace(".kv_b_proj.", ".legacy_kv_b_proj.")
            name = name.replace(".q_b_proj.", ".wq_b.")
            name = name.replace(".o_proj.", ".legacy_o_proj.")

        name = name.replace(".gate.tid2eid", ".topk.tid2eid")
        name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        if "mlp" in name:
            name = name.replace(".scale", ".weight_scale_inv")

        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        if not envs.SGLANG_OPT_FP8_WO_A_GEMM.get():
            weights = list(weights)
            exists_wo_a_scale = any(n.endswith(".wo_a.scale") for n, t in weights)
            if exists_wo_a_scale:
                logger.info("Execute dequant fp8 wo_a")
                weights = _dequant_fp8_wo_a(weights)
            else:
                logger.info("Skip dequant fp8 wo_a")

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"

        fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]] = {}

        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    # DeepSeek-V4-Flash exposes the shared expert MLP at the
                    # top-level mlp.* names. When shared-expert fusion is
                    # disabled, route those tensors through the dedicated
                    # shared_experts branch so the later expert mapping can
                    # rewrite them into the fused MoE expert slot.
                    if (
                        self.num_fused_shared_experts == 0
                        and ".mlp.shared_experts." not in name
                        and ".mlp.experts." not in name
                        and any(
                            token in name
                            for token in (
                                ".mlp.gate_proj.",
                                ".mlp.up_proj.",
                                ".mlp.down_proj.",
                            )
                        )
                    ):
                        name = name.replace(".mlp.", ".mlp.shared_experts.", 1)
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                continue
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(name)
                            break
                        else:
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            if (
                                name == "model.norm.weight"
                                and not self.pp_group.is_last_rank
                            ):
                                continue
                            if (
                                name.startswith("model.hc_head_")
                                or name == "lm_head.weight"
                            ) and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        loaded_weight,
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            elif fuse_wqa_wkv and (
                                name.endswith(".wq_a.weight")
                                or name.endswith(".wq_a.weight_scale_inv")
                                or name.endswith(".wkv.weight")
                                or name.endswith(".wkv.weight_scale_inv")
                            ):
                                is_q = ".wq_a." in name
                                param_name = name.replace(
                                    ".wq_a." if is_q else ".wkv.", ".wqkv_a."
                                )
                                bucket = cache_wqkv_a_weight.setdefault(param_name, {})
                                shard_key = "q" if is_q else "kv"
                                assert (
                                    shard_key not in bucket
                                ), f"duplicate shard {shard_key} for {param_name}"
                                bucket[shard_key] = loaded_weight
                                if len(bucket) == 2:
                                    fused_weight = torch.cat(
                                        [bucket["q"], bucket["kv"]], dim=0
                                    )
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_wqkv_a_weight.pop(param_name)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    if not name.startswith("mtp"):
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    e.add_note(f"{name=} {loaded_weight.shape=}")
                    raise

            for future in concurrent.futures.as_completed(futures):
                future.result()
            logger.info(
                "DeepSeek-V4 load_weights async futures completed. loaded_params=%s weight_names=%s",
                len(loaded_params),
                len(weight_names),
            )

        assert len(cache_compressor_weight) == 0
        if cache_wqkv_a_weight:
            has_legacy_kv = any(".self_attn.legacy_kv_" in name for name in weight_names)
            has_wkv = any(".self_attn.wkv." in name for name in weight_names)
            if has_legacy_kv and not has_wkv:
                unresolved_params = sorted(cache_wqkv_a_weight.keys())
                logger.info(
                    "Synthesizing q-only fused wqkv_a tensors for legacy_kv "
                    "checkpoint family: unresolved=%s",
                    unresolved_params,
                )
                for param_name, bucket in list(cache_wqkv_a_weight.items()):
                    q_tensor = bucket.get("q")
                    if q_tensor is None:
                        continue
                    if param_name not in params_dict:
                        continue
                    param = params_dict[param_name]
                    fused_tensor = _build_q_only_fused_wqkv_tensor(
                        param=param,
                        q_tensor=q_tensor,
                        param_name=param_name,
                    )
                    weight_loader = auto_weight_loader(param)
                    weight_loader(param, fused_tensor)
                    loaded_params.add(param_name)
                    cache_wqkv_a_weight.pop(param_name)
        if cache_wqkv_a_weight:
            raise RuntimeError(
                _format_unresolved_wqkv_a_contract_error(
                    cache_wqkv_a_weight=cache_wqkv_a_weight,
                    weight_names=weight_names,
                )
            )
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = ["attn_mqa.k_scale", "attn_mqa.v_scale"]
        if not self.pp_group.is_first_rank:
            skipped_checking_patterns.append("embed_tokens")
        if not self.pp_group.is_last_rank:
            skipped_checking_patterns.append("model.norm.")
            skipped_checking_patterns.extend(["lm_head", "hc_head_"])
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if unloaded_params:
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )

        logger.info(
            "DeepSeek-V4 post_load_weights begin. is_nextn=%s unloaded_params=%s",
            is_nextn,
            len(unloaded_params),
        )
        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)
        logger.info("DeepSeek-V4 post_load_weights done. is_nextn=%s", is_nextn)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from einops import rearrange

    assert (
        weight.dtype == torch.float8_e4m3fn
    ), f"expected fp8_e4m3fn, got {weight.dtype}"
    assert scale.dtype in (
        torch.float8_e8m0fnu,
        torch.float32,
    ), f"expected fp8_e8m0fnu or float32, got {scale.dtype}"

    weight_f32 = rearrange(
        weight.float(), "(sn bn) (sk bk) -> sn bn sk bk", bn=128, bk=128
    )
    result = rearrange(
        weight_f32 * scale.float()[:, None, :, None], "sn bn sk bk -> (sn bn) (sk bk)"
    )

    return result.to(torch.bfloat16)


def _dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    weights_dict = dict(weights)

    for name in list(weights_dict.keys()):
        if name not in weights_dict:
            continue
        if not name.endswith(".wo_a.weight"):
            continue
        scale_name = name.replace(".wo_a.weight", ".wo_a.scale")
        assert scale_name in weights_dict
        weight = weights_dict.pop(name)
        scale = weights_dict.pop(scale_name)
        yield name, _dequant_fp8(weight, scale)

    yield from weights_dict.items()
