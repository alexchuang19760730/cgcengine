"""
MetalShaderGenerator - Metal Shader 生成器
==========================================

對標 ds4.c (antirez) 的 17 個 Metal Shader，根據計算圖分析結果自動生成對應的 Metal Shader 代碼。

ds4.c 17 個 Shader 清單:
1. MoE Router - 專家路由計算
2. MoE Expert 2-bit - 稀疏專家（2-bit 量化）
3. MoE Expert Q8 - 共享專家（Q8 量化）
4. Attention GQA - 分組查詢注意力
5. Attention QKV Projection - QKV 投影
6. Attention RoPE - RoPE 位置編碼
7. Attention FlashAttention - Flash Attention 實現
8. FFN SiLU - FFN SiLU 激活
9. FFN 2-bit Quant - FFN 2-bit 量化矩陣乘
10. FFN Q8 Quant - FFN Q8 量化矩陣乘
11. RMSNorm - RMS 歸一化
12. Quantize - 量化解量化
13. Dequantize - 解量化
14. Residual Add - 殘差連接
15. Softmax - Softmax 計算
16. KV Cache - KV 緩存管理
17. Weight mmap - 權重內存映射加載

Author: CGC Engine Team
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class ShaderType(Enum):
    """Metal Shader 類型"""
    MOE_ROUTER = "moe_router"
    MOE_EXPERT_2BIT = "moe_expert_2bit"
    MOE_EXPERT_Q8 = "moe_expert_q8"
    ATTENTION_GQA = "attention_gqa"
    ATTENTION_QKV_PROJ = "attention_qkv_proj"
    ATTENTION_ROPE = "attention_rope"
    ATTENTION_FLASH = "attention_flash"
    FFN_SILU = "ffn_silu"
    FFN_2BIT = "ffn_2bit"
    FFN_Q8 = "ffn_q8"
    RMS_NORM = "rms_norm"
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"
    RESIDUAL_ADD = "residual_add"
    SOFTMAX = "softmax"
    KV_CACHE = "kv_cache"
    WEIGHT_MMAP = "weight_mmap"


@dataclass
class ShaderConfig:
    """Shader 配置"""
    shader_type: ShaderType
    thread_group_size: Tuple[int, int, int] = (32, 1, 1)
    max_shared_memory: int = 65536
    uses_uniform_memory: bool = True
    priority: int = 0


@dataclass
class GraphAnalysisResult:
    """計算圖分析結果"""
    has_moe: bool = False
    has_moe_router: bool = False
    num_experts: int = 0
    num_active_experts: int = 0
    has_gqa: bool = False
    has_flash_attention: bool = False
    has_rope: bool = False
    has_rms_norm: bool = False
    has_quantization: bool = False
    quantization_bits: int = 0
    num_layers: int = 0
    hidden_dim: int = 0
    num_heads: int = 0
    head_dim: int = 0
    seq_len: int = 0
    batch_size: int = 1
    detected_ops: List[str] = field(default_factory=list)


class MetalShaderGenerator:
    """
    Metal Shader 生成器

    根據計算圖分析結果，生成對應的 Metal Shader 代碼
    """

    SHADER_DEFINITIONS: Dict[ShaderType, ShaderConfig] = {
        ShaderType.MOE_ROUTER: ShaderConfig(ShaderType.MOE_ROUTER, (32, 8, 1), priority=10),
        ShaderType.MOE_EXPERT_2BIT: ShaderConfig(ShaderType.MOE_EXPERT_2BIT, (64, 1, 1), priority=9),
        ShaderType.MOE_EXPERT_Q8: ShaderConfig(ShaderType.MOE_EXPERT_Q8, (64, 1, 1), priority=9),
        ShaderType.ATTENTION_GQA: ShaderConfig(ShaderType.ATTENTION_GQA, (32, 8, 1), priority=8),
        ShaderType.ATTENTION_QKV_PROJ: ShaderConfig(ShaderType.ATTENTION_QKV_PROJ, (32, 4, 1), priority=7),
        ShaderType.ATTENTION_ROPE: ShaderConfig(ShaderType.ATTENTION_ROPE, (32, 4, 1), priority=6),
        ShaderType.ATTENTION_FLASH: ShaderConfig(ShaderType.ATTENTION_FLASH, (32, 8, 1), priority=8),
        ShaderType.FFN_SILU: ShaderConfig(ShaderType.FFN_SILU, (64, 1, 1), priority=5),
        ShaderType.FFN_2BIT: ShaderConfig(ShaderType.FFN_2BIT, (64, 1, 1), priority=5),
        ShaderType.FFN_Q8: ShaderConfig(ShaderType.FFN_Q8, (64, 1, 1), priority=5),
        ShaderType.RMS_NORM: ShaderConfig(ShaderType.RMS_NORM, (256, 1, 1), priority=4),
        ShaderType.QUANTIZE: ShaderConfig(ShaderType.QUANTIZE, (256, 1, 1), priority=3),
        ShaderType.DEQUANTIZE: ShaderConfig(ShaderType.DEQUANTIZE, (256, 1, 1), priority=3),
        ShaderType.RESIDUAL_ADD: ShaderConfig(ShaderType.RESIDUAL_ADD, (256, 1, 1), priority=2),
        ShaderType.SOFTMAX: ShaderConfig(ShaderType.SOFTMAX, (32, 8, 1), priority=6),
        ShaderType.KV_CACHE: ShaderConfig(ShaderType.KV_CACHE, (32, 1, 1), priority=7),
        ShaderType.WEIGHT_MMAP: ShaderConfig(ShaderType.WEIGHT_MMAP, (1, 1, 1), priority=10),
    }

    def __init__(self, output_dir: str = "./generated_shaders"):
        self.output_dir = output_dir
        self.generated_shaders: Dict[ShaderType, str] = {}

    def generate(
        self,
        graph_analysis: GraphAnalysisResult,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[ShaderType, str]:
        """
        根據計算圖分析結果生成 Metal Shader

        Args:
            graph_analysis: 計算圖分析結果
            config: 生成配置

        Returns:
            Dict[ShaderType, str] - 生成的 shader 名稱和內容
        """
        config = config or {}
        self.generated_shaders.clear()

        logger.info("[MetalShaderGenerator] === 開始生成 Metal Shader ===")
        logger.info(f"[MetalShaderGenerator] 圖分析結果: {graph_analysis}")

        shaders_needed = self._determine_required_shaders(graph_analysis)
        logger.info(f"[MetalShaderGenerator] 需要生成 {len(shaders_needed)} 個 Shader")

        for shader_type in shaders_needed:
            shader_code = self._generate_shader(shader_type, graph_analysis, config)
            self.generated_shaders[shader_type] = shader_code
            logger.info(f"[MetalShaderGenerator] 生成 Shader: {shader_type.value}")

        self._emit_coverage_report(shaders_needed)

        return self.generated_shaders

    def _determine_required_shaders(
        self,
        graph_analysis: GraphAnalysisResult,
    ) -> List[ShaderType]:
        """根據圖分析結果確定需要生成的 Shader"""
        required = []

        required.append(ShaderType.ATTENTION_QKV_PROJ)
        if graph_analysis.has_gqa:
            required.append(ShaderType.ATTENTION_GQA)
        if graph_analysis.has_flash_attention:
            required.append(ShaderType.ATTENTION_FLASH)
        if graph_analysis.has_rope:
            required.append(ShaderType.ATTENTION_ROPE)
        required.append(ShaderType.SOFTMAX)

        if graph_analysis.has_moe:
            required.append(ShaderType.MOE_ROUTER)
            required.append(ShaderType.MOE_EXPERT_2BIT)
            required.append(ShaderType.MOE_EXPERT_Q8)

        required.append(ShaderType.FFN_SILU)
        if graph_analysis.has_quantization:
            required.append(ShaderType.FFN_2BIT)
            required.append(ShaderType.FFN_Q8)
            required.append(ShaderType.QUANTIZE)
            required.append(ShaderType.DEQUANTIZE)

        required.append(ShaderType.RMS_NORM)
        required.append(ShaderType.RESIDUAL_ADD)
        required.append(ShaderType.KV_CACHE)
        required.append(ShaderType.WEIGHT_MMAP)

        return required

    def _generate_shader(
        self,
        shader_type: ShaderType,
        graph_analysis: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """根據 shader 類型生成對應的 Metal Shader 代碼"""
        generators = {
            ShaderType.MOE_ROUTER: self._generate_moe_router,
            ShaderType.MOE_EXPERT_2BIT: self._generate_moe_expert_2bit,
            ShaderType.MOE_EXPERT_Q8: self._generate_moe_expert_q8,
            ShaderType.ATTENTION_GQA: self._generate_attention_gqa,
            ShaderType.ATTENTION_QKV_PROJ: self._generate_attention_qkv_proj,
            ShaderType.ATTENTION_ROPE: self._generate_attention_rope,
            ShaderType.ATTENTION_FLASH: self._generate_attention_flash,
            ShaderType.FFN_SILU: self._generate_ffn_silu,
            ShaderType.FFN_2BIT: self._generate_ffn_2bit,
            ShaderType.FFN_Q8: self._generate_ffn_q8,
            ShaderType.RMS_NORM: self._generate_rms_norm,
            ShaderType.QUANTIZE: self._generate_quantize,
            ShaderType.DEQUANTIZE: self._generate_dequantize,
            ShaderType.RESIDUAL_ADD: self._generate_residual_add,
            ShaderType.SOFTMAX: self._generate_softmax,
            ShaderType.KV_CACHE: self._generate_kv_cache,
            ShaderType.WEIGHT_MMAP: self._generate_weight_mmap,
        }

        generator = generators.get(shader_type)
        if generator:
            return generator(graph_analysis, config)
        return self._generate_placeholder(shader_type)

    def _generate_moe_router(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 MoE Router Shader - 專家路由計算"""
        num_experts = graph.num_experts or 8
        hidden_dim = graph.hidden_dim or 4096

        return f'''// MoE Router Shader - 專家路由計算
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c moe_router.metal

#include <metal_stdlib>
using namespace metal;

struct MoERouterParams {{
    int batch_size;
    int seq_len;
    int hidden_dim;
    int num_experts;
    int top_k;
    float temperature;
}};

struct MoERouterTensors {{
    device const float* input;      // [batch, seq, hidden]
    device const float* gate_weight; // [num_experts, hidden]
    device float* logits;           // [batch, seq, num_experts]
    device float* probs;            // [batch, seq, num_experts]
    device int* topk_indices;        // [batch, seq, top_k]
    device float* topk_weights;      // [batch, seq, top_k]
}};

kernel void moe_router_kernel(
    MoERouterTensors tensors [[buffer(0)]],
    constant MoERouterParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int s = gid.y;
    const int e = gid.z;

    if (b >= params.batch_size || s >= params.seq_len || e >= params.num_experts) return;

    const int input_offset = (b * params.seq_len + s) * params.hidden_dim;
    const int gate_offset = e * params.hidden_dim;

    // 計算 gate logits: input @ gate_weight.T
    float logit = 0.0f;
    for (int i = 0; i < params.hidden_dim; i++) {{
        logit += tensors.input[input_offset + i] * tensors.gate_weight[gate_offset + i];
    }}

    const int logit_offset = (b * params.seq_len + s) * params.num_experts;
    tensors.logits[logit_offset + e] = logit / params.temperature;

    // Softmax
    if (e == 0) {{
        float sum_exp = 0.0f;
        for (int k = 0; k < params.num_experts; k++) {{
            sum_exp += exp(tensors.logits[logit_offset + k]);
        }}
        for (int k = 0; k < params.num_experts; k++) {{
            tensors.probs[logit_offset + k] = exp(tensors.logits[logit_offset + k]) / sum_exp;
        }}

        // Top-K 選擇
        float topk_vals[{config.get('top_k', 2)}];
        int topk_idx[{config.get('top_k', 2)}];
        for (int k = 0; k < params.top_k; k++) {{
            topk_vals[k] = -INFINITY;
            topk_idx[k] = -1;
        }}

        for (int k = 0; k < params.num_experts; k++) {{
            float prob = tensors.probs[logit_offset + k];
            for (int t = 0; t < params.top_k; t++) {{
                if (prob > topk_vals[t]) {{
                    for (int tt = params.top_k - 1; tt > t; tt--) {{
                        topk_vals[tt] = topk_vals[tt - 1];
                        topk_idx[tt] = topk_idx[tt - 1];
                    }}
                    topk_vals[t] = prob;
                    topk_idx[t] = k;
                    break;
                }}
            }}
        }}

        const int topk_offset = (b * params.seq_len + s) * params.top_k;
        for (int k = 0; k < params.top_k; k++) {{
            tensors.topk_indices[topk_offset + k] = topk_idx[k];
            tensors.topk_weights[topk_offset + k] = topk_vals[k];
        }}
    }}
}}
'''

    def _generate_moe_expert_2bit(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 MoE Expert 2-bit Shader - 稀疏專家（2-bit 量化）"""
        hidden_dim = graph.hidden_dim or 4096

        return f'''// MoE Expert 2-bit Shader - 稀疏專家（2-bit 量化）
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c moe_expert_2bit.metal

#include <metal_stdlib>
using namespace metal;

struct MoEExpert2BitParams {{
    int batch_size;
    int seq_len;
    int hidden_dim;
    int expert_dim;
    int num_experts;
    int top_k;
}};

struct MoEExpert2BitTensors {{
    device const float* input;         // [batch, seq, hidden]
    device const uint8_t* expert_weights; // [num_experts, expert_dim, hidden/4] 2-bit packed
    device const float* shared_expert_scale;
    device int* topk_indices;            // [batch, seq, top_k]
    device float* topk_weights;          // [batch, seq, top_k]
    device float* output;                // [batch, seq, hidden]
}};

// 2-bit 解碼表
constant float k2BitDecode[4] = {{0.0f, 0.333f, 0.667f, 1.0f}};

kernel void moe_expert_2bit_kernel(
    MoEExpert2BitTensors tensors [[buffer(0)]],
    constant MoEExpert2BitParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int s = gid.y;
    const int d = gid.z;

    if (b >= params.batch_size || s >= params.seq_len || d >= params.hidden_dim) return;

    const int input_offset = (b * params.seq_len + s) * params.hidden_dim;
    const int topk_offset = (b * params.seq_len + s) * params.top_k;
    const int output_offset = (b * params.seq_len + s) * params.hidden_dim;

    float result = 0.0f;

    // 只計算被選中的 top-k 專家
    for (int k = 0; k < params.top_k; k++) {{
        const int expert_id = tensors.topk_indices[topk_offset + k];
        const float weight = tensors.topk_weights[topk_offset + k];

        // 解碼 2-bit 權重
        const int weight_offset = expert_id * params.expert_dim * (params.hidden_dim / 4);
        const int pack_idx = d / 4;
        const int bit_offset = (d % 4) * 2;
        const uint8_t packed = tensors.expert_weights[weight_offset + pack_idx];
        const float w = k2BitDecode[(packed >> bit_offset) & 0x3];

        // 矩陣乘: input @ expert_weight
        float sum = 0.0f;
        for (int e = 0; e < params.expert_dim; e++) {{
            sum += tensors.input[input_offset + e] * w;
        }}

        result += weight * sum * tensors.shared_expert_scale[expert_id];
    }}

    tensors.output[output_offset + d] = result;
}}
'''

    def _generate_moe_expert_q8(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 MoE Expert Q8 Shader - 共享專家（Q8 量化）"""
        hidden_dim = graph.hidden_dim or 4096

        return f'''// MoE Expert Q8 Shader - 共享專家（Q8 量化）
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c moe_expert_q8.metal

#include <metal_stdlib>
using namespace metal;

struct MoEExpertQ8Params {{
    int batch_size;
    int seq_len;
    int hidden_dim;
    int num_shared_experts;
}};

struct MoEExpertQ8Tensors {{
    device const float* input;           // [batch, seq, hidden]
    device const uint8_t* expert_weights; // [num_experts, expert_dim, hidden/8] Q8 packed
    device const float* scales;           // [num_experts, hidden]
    device float* output;                  // [batch, seq, hidden]
}};

kernel void moe_expert_q8_kernel(
    MoEExpertQ8Tensors tensors [[buffer(0)]],
    constant MoEExpertQ8Params& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int s = gid.y;
    const int d = gid.z;

    if (b >= params.batch_size || s >= params.seq_len || d >= params.hidden_dim) return;

    const int input_offset = (b * params.seq_len + s) * params.hidden_dim;
    const int output_offset = (b * params.seq_len + s) * params.hidden_dim;

    float result = 0.0f;

    for (int e = 0; e < params.num_shared_experts; e++) {{
        const int weight_offset = e * params.hidden_dim * (params.hidden_dim / 8);
        const int scale_offset = e * params.hidden_dim;

        // Q8 解碼: 實際值 = pack_val * scale
        const int pack_idx = d / 8;
        const int bit_offset = (d % 8) * 8;
        const uint8_t packed_val = tensors.expert_weights[weight_offset + pack_idx];
        const float val = (float)packed_val / 255.0f * tensors.scales[scale_offset + d];

        float sum = 0.0f;
        for (int k = 0; k < params.hidden_dim; k++) {{
            const int q8_offset = weight_offset + (k * params.hidden_dim / 8) + (d / 8);
            const uint8_t q8_byte = tensors.expert_weights[q8_offset];
            const int q8_bit = (d % 8) * 8;
            const float q8_val = ((q8_byte >> q8_bit) & 0xFF) / 255.0f * tensors.scales[scale_offset + k];
            sum += tensors.input[input_offset + k] * q8_val;
        }}

        result += sum;
    }}

    tensors.output[output_offset + d] = result;
}}
'''

    def _generate_attention_gqa(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 GQA Attention Shader"""
        num_heads = graph.num_heads or 32
        head_dim = graph.head_dim or 128
        num_kv_heads = max(1, num_heads // 4)

        return f'''// GQA Attention Shader - 分組查詢注意力
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c attention_gqa.metal

#include <metal_stdlib>
using namespace metal;

structGQAParams {{
    int batch_size;
    int seq_len;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    float scale;
}};

structGQATensors {{
    device const float* Q;   // [batch, num_heads, seq, head_dim]
    device const float* K;   // [batch, num_kv_heads, seq, head_dim]
    device const float* V;   // [batch, num_kv_heads, seq, head_dim]
    device float* O;         // [batch, num_heads, seq, head_dim]
    device float* attn_scores; // [batch, num_heads, seq, seq]
}};

kernel void gqa_attention_kernel(
    GQATensors tensors [[buffer(0)]],
    constant GQAParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int h = gid.y;
    const int t = gid.z;

    if (b >= params.batch_size || h >= params.num_heads || t >= params.seq_len) return;

    const int kv_head = h / (params.num_heads / params.num_kv_heads);
    const float scale = 1.0f / sqrt((float)params.head_dim);

    const int q_offset = ((b * params.num_heads + h) * params.seq_len + t) * params.head_dim;
    const int kv_offset = ((b * params.num_kv_heads + kv_head) * params.seq_len) * params.head_dim;

    // 計算 attention score
    float max_score = -INFINITY;
    float exp_sum = 0.0f;
    float attn_output[128];

    for (int d = 0; d < params.head_dim; d++) {{
        attn_output[d] = 0.0f;
    }}

    // Flash Attention 風格的 online softmax
    for (int s = 0; s <= t; s++) {{
        float score = 0.0f;
        for (int d = 0; d < params.head_dim; d++) {{
            score += tensors.Q[q_offset + d] * tensors.K[kv_offset + s * params.head_dim + d];
        }}
        score *= scale;

        if (score > max_score) {{
            float exp_sum_scaled = exp_sum * exp(max_score - score);
            max_score = score;
            exp_sum = 1.0f + exp_sum_scaled;
            for (int d = 0; d < params.head_dim; d++) {{
                attn_output[d] = tensors.V[kv_offset + s * params.head_dim + d] + exp_sum_scaled * attn_output[d];
            }}
        }} else {{
            float exp_score = exp(score - max_score);
            exp_sum += exp_score;
            for (int d = 0; d < params.head_dim; d++) {{
                attn_output[d] += exp_score * tensors.V[kv_offset + s * params.head_dim + d];
            }}
        }}
    }}

    // Normalize
    const int o_offset = ((b * params.num_heads + h) * params.seq_len + t) * params.head_dim;
    for (int d = 0; d < params.head_dim; d++) {{
        tensors.O[o_offset + d] = attn_output[d] / exp_sum;
    }}
}}
'''

    def _generate_attention_qkv_proj(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 Attention QKV Projection Shader"""
        hidden_dim = graph.hidden_dim or 4096
        num_heads = graph.num_heads or 32
        head_dim = graph.head_dim or 128

        return f'''// Attention QKV Projection Shader
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c attention_qkv_proj.metal

#include <metal_stdlib>
using namespace metal;

struct QKVProjParams {{
    int batch_size;
    int seq_len;
    int hidden_dim;
    int num_heads;
    int head_dim;
}};

struct QKVProjTensors {{
    device const float* input;    // [batch, seq, hidden]
    device const float* W_q;       // [hidden, num_heads * head_dim]
    device const float* W_k;       // [hidden, num_heads * head_dim]
    device const float* W_v;       // [hidden, num_heads * head_dim]
    device float* Q;               // [batch, num_heads, seq, head_dim]
    device float* K;               // [batch, num_heads, seq, head_dim]
    device float* V;               // [batch, num_heads, seq, head_dim]
}};

kernel void qkv_proj_kernel(
    QKVProjTensors tensors [[buffer(0)]],
    constant QKVProjParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int s = gid.y;
    const int h = gid.z;

    if (b >= params.batch_size || s >= params.seq_len || h >= params.num_heads) return;

    const int input_offset = (b * params.seq_len + s) * params.hidden_dim;
    const int q_offset = ((b * params.num_heads + h) * params.seq_len + s) * params.head_dim;
    const int k_offset = ((b * params.num_heads + h) * params.seq_len + s) * params.head_dim;
    const int v_offset = ((b * params.num_heads + h) * params.seq_len + s) * params.head_dim;

    // Q 投影
    for (int d = 0; d < params.head_dim; d++) {{
        float sum = 0.0f;
        for (int i = 0; i < params.hidden_dim; i++) {{
            sum += tensors.input[input_offset + i] * tensors.W_q[h * params.head_dim + d];
        }}
        tensors.Q[q_offset + d] = sum;
    }}

    // K 投影
    for (int d = 0; d < params.head_dim; d++) {{
        float sum = 0.0f;
        for (int i = 0; i < params.hidden_dim; i++) {{
            sum += tensors.input[input_offset + i] * tensors.W_k[h * params.head_dim + d];
        }}
        tensors.K[k_offset + d] = sum;
    }}

    // V 投影
    for (int d = 0; d < params.head_dim; d++) {{
        float sum = 0.0f;
        for (int i = 0; i < params.hidden_dim; i++) {{
            sum += tensors.input[input_offset + i] * tensors.W_v[h * params.head_dim + d];
        }}
        tensors.V[v_offset + d] = sum;
    }}
}}
'''

    def _generate_attention_rope(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 RoPE (Rotary Position Encoding) Shader"""
        head_dim = graph.head_dim or 128

        return f'''// RoPE Shader - 旋轉位置編碼
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c attention_rope.metal

#include <metal_stdlib>
using namespace metal;

struct RoPEParams {{
    int batch_size;
    int seq_len;
    int num_heads;
    int head_dim;
    float base;  // 通常是 10000 或 500000
}};

struct RoPETensors {{
    device float* Q;  // [batch, num_heads, seq, head_dim] in-place
    device float* K;  // [batch, num_heads, seq, head_dim] in-place
    device const float* freqs_cis;  // 預計算的旋轉角度 [seq, head_dim/2]
}};

kernel void rope_kernel(
    RoPETensors tensors [[buffer(0)]],
    constant RoPEParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int h = gid.y;
    const int s = gid.z;

    if (b >= params.batch_size || h >= params.num_heads || s >= params.seq_len) return;

    const int offset = ((b * params.num_heads + h) * params.seq_len + s);
    const int half_dim = params.head_dim / 2;

    for (int i = 0; i < half_dim; i++) {{
        float freq = 1.0f / pow(params.base, (2.0f * i) / params.head_dim);
        float theta = freq * s;

        float cos_theta = cos(theta);
        float sin_theta = sin(theta);

        const int idx0 = offset * params.head_dim + i;
        const int idx1 = offset * params.head_dim + i + half_dim;

        float q0 = tensors.Q[idx0];
        float q1 = tensors.Q[idx1];
        tensors.Q[idx0] = q0 * cos_theta - q1 * sin_theta;
        tensors.Q[idx1] = q0 * sin_theta + q1 * cos_theta;

        float k0 = tensors.K[idx0];
        float k1 = tensors.K[idx1];
        tensors.K[idx0] = k0 * cos_theta - k1 * sin_theta;
        tensors.K[idx1] = k0 * sin_theta + k1 * cos_theta;
    }}
}}
'''

    def _generate_attention_flash(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 Flash Attention Shader"""
        head_dim = graph.head_dim or 128

        return f'''// Flash Attention Shader
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c attention_flash.metal

#include <metal_stdlib>
using namespace metal;

struct FlashAttnParams {{
    int batch_size;
    int seq_len;
    int num_heads;
    int head_dim;
    float scale;
    int block_size;
}};

struct FlashAttnTensors {{
    device const float* Q;   // [batch, num_heads, seq, head_dim]
    device const float* K;   // [batch, num_heads, seq, head_dim]
    device const float* V;   // [batch, num_heads, seq, head_dim]
    device float* O;         // [batch, num_heads, seq, head_dim]
    device float* L;          // [batch, num_heads, seq] logsumexp
}};

kernel void flash_attention_kernel(
    FlashAttnTensors tensors [[buffer(0)]],
    constant FlashAttnParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int h = gid.y;
    const int t = gid.z;

    if (b >= params.batch_size || h >= params.num_heads || t >= params.seq_len) return;

    const int q_offset = ((b * params.num_heads + h) * params.seq_len + t) * params.head_dim;

    // Flash Attention 算法
    float m = -INFINITY;
    float l = 0.0f;
    float acc[128] = {{0.0f}};

    // Block-wise processing
    for (int block = 0; block < t; block += params.block_size) {{
        const int k_offset = ((b * params.num_heads + h) * params.seq_len + block) * params.head_dim;

        // 計算 QK^T
        float score_max = -INFINITY;
        for (int s = block; s < min(t, block + params.block_size); s++) {{
            float score = 0.0f;
            for (int d = 0; d < params.head_dim; d++) {{
                score += tensors.Q[q_offset + d] * tensors.K[k_offset + s * params.head_dim + d];
            }}
            score *= params.scale;
            if (score > score_max) score_max = score;
        }}

        // Online softmax
        float l_block = 0.0f;
        for (int s = block; s < min(t, block + params.block_size); s++) {{
            float score = 0.0f;
            for (int d = 0; d < params.head_dim; d++) {{
                score += tensors.Q[q_offset + d] * tensors.K[k_offset + s * params.head_dim + d];
            }}
            score *= params.scale;
            score = exp(score - score_max);

            l_block += score;
            for (int d = 0; d < params.head_dim; d++) {{
                acc[d] += score * tensors.V[k_offset + s * params.head_dim + d];
            }}
        }}

        float l_old = l;
        l = l_old * exp(m - score_max) + l_block;
        m = score_max + log(exp(m - score_max) + l_block / l);

        float l_old_scaled = l_old * exp(m - score_max);
        for (int d = 0; d < params.head_dim; d++) {{
            acc[d] = (l_old_scaled * acc[d] + l_block * acc[d]) / l;
        }}
    }}

    // 寫入輸出
    const int o_offset = ((b * params.num_heads + h) * params.seq_len + t) * params.head_dim;
    const int l_offset = (b * params.num_heads + h) * params.seq_len + t;

    for (int d = 0; d < params.head_dim; d++) {{
        tensors.O[o_offset + d] = acc[d];
    }}
    tensors.L[l_offset] = m - log(l);
}}
'''

    def _generate_ffn_silu(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 FFN SiLU Shader"""
        hidden_dim = graph.hidden_dim or 4096
        ffn_dim = hidden_dim * 4 // 3

        return f'''// FFN SiLU Shader - FFN SiLU 激活
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c ffn_silu.metal

#include <metal_stdlib>
using namespace metal;

struct FFNSiLUParams {{
    int batch_size;
    int seq_len;
    int hidden_dim;
    int ffn_dim;
}};

struct FFNSiLUTensors {{
    device const float* input;     // [batch, seq, hidden]
    device const float* up_weight; // [hidden, ffn_dim]
    device const float* gate_weight; // [hidden, ffn_dim]
    device float* output;           // [batch, seq, ffn_dim]
}};

// SiLU / Swish 激活函數
inline float silu(float x) {{
    return x / (1.0f + exp(-x));
}}

kernel void ffn_silu_kernel(
    FFNSiLUTensors tensors [[buffer(0)]],
    constant FFNSiLUParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int s = gid.y;
    const int d = gid.z;

    if (b >= params.batch_size || s >= params.seq_len || d >= params.ffn_dim) return;

    const int input_offset = (b * params.seq_len + s) * params.hidden_dim;
    const int output_offset = (b * params.seq_len + s) * params.ffn_dim;

    // 計算 gate: silu(input) * up
    float gate = 0.0f;
    for (int i = 0; i < params.hidden_dim; i++) {{
        gate += tensors.input[input_offset + i] * tensors.gate_weight[i * params.ffn_dim + d];
    }}
    gate = silu(gate);

    float up = 0.0f;
    for (int i = 0; i < params.hidden_dim; i++) {{
        up += tensors.input[input_offset + i] * tensors.up_weight[i * params.ffn_dim + d];
    }}

    tensors.output[output_offset + d] = gate * up;
}}
'''

    def _generate_ffn_2bit(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 FFN 2-bit 量化矩陣乘 Shader"""
        return '''// FFN 2-bit Quantized Matrix Multiply Shader
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c ffn_2bit.metal

#include <metal_stdlib>
using namespace metal;

struct FFN2BitParams {
    int batch_size;
    int seq_len;
    int hidden_dim;
    int ffn_dim;
};

struct FFN2BitTensors {
    device const float* input;
    device const uint8_t* weights;  // [ffn_dim, hidden/4] 2-bit packed
    device const float* scales;
    device float* output;
};

constant float k2BitDecodeTable[4] = {0.0f, 0.333f, 0.667f, 1.0f};

kernel void ffn_2bit_kernel(
    FFN2BitTensors tensors [[buffer(0)]],
    constant FFN2BitParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int b = gid.x;
    const int s = gid.y;
    const int d = gid.z;

    if (b >= params.batch_size || s >= params.seq_len || d >= params.ffn_dim) return;

    const int input_offset = (b * params.seq_len + s) * params.hidden_dim;
    const int output_offset = (b * params.seq_len + s) * params.ffn_dim;

    float sum = 0.0f;
    for (int i = 0; i < params.hidden_dim; i++) {
        const int pack_idx = i / 4;
        const int bit_offset = (i % 4) * 2;
        const uint8_t packed = tensors.weights[d * (params.hidden_dim / 4) + pack_idx];
        const float w = k2BitDecodeTable[(packed >> bit_offset) & 0x3] * tensors.scales[d];
        sum += tensors.input[input_offset + i] * w;
    }

    tensors.output[output_offset + d] = sum;
}
'''

    def _generate_ffn_q8(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 FFN Q8 量化矩陣乘 Shader"""
        return '''// FFN Q8 Quantized Matrix Multiply Shader
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c ffn_q8.metal

#include <metal_stdlib>
using namespace metal;

struct FFNQ8Params {
    int batch_size;
    int seq_len;
    int hidden_dim;
    int ffn_dim;
};

struct FFNQ8Tensors {
    device const float* input;
    device const uint8_t* weights;  // [ffn_dim, hidden/8] Q8 packed
    device const float* scales;
    device float* output;
};

kernel void ffn_q8_kernel(
    FFNQ8Tensors tensors [[buffer(0)]],
    constant FFNQ8Params& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int b = gid.x;
    const int s = gid.y;
    const int d = gid.z;

    if (b >= params.batch_size || s >= params.seq_len || d >= params.ffn_dim) return;

    const int input_offset = (b * params.seq_len + s) * params.hidden_dim;
    const int output_offset = (b * params.seq_len + s) * params.ffn_dim;

    float sum = 0.0f;
    for (int i = 0; i < params.hidden_dim; i++) {
        const int q8_idx = d * (params.hidden_dim / 8) + (i / 8);
        const uint8_t q8_val = tensors.weights[q8_idx];
        const float w = ((float)q8_val - 128.0f) / 127.0f * tensors.scales[d];
        sum += tensors.input[input_offset + i] * w;
    }

    tensors.output[output_offset + d] = sum;
}
'''

    def _generate_rms_norm(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 RMSNorm Shader"""
        hidden_dim = graph.hidden_dim or 4096

        return f'''// RMSNorm Shader - RMS 歸一化
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c rms_norm.metal

#include <metal_stdlib>
using namespace metal;

struct RMSNormParams {{
    int batch_size;
    int seq_len;
    int hidden_dim;
    float eps;
}};

struct RMSNormTensors {{
    device const float* input;     // [batch, seq, hidden]
    device const float* weight;    // [hidden]
    device float* output;          // [batch, seq, hidden]
}};

kernel void rms_norm_kernel(
    RMSNormTensors tensors [[buffer(0)]],
    constant RMSNormParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int s = gid.y;
    const int d = gid.z;

    if (b >= params.batch_size || s >= params.seq_len || d >= params.hidden_dim) return;

    // 計算 RMS
    float sum_sq = 0.0f;
    const int offset = (b * params.seq_len + s) * params.hidden_dim;

    for (int i = 0; i < params.hidden_dim; i++) {{
        float x = tensors.input[offset + i];
        sum_sq += x * x;
    }}
    float rms = rsqrt(sum_sq / params.hidden_dim + params.eps);

    // 應用歸一化和權重
    tensors.output[offset + d] = tensors.input[offset + d] * rms * tensors.weight[d];
}}
'''

    def _generate_quantize(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 Quantize Shader"""
        bits = graph.quantization_bits or 8

        return f'''// Quantize Shader - 量化
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c quantize.metal

#include <metal_stdlib>
using namespace metal;

struct QuantizeParams {{
    int size;
    int bits;
    float scale;
}};

struct QuantizeTensors {{
    device const float* input;   // [size]
    device uint8_t* output;       // [size / (8/bits)]
}};

kernel void quantize_kernel(
    QuantizeTensors tensors [[buffer(0)]],
    constant QuantizeParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    const int pack_size = 8 / params.bits;
    const int idx = gid * pack_size;

    if (idx + pack_size > params.size) return;

    uint8_t packed = 0;
    for (int i = 0; i < pack_size; i++) {{
        float val = tensors.input[idx + i] / params.scale;
        uint8_t quantized = clamp((int)(val * 127 + 128), 0, 255);
        packed |= quantized << (i * params.bits);
    }}

    tensors.output[gid] = packed;
}}
'''

    def _generate_dequantize(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 Dequantize Shader"""
        bits = graph.quantization_bits or 8

        return f'''// Dequantize Shader - 解量化
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c dequantize.metal

#include <metal_stdlib>
using namespace metal;

struct DequantizeParams {{
    int size;
    int bits;
    float scale;
}};

struct DequantizeTensors {{
    device const uint8_t* input;  // [size / (8/bits)]
    device const float* scales;
    device float* output;          // [size]
}};

kernel void dequantize_kernel(
    DequantizeTensors tensors [[buffer(0)]],
    constant DequantizeParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    const int pack_size = 8 / params.bits;
    const int idx = gid * pack_size;

    if (idx + pack_size > params.size) return;

    const uint8_t packed = tensors.input[gid];

    for (int i = 0; i < pack_size; i++) {{
        uint8_t val = (packed >> (i * params.bits)) & ((1 << params.bits) - 1);
        tensors.output[idx + i] = ((float)val - 128.0f) / 127.0f * tensors.scales[idx + i];
    }}
}}
'''

    def _generate_residual_add(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 Residual Add Shader"""
        hidden_dim = graph.hidden_dim or 4096

        return f'''// Residual Add Shader - 殘差連接
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c residual_add.metal

#include <metal_stdlib>
using namespace metal;

struct ResidualAddParams {{
    int batch_size;
    int seq_len;
    int hidden_dim;
    float alpha;  // 殘差係數
}};

struct ResidualAddTensors {{
    device const float* input;    // [batch, seq, hidden]
    device const float* residual; // [batch, seq, hidden]
    device float* output;         // [batch, seq, hidden]
}};

kernel void residual_add_kernel(
    ResidualAddTensors tensors [[buffer(0)]],
    constant ResidualAddParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int s = gid.y;
    const int d = gid.z;

    if (b >= params.batch_size || s >= params.seq_len || d >= params.hidden_dim) return;

    const int offset = (b * params.seq_len + s) * params.hidden_dim;
    tensors.output[offset + d] = tensors.input[offset + d] + params.alpha * tensors.residual[offset + d];
}}
'''

    def _generate_softmax(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 Softmax Shader"""
        return '''// Softmax Shader
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c softmax.metal

#include <metal_stdlib>
using namespace metal;

struct SoftmaxParams {
    int batch_size;
    int num_heads;
    int seq_len;
    float scale;
};

struct SoftmaxTensors {
    device const float* input;   // [batch, num_heads, seq, seq]
    device float* output;         // [batch, num_heads, seq, seq]
};

kernel void softmax_kernel(
    SoftmaxTensors tensors [[buffer(0)]],
    constant SoftmaxParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int b = gid.x;
    const int h = gid.y;
    const int t = gid.z;

    if (b >= params.batch_size || h >= params.num_heads || t >= params.seq_len) return;

    const int row_offset = ((b * params.num_heads + h) * params.seq_len + t) * params.seq_len;

    // 找最大值
    float max_val = -INFINITY;
    for (int s = 0; s < params.seq_len; s++) {
        float val = tensors.input[row_offset + s] * params.scale;
        if (val > max_val) max_val = val;
    }

    // 計算 exp 并求和
    float sum_exp = 0.0f;
    for (int s = 0; s < params.seq_len; s++) {
        float val = tensors.input[row_offset + s] * params.scale;
        sum_exp += exp(val - max_val);
    }

    // 歸一化
    for (int s = 0; s < params.seq_len; s++) {
        float val = tensors.input[row_offset + s] * params.scale;
        tensors.output[row_offset + s] = exp(val - max_val) / sum_exp;
    }
}
'''

    def _generate_kv_cache(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 KV Cache Shader"""
        num_heads = graph.num_heads or 32
        head_dim = graph.head_dim or 128
        max_seq_len = graph.seq_len or 4096

        return f'''// KV Cache Shader - KV 緩存管理
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c kv_cache.metal

#include <metal_stdlib>
using namespace metal;

struct KVCacheParams {{
    int batch_size;
    int num_heads;
    int head_dim;
    int max_seq_len;
    int cache_pos;  // 當前位置
}};

struct KVCacheTensors {{
    device const float* K_new;   // [batch, num_heads, 1, head_dim]
    device const float* V_new;   // [batch, num_heads, 1, head_dim]
    device float* K_cache;       // [batch, num_heads, max_seq_len, head_dim]
    device float* V_cache;       // [batch, num_heads, max_seq_len, head_dim]
    device float* K_out;          // [batch, num_heads, current_len, head_dim]
    device float* V_out;          // [batch, num_heads, current_len, head_dim]
}};

kernel void kv_cache_update_kernel(
    KVCacheTensors tensors [[buffer(0)]],
    constant KVCacheParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    const int b = gid.x;
    const int h = gid.y;
    const int d = gid.z;

    if (b >= params.batch_size || h >= params.num_heads || d >= params.head_dim) return;

    // 更新 cache
    const int cache_offset = ((b * params.num_heads + h) * params.max_seq_len + params.cache_pos) * params.head_dim;
    tensors.K_cache[cache_offset + d] = tensors.K_new[((b * params.num_heads + h) * 1 + 0) * params.head_dim + d];
    tensors.V_cache[cache_offset + d] = tensors.V_new[((b * params.num_heads + h) * 1 + 0) * params.head_dim + d];

    // 讀取完整的 KV 序列
    for (int s = 0; s <= params.cache_pos; s++) {{
        const int in_offset = ((b * params.num_heads + h) * params.max_seq_len + s) * params.head_dim;
        const int out_offset = ((b * params.num_heads + h) * (params.cache_pos + 1) + s) * params.head_dim;
        tensors.K_out[out_offset + d] = tensors.K_cache[in_offset + d];
        tensors.V_out[out_offset + d] = tensors.V_cache[in_offset + d];
    }}
}}
'''

    def _generate_weight_mmap(
        self,
        graph: GraphAnalysisResult,
        config: Dict[str, Any],
    ) -> str:
        """生成 Weight mmap Shader - 權重內存映射加載"""
        return '''// Weight mmap Shader - 權重內存映射加載
// 自動生成 by MetalShaderGenerator
// 對標 ds4.c weight_mmap.metal

#include <metal_stdlib>
using namespace metal;

// 這個 Shader 主要用於驗證 mmap 載入的權重是否正確
// 實際上 ds4.c 的 mmap 權重是直接通過 CPU 載入到 GPU 緩沖區的

struct WeightMmapParams {
    int num_weights;
    int weight_dim;
    int alignment;  // 256 字節對齊
};

struct WeightMmapTensors {
    device const void* mmap_base;  // mmap 起始地址
    device float* verify_output;   // 驗證輸出
};

kernel void weight_verify_kernel(
    WeightMmapTensors tensors [[buffer(0)]],
    constant WeightMmapParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_weights) return;

    // 驗證權重對齊
    const device uint8_t* bytes = (const device uint8_t*)tensors.mmap_base;
    uint8_t checksum = 0;
    for (int i = 0; i < params.weight_dim * sizeof(float); i++) {
        checksum += bytes[gid * params.alignment + i];
    }
    tensors.verify_output[gid] = (float)checksum / 255.0f;
}
'''

    def _generate_placeholder(self, shader_type: ShaderType) -> str:
        """生成 placeholder shader"""
        return f'''// Placeholder Shader for {shader_type.value}
// 自動生成 by MetalShaderGenerator

#include <metal_stdlib>
using namespace metal;

kernel void {shader_type.value}_placeholder(
    device float* data [[buffer(0)]],
    constant int& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid < size) {{
        data[gid] = 0.0f;
    }}
}}
'''

    def _emit_coverage_report(self, shaders: List[ShaderType]) -> None:
        """輸出覆蓋率報告"""
        ds4_total = 17
        generated = len(shaders)

        logger.info("=" * 60)
        logger.info("[MetalShaderGenerator] === Shader 覆蓋率報告 ===")
        logger.info(f"[MetalShaderGenerator] ds4.c 總計: {ds4_total} 個 Shader")
        logger.info(f"[MetalShaderGenerator] 已生成: {generated} 個 Shader")
        logger.info(f"[MetalShaderGenerator] 覆蓋率: {generated/ds4_total*100:.1f}%")

        logger.info("[MetalShaderGenerator] 生成的 Shader 清單:")
        for i, s in enumerate(shaders, 1):
            cfg = self.SHADER_DEFINITIONS.get(s)
            logger.info(f"  {i}. {s.value} (priority={cfg.priority if cfg else 'N/A'})")

        missing = set(ShaderType) - set(shaders)
        if missing:
            logger.info("[MetalShaderGenerator] 缺失的 Shader:")
            for s in missing:
                logger.info(f"  - {s.value}")

    def save_shaders(self, output_dir: Optional[str] = None) -> List[str]:
        """保存所有生成的 Shader 到文件"""
        import os

        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        for shader_type, code in self.generated_shaders.items():
            filename = f"{shader_type.value}.metal"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(code)
            saved_files.append(filepath)
            logger.info(f"[MetalShaderGenerator] 保存: {filepath}")

        return saved_files
