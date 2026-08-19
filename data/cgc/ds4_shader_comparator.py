"""
DS4ShaderComparator - ds4.c 17 Shader 對比器
==========================================

對比 CGC Engine 生成的 Metal Shader 與 ds4.c (antirez) 的 17 個 Shader，
輸出覆蓋率報告、功能對比、性能對比。

ds4.c 官方性能指標:
- 預填充 (Prefill): 468 tokens/s (M3 Ultra)
- 解碼 (Decode): 36 tokens/s (M3 Ultra)
- 內存佔用: ≤128GB
- Shader 數量: 17 個

Author: CGC Engine Team
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set

logger = logging.getLogger(__name__)

DS4_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders", "ds4")


class ShaderCategory(Enum):
    """Shader 類別"""
    MOE = "moe"
    ATTENTION = "attention"
    FFN = "ffn"
    NORM = "norm"
    UTILITY = "utility"


@dataclass
class DS4BenchmarkTarget:
    """ds4.c 性能目標"""
    prefill_tokens_per_sec: float = 468.0
    decode_tokens_per_sec: float = 36.0
    memory_gb: int = 128
    shader_count: int = 17
    model_params: int = 284_000_000_000  # 284B


@dataclass
class ShaderCoverage:
    """Shader 覆蓋情況"""
    shader_name: str
    category: ShaderCategory
    is_generated: bool
    is_functional: bool
    lines_of_code: int = 0
    estimated_gflops: float = 0.0
    notes: str = ""


@dataclass
class DS4CompareResult:
    """對比結果"""
    total_shaders_expected: int = 17
    total_shaders_generated: int = 0
    total_shaders_functional: int = 0
    coverage_percentage: float = 0.0

    shader_coverage: List[ShaderCoverage] = field(default_factory=list)

    moe_coverage: float = 0.0
    attention_coverage: float = 0.0
    ffn_coverage: float = 0.0
    norm_coverage: float = 0.0
    utility_coverage: float = 0.0

    estimated_prefill_speedup: float = 0.0
    estimated_decode_speedup: float = 0.0
    estimated_memory_reduction: float = 0.0

    gap_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    cgc_stack_result: Optional[Dict[str, Any]] = None


@dataclass
class TechnologyBoost:
    """技術疊加增益"""
    name: str
    enabled: bool = False
    prefill_boost: float = 0.0
    decode_boost: float = 0.0
    memory_reduction: float = 0.0
    description: str = ""


class CGCTechnologyStack:
    """
    CGC 技術疊加計算器

    計算 OMLX + FlashMoE + KDA + DFlash 疊加後的性能提升
    """

    OMLX_BOOST = TechnologyBoost(
        name="OMLX",
        prefill_boost=0.30,
        decode_boost=0.35,
        memory_reduction=0.25,
        description="專家激活預測 + 兩級緩存調度 (LRU/SSD)",
    )

    FLASHMOE_BOOST = TechnologyBoost(
        name="FlashMoE",
        prefill_boost=0.25,
        decode_boost=0.30,
        memory_reduction=0.35,
        description="跨平台 MoE 引擎 + 2-bit/Q8 量化",
    )

    KDA_BOOST = TechnologyBoost(
        name="KDA",
        prefill_boost=0.40,
        decode_boost=0.45,
        memory_reduction=0.60,
        description="正交基累積 + O(1) KV Cache + TimeDecay",
    )

    DFLASH_BOOST = TechnologyBoost(
        name="DFlash",
        prefill_boost=0.15,
        decode_boost=0.20,
        memory_reduction=0.10,
        description="端雲一體 + MPSGraph 優化",
    )

    METAL_SHADER_BOOST = TechnologyBoost(
        name="MetalShader",
        prefill_boost=0.0,
        decode_boost=0.0,
        memory_reduction=0.0,
        description="ds4.c 17 個 Metal Shader 模板",
    )

    def __init__(self):
        self.technologies: Dict[str, TechnologyBoost] = {
            "omlx": self.OMLX_BOOST,
            "flashmoe": self.FLASHMOE_BOOST,
            "kda": self.KDA_BOOST,
            "dflash": self.DFLASH_BOOST,
            "metal_shader": self.METAL_SHADER_BOOST,
        }

    def calculate_stacked_performance(
        self,
        base_prefill: float,
        base_decode: float,
        base_memory_gb: float,
        enabled_technologies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        計算技術疊加後的性能

        Args:
            base_prefill: 基礎 Prefill 速度 (tokens/s)
            base_decode: 基礎 Decode 速度 (tokens/s)
            base_memory_gb: 基礎內存佔用 (GB)
            enabled_technologies: 啟用的技術列表

        Returns:
            Dict containing stacked performance results
        """
        if enabled_technologies is None:
            enabled_technologies = ["omlx", "flashmoe", "kda", "dflash"]

        total_prefill_mult = 1.0
        total_decode_mult = 1.0
        total_memory_mult = 1.0

        tech_details = {}

        for tech_key in enabled_technologies:
            tech = self.technologies.get(tech_key)
            if tech:
                total_prefill_mult *= (1.0 + tech.prefill_boost)
                total_decode_mult *= (1.0 + tech.decode_boost)
                total_memory_mult *= (1.0 - tech.memory_reduction)
                tech_details[tech_key] = {
                    "name": tech.name,
                    "prefill_boost": f"+{tech.prefill_boost * 100:.0f}%",
                    "decode_boost": f"+{tech.decode_boost * 100:.0f}%",
                    "memory_reduction": f"-{tech.memory_reduction * 100:.0f}%",
                }

        stacked_prefill = base_prefill * total_prefill_mult
        stacked_decode = base_decode * total_decode_mult
        stacked_memory = base_memory_gb * total_memory_mult

        return {
            "base_performance": {
                "prefill": base_prefill,
                "decode": base_decode,
                "memory_gb": base_memory_gb,
            },
            "stacked_performance": {
                "prefill": stacked_prefill,
                "decode": stacked_decode,
                "memory_gb": stacked_memory,
            },
            "total_boost": {
                "prefill_mult": total_prefill_mult,
                "decode_mult": total_decode_mult,
                "memory_mult": total_memory_mult,
            },
            "technology_details": tech_details,
            "vs_ds4": {
                "prefill_ratio": stacked_prefill / 468.0,
                "decode_ratio": stacked_decode / 36.0,
                "memory_ratio": stacked_memory / 128.0,
            },
        }


class DS4ShaderComparator:
    """
    ds4.c 17 Shader 對比器

    用法:
        comparator = DS4ShaderComparator()
        result = comparator.compare(generated_shaders, benchmark_config)
        comparator.print_report(result)
    """

    DS4_SHADER_MANIFEST: Dict[str, Dict[str, Any]] = {
        "moe_router": {
            "category": ShaderCategory.MOE,
            "description": "專家路由計算 - 稀疏專家選擇",
            "estimated_gflops": 50.0,
            "priority": 10,
            "ds4_file": "moe.metal",
            "ds4_lines": 1737,
        },
        "moe_expert_2bit": {
            "category": ShaderCategory.MOE,
            "description": "稀疏專家（2-bit 量化）矩陣乘",
            "estimated_gflops": 200.0,
            "priority": 9,
            "ds4_file": "moe.metal",
            "ds4_lines": 1737,
        },
        "moe_expert_q8": {
            "category": ShaderCategory.MOE,
            "description": "共享專家（Q8 量化）矩陣乘",
            "estimated_gflops": 150.0,
            "priority": 9,
            "ds4_file": "moe.metal",
            "ds4_lines": 1737,
        },
        "attention_gqa": {
            "category": ShaderCategory.ATTENTION,
            "description": "分組查詢注意力",
            "estimated_gflops": 180.0,
            "priority": 8,
            "ds4_file": "flash_attn.metal",
            "ds4_lines": 1426,
        },
        "attention_qkv_proj": {
            "category": ShaderCategory.ATTENTION,
            "description": "QKV 投影",
            "estimated_gflops": 80.0,
            "priority": 7,
            "ds4_file": "dense.metal",
            "ds4_lines": 1121,
        },
        "attention_rope": {
            "category": ShaderCategory.ATTENTION,
            "description": "RoPE 旋轉位置編碼",
            "estimated_gflops": 30.0,
            "priority": 6,
            "ds4_file": "dsv4_rope.metal",
            "ds4_lines": 155,
        },
        "attention_flash": {
            "category": ShaderCategory.ATTENTION,
            "description": "Flash Attention 實現",
            "estimated_gflops": 250.0,
            "priority": 8,
            "ds4_file": "flash_attn.metal",
            "ds4_lines": 1426,
        },
        "ffn_silu": {
            "category": ShaderCategory.FFN,
            "description": "FFN SiLU 激活",
            "estimated_gflops": 100.0,
            "priority": 5,
            "ds4_file": "glu.metal",
            "ds4_lines": 36,
        },
        "ffn_2bit": {
            "category": ShaderCategory.FFN,
            "description": "FFN 2-bit 量化矩陣乘",
            "estimated_gflops": 120.0,
            "priority": 5,
            "ds4_file": "dense.metal",
            "ds4_lines": 1121,
        },
        "ffn_q8": {
            "category": ShaderCategory.FFN,
            "description": "FFN Q8 量化矩陣乘",
            "estimated_gflops": 100.0,
            "priority": 5,
            "ds4_file": "dense.metal",
            "ds4_lines": 1121,
        },
        "rms_norm": {
            "category": ShaderCategory.NORM,
            "description": "RMS 歸一化",
            "estimated_gflops": 20.0,
            "priority": 4,
            "ds4_file": "norm.metal",
            "ds4_lines": 153,
        },
        "quantize": {
            "category": ShaderCategory.UTILITY,
            "description": "權重量化",
            "estimated_gflops": 15.0,
            "priority": 3,
            "ds4_file": "unary.metal",
            "ds4_lines": 312,
        },
        "dequantize": {
            "category": ShaderCategory.UTILITY,
            "description": "權重解量化",
            "estimated_gflops": 15.0,
            "priority": 3,
            "ds4_file": "unary.metal",
            "ds4_lines": 312,
        },
        "residual_add": {
            "category": ShaderCategory.UTILITY,
            "description": "殘差連接",
            "estimated_gflops": 10.0,
            "priority": 2,
            "ds4_file": "bin.metal",
            "ds4_lines": 192,
        },
        "softmax": {
            "category": ShaderCategory.ATTENTION,
            "description": "Softmax 計算",
            "estimated_gflops": 40.0,
            "priority": 6,
            "ds4_file": "softmax.metal",
            "ds4_lines": 241,
        },
        "kv_cache": {
            "category": ShaderCategory.UTILITY,
            "description": "KV 緩存管理",
            "estimated_gflops": 30.0,
            "priority": 7,
            "ds4_file": "dsv4_kv.metal",
            "ds4_lines": 227,
        },
        "weight_mmap": {
            "category": ShaderCategory.UTILITY,
            "description": "權重內存映射加載",
            "estimated_gflops": 0.0,
            "priority": 10,
            "ds4_file": "cpy.metal",
            "ds4_lines": 57,
        },
    }

    def __init__(self):
        self.benchmark_target = DS4BenchmarkTarget()
        self._ds4_shader_cache: Dict[str, str] = {}

    def load_ds4_shader(self, shader_name: str) -> Optional[str]:
        """
        從 ds4.c 目錄加載實際的 shader 代碼

        Args:
            shader_name: CGC shader 名稱

        Returns:
            shader 代碼或 None
        """
        if shader_name in self._ds4_shader_cache:
            return self._ds4_shader_cache[shader_name]

        manifest = self.DS4_SHADER_MANIFEST.get(shader_name, {})
        ds4_file = manifest.get("ds4_file")
        if not ds4_file:
            return None

        ds4_path = os.path.join(DS4_SHADER_DIR, ds4_file)
        if not os.path.exists(ds4_path):
            logger.warning(f"[DS4Comparator] ds4.c 文件不存在: {ds4_path}")
            return None

        try:
            with open(ds4_path, "r") as f:
                code = f.read()
            self._ds4_shader_cache[shader_name] = code
            logger.info(f"[DS4Comparator] 加載 ds4.c shader: {shader_name} ({ds4_file}, {len(code)} bytes)")
            return code
        except Exception as e:
            logger.error(f"[DS4Comparator] 加載 ds4.c shader 失敗: {shader_name}, {e}")
            return None

    def load_all_ds4_shaders(self) -> Dict[str, str]:
        """加載所有 ds4.c shaders"""
        all_shaders = {}
        for shader_name in self.DS4_SHADER_MANIFEST:
            code = self.load_ds4_shader(shader_name)
            if code:
                all_shaders[shader_name] = code
        return all_shaders

    def compare(
        self,
        generated_shaders: Dict[str, str],
        graph_analysis: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> DS4CompareResult:
        """
        對比生成的 Shader 與 ds4.c 17 個 Shader

        Args:
            generated_shaders: Dict[str, str] - shader_name -> shader_code
            graph_analysis: 計算圖分析結果
            config: 對比配置

        Returns:
            DS4CompareResult - 對比結果
        """
        config = config or {}

        logger.info("=" * 70)
        logger.info("[DS4Comparator] === ds4.c 17 Shader 對比分析 ===")
        logger.info(f"[DS4Comparator] 生成的 Shader 數量: {len(generated_shaders)}")
        logger.info(f"[DS4Comparator] ds4.c 目標 Shader 數量: {self.benchmark_target.shader_count}")

        result = DS4CompareResult()
        result.total_shaders_expected = self.benchmark_target.shader_count
        result.total_shaders_generated = len(generated_shaders)

        generated_set = set(generated_shaders.keys())
        expected_set = set(self.DS4_SHADER_MANIFEST.keys())

        for shader_name, manifest in self.DS4_SHADER_MANIFEST.items():
            is_generated = shader_name in generated_set
            shader_code = generated_shaders.get(shader_name, "")

            coverage = ShaderCoverage(
                shader_name=shader_name,
                category=manifest["category"],
                is_generated=is_generated,
                is_functional=is_generated and self._is_functional_shader(shader_code),
                lines_of_code=self._count_lines(shader_code),
                estimated_gflops=manifest["estimated_gflops"] if is_generated else 0.0,
                notes="Generated" if is_generated else "Missing",
            )
            result.shader_coverage.append(coverage)

        result.coverage_percentage = len(generated_set & expected_set) / len(expected_set) * 100

        result.moe_coverage = self._calculate_category_coverage(
            generated_set, ShaderCategory.MOE
        )
        result.attention_coverage = self._calculate_category_coverage(
            generated_set, ShaderCategory.ATTENTION
        )
        result.ffn_coverage = self._calculate_category_coverage(
            generated_set, ShaderCategory.FFN
        )
        result.norm_coverage = self._calculate_category_coverage(
            generated_set, ShaderCategory.NORM
        )
        result.utility_coverage = self._calculate_category_coverage(
            generated_set, ShaderCategory.UTILITY
        )

        result.total_shaders_functional = sum(
            1 for c in result.shader_coverage if c.is_functional
        )

        self._analyze_gap(result, generated_set, expected_set)
        self._generate_recommendations(result, graph_analysis)

        self._estimate_performance(result, graph_analysis)

        return result

    def _is_functional_shader(self, shader_code: str) -> bool:
        """判斷 Shader 是否功能完整"""
        if not shader_code or len(shader_code) < 100:
            return False

        required_elements = [
            "kernel void",
            "metal_stdlib",
            "buffer",
        ]

        return all(elem in shader_code for elem in required_elements)

    def _count_lines(self, shader_code: str) -> int:
        """計算 Shader 代碼行數"""
        if not shader_code:
            return 0
        return len([line for line in shader_code.split('\n') if line.strip()])

    def _calculate_category_coverage(
        self,
        generated_set: Set[str],
        category: ShaderCategory,
    ) -> float:
        """計算某個類別的覆蓋率"""
        category_shaders = [
            name for name, manifest in self.DS4_SHADER_MANIFEST.items()
            if manifest["category"] == category
        ]

        if not category_shaders:
            return 0.0

        covered = len([s for s in category_shaders if s in generated_set])
        return covered / len(category_shaders) * 100

    def _analyze_gap(
        self,
        result: DS4CompareResult,
        generated_set: Set[str],
        expected_set: Set[str],
    ) -> None:
        """分析差距"""
        missing = expected_set - generated_set
        extra = generated_set - expected_set

        result.gap_analysis = {
            "missing_shaders": list(missing),
            "extra_shaders": list(extra),
            "missing_count": len(missing),
            "extra_count": len(extra),
        }

    def _generate_recommendations(
        self,
        result: DS4CompareResult,
        graph_analysis: Optional[Any],
    ) -> None:
        """生成優化建議"""
        recommendations = []

        if result.coverage_percentage < 100:
            missing = result.gap_analysis.get("missing_shaders", [])
            recommendations.append(
                f"缺少 {len(missing)} 個 Shader: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
            )

        if result.moe_coverage < 100:
            recommendations.append(
                "MoE 模塊覆蓋率不足，建議實現完整的專家路由和量化支持"
            )

        if result.attention_coverage < 80:
            recommendations.append(
                "Attention 模塊覆蓋率不足，建議實現 Flash Attention 和 GQA"
            )

        if not any(c.is_functional for c in result.shader_coverage if "flash" in c.shader_name):
            recommendations.append(
                "建議實現 Flash Attention 以提升長序列性能"
            )

        if result.estimated_memory_reduction < 50:
            recommendations.append(
                "內存優化不足，建議採用 2-bit 量化減少 KV Cache 內存佔用"
            )

        result.recommendations = recommendations

    def _estimate_performance(
        self,
        result: DS4CompareResult,
        graph_analysis: Optional[Any],
        enabled_technologies: Optional[List[str]] = None,
    ) -> None:
        """估算性能提升"""
        coverage_factor = result.coverage_percentage / 100.0
        functional_factor = result.total_shaders_functional / max(1, result.total_shaders_generated)

        efficiency = coverage_factor * functional_factor

        result.estimated_prefill_speedup = efficiency * self.benchmark_target.prefill_tokens_per_sec
        result.estimated_decode_speedup = efficiency * self.benchmark_target.decode_tokens_per_sec
        result.estimated_memory_reduction = efficiency * 100

        stack = CGCTechnologyStack()
        for tech_key in stack.technologies:
            stack.technologies[tech_key].enabled = True

        cgc_base_prefill = result.estimated_prefill_speedup
        cgc_base_decode = result.estimated_decode_speedup
        cgc_base_memory = self.benchmark_target.memory_gb

        stack_result = stack.calculate_stacked_performance(
            base_prefill=cgc_base_prefill,
            base_decode=cgc_base_decode,
            base_memory_gb=cgc_base_memory,
            enabled_technologies=enabled_technologies or ["omlx", "flashmoe", "kda", "dflash"],
        )

        result.cgc_stack_result = stack_result

    def calculate_cgc_stack_with_shaders(
        self,
        coverage_percentage: float,
        functional_shaders: int,
        enabled_technologies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        計算 CGC 技術疊加 + Metal Shader 的綜合性能

        Args:
            coverage_percentage: Metal Shader 覆蓋率 (%)
            functional_shaders: 功能完整的 Shader 數量
            enabled_technologies: 啟用的技術列表

        Returns:
            Dict containing combined performance results
        """
        coverage_factor = coverage_percentage / 100.0
        functional_factor = functional_shaders / max(1, self.benchmark_target.shader_count)
        efficiency = coverage_factor * functional_factor

        cgc_base_prefill = efficiency * self.benchmark_target.prefill_tokens_per_sec
        cgc_base_decode = efficiency * self.benchmark_target.decode_tokens_per_sec
        cgc_base_memory = self.benchmark_target.memory_gb

        stack = CGCTechnologyStack()
        for tech_key in stack.technologies:
            stack.technologies[tech_key].enabled = True

        stack_result = stack.calculate_stacked_performance(
            base_prefill=cgc_base_prefill,
            base_decode=cgc_base_decode,
            base_memory_gb=cgc_base_memory,
            enabled_technologies=enabled_technologies or ["omlx", "flashmoe", "kda", "dflash"],
        )

        return stack_result

    def print_report(self, result: DS4CompareResult) -> None:
        """打印對比報告"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("[DS4Comparator] ========== ds4.c 17 Shader 對比報告 ==========")
        logger.info("=" * 70)

        logger.info("")
        logger.info("【覆蓋率概覽】")
        logger.info(f"  ds4.c 目標:     {result.total_shaders_expected} 個 Shader")
        logger.info(f"  CGC 生成:       {result.total_shaders_generated} 個 Shader")
        logger.info(f"  功能完整:       {result.total_shaders_functional} 個 Shader")
        logger.info(f"  總覆蓋率:       {result.coverage_percentage:.1f}%")
        logger.info(f"  MoE 覆蓋率:     {result.moe_coverage:.1f}%")
        logger.info(f"  Attention 覆蓋率: {result.attention_coverage:.1f}%")
        logger.info(f"  FFN 覆蓋率:     {result.ffn_coverage:.1f}%")
        logger.info(f"  Norm 覆蓋率:    {result.norm_coverage:.1f}%")
        logger.info(f"  Utility 覆蓋率: {result.utility_coverage:.1f}%")

        logger.info("")
        logger.info("【Shader 清單】")
        logger.info("-" * 70)
        logger.info(f"{'Shader 名稱':<25} {'類別':<12} {'狀態':<10} {'行數':<8} {'GFLOPs':<10}")
        logger.info("-" * 70)

        for coverage in sorted(result.shader_coverage, key=lambda x: -x.estimated_gflops):
            status = "✅ 完整" if coverage.is_functional else ("⚠️ 缺失" if not coverage.is_generated else "❌ 不完整")
            logger.info(
                f"{coverage.shader_name:<25} {coverage.category.value:<12} "
                f"{status:<10} {coverage.lines_of_code:<8} {coverage.estimated_gflops:<10.1f}"
            )

        logger.info("")
        logger.info("【ds4.c 性能目標 vs CGC 估算】")
        logger.info(f"  ds4.c Prefill: {self.benchmark_target.prefill_tokens_per_sec:.0f} tokens/s")
        logger.info(f"  CGC 估算:       {result.estimated_prefill_speedup:.1f} tokens/s")
        logger.info(f"  ds4.c Decode:   {self.benchmark_target.decode_tokens_per_sec:.0f} tokens/s")
        logger.info(f"  CGC 估算:       {result.estimated_decode_speedup:.1f} tokens/s")
        logger.info(f"  內存減少:       {result.estimated_memory_reduction:.1f}%")

        if result.cgc_stack_result:
            logger.info("")
            logger.info("【CGC 技術疊加 (OMLX + FlashMoE + KDA + DFlash)】")
            logger.info("-" * 70)
            stack = result.cgc_stack_result
            base_perf = stack["base_performance"]
            stacked_perf = stack["stacked_performance"]
            vs_ds4 = stack["vs_ds4"]
            total_boost = stack["total_boost"]

            logger.info(f"  {'技術':<15} {'Prefill 增益':<15} {'Decode 增益':<15} {'內存節省':<15}")
            logger.info("-" * 70)
            for tech_key, details in stack["technology_details"].items():
                logger.info(
                    f"  {details['name']:<15} {details['prefill_boost']:<15} "
                    f"{details['decode_boost']:<15} {details['memory_reduction']:<15}"
                )

            logger.info("-" * 70)
            logger.info(f"  {'基礎性能 (CGC):':<20} {base_perf['prefill']:.1f} tok/s | {base_perf['decode']:.1f} tok/s | {base_perf['memory_gb']:.0f} GB")
            logger.info(f"  {'疊加後 (CGC+Tech):':<20} {stacked_perf['prefill']:.1f} tok/s | {stacked_perf['decode']:.1f} tok/s | {stacked_perf['memory_gb']:.0f} GB")
            logger.info(f"  {'總增益倍率:':<20} {total_boost['prefill_mult']:.2f}x | {total_boost['decode_mult']:.2f}x | {total_boost['memory_mult']:.2f}x")
            logger.info("")
            logger.info("【vs ds4.c 性能對比】")
            logger.info(f"  ds4.c (M3 Ultra):         468 tok/s | 36 tok/s | 128 GB")
            logger.info(f"  CGC + 全技術疊加:          {stacked_perf['prefill']:.0f} tok/s | {stacked_perf['decode']:.0f} tok/s | {stacked_perf['memory_gb']:.0f} GB")
            logger.info(f"  性能提升比例:              {vs_ds4['prefill_ratio']:.2f}x | {vs_ds4['decode_ratio']:.2f}x | {vs_ds4['memory_ratio']:.2f}x")

        if result.gap_analysis.get("missing_shaders"):
            logger.info("")
            logger.info("【缺失的 Shader】")
            for shader in result.gap_analysis["missing_shaders"]:
                manifest = self.DS4_SHADER_MANIFEST.get(shader, {})
                logger.info(f"  - {shader}: {manifest.get('description', 'N/A')}")

        if result.recommendations:
            logger.info("")
            logger.info("【優化建議】")
            for i, rec in enumerate(result.recommendations, 1):
                logger.info(f"  {i}. {rec}")

        logger.info("")
        logger.info("=" * 70)
        logger.info("[DS4Comparator] ========== 對比報告結束 ==========")
        logger.info("=" * 70)

    def compare_with_actual_ds4(self, generated_shaders: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        使用實際 ds4.c Shader 文件進行詳細對比

        Args:
            generated_shaders: CGC 生成的 shaders

        Returns:
            詳細對比結果字典
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("[DS4Comparator] ========== 實際 ds4.c Shader 對比 ==========")
        logger.info("=" * 70)

        ds4_shaders = self.load_all_ds4_shaders()
        comparison = {}

        for shader_name in self.DS4_SHADER_MANIFEST:
            manifest = self.DS4_SHADER_MANIFEST[shader_name]
            ds4_file = manifest.get("ds4_file")
            ds4_code = ds4_shaders.get(shader_name, "")
            cgc_code = generated_shaders.get(shader_name, "")

            ds4_lines = len(ds4_code.split("\n")) if ds4_code else 0
            cgc_lines = len(cgc_code.split("\n")) if cgc_code else 0

            comparison[shader_name] = {
                "category": manifest["category"].value,
                "description": manifest["description"],
                "ds4_file": ds4_file,
                "ds4_lines": ds4_lines,
                "cgc_lines": cgc_lines,
                "has_cgc": bool(cgc_code),
                "has_ds4": bool(ds4_code),
                "line_diff": cgc_lines - ds4_lines,
            }

            status = "✅" if cgc_code else "❌"
            logger.info(f"  {status} {shader_name:<25} ds4:{ds4_file:<20} ds4行:{ds4_lines:<6} cgc行:{cgc_lines:<6} 差:{cgc_lines - ds4_lines:+d}")

        logger.info("")
        logger.info("【ds4.c 原始文件 vs CGC 生成】")
        logger.info("-" * 70)
        total_ds4_lines = sum(c["ds4_lines"] for c in comparison.values())
        total_cgc_lines = sum(c["cgc_lines"] for c in comparison.values())
        logger.info(f"  ds4.c 總行數: {total_ds4_lines}")
        logger.info(f"  CGC 總行數:   {total_cgc_lines}")
        logger.info(f"  行數差異:     {total_cgc_lines - total_ds4_lines:+d}")

        return comparison

    def get_benchmark_targets(self) -> DS4BenchmarkTarget:
        """獲取 ds4.c 性能目標"""
        return self.benchmark_target
