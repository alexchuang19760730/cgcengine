"""gate_verifiers — Gate 1.0/2.0 真实验证器

替换 cgc_engine/cli.py 中 model_verify_command 的 stub checks.extend 逻辑，
提供端到端真实验证能力。

Gate 1.0:
  - DOPDVerifier        -> cgc_engine.pd.dopd_runtime + dopd_schema
  - CQ4Verifier         -> Backend.CGC.edge_moe_transport.cq4_session
  - ZeroCopyVerifier    -> torch.cuda + 模拟直接内存映射
  - TrueOrthoKDAVerifier -> KV 压缩 + 可移植状态（在 cli.py 内联实现）

Gate 2.0 本体:
  - LayerAdaptiveVerifier -> max_local_layer / finished_layer 配置校验
  - SGLangTP4EP4Verifier  -> SGLang TP4EP4 prefill 主干 launch kwargs
  - DeepEPModeVerifier    -> Backend.CGC.deepep_sglang_patch（已弃用，保留兼容）
  - EPLBVerifier          -> Backend.CGC.cloud_sglang...eplb (EPLBManager 算法层)
  - WaterfillVerifier     -> Backend.CGC.cloud_sglang...deepep_waterfill (Triton kernel)
  - LPLBVerifier          -> cgc_engine.lplb_solver (新增实现)

Gate 2.1 投机解码:
  - JetSpecVerifier       -> Backend.CGC.vendored.jetspec_adapter
  - DSparkVerifier        -> Backend.CGC.vendored.dspark_adapter

Gate 2.2 KV Cache:
  - KVCacheManagementVerifier / KVCacheReuseVerifier /
    KVDynamicSizingVerifier / KVCachePrefetchingVerifier

Gate 2.3 RSWA + Prefill Pool + 传输:
  - UnifiedIRInjectVerifier      -> Backend.CGC.compiler.unified_compiler
  - EndToEndMoETransportVerifier -> Backend.CGC.edge_moe_transport
  - NFSoRDMAVerifier             -> Backend.CGC.edge_moe_transport.nfsordma_transport
"""
from .base import VerificationResult, VerificationStatus
from .dopd_verifier import DOPDVerifier
from .cq4_verifier import CQ4Verifier
from .zero_copy_verifier import ZeroCopyVerifier
from .trueorthokda_verifier import TrueOrthoKDAVerifier
from .deepseek_v4_flash_resume_verifier import DeepSeekV4FlashResumeVerifier
from .edge_omlx_flashmoe_verifier import EdgeOMLXFlashMoEVerifier
from .layer_adaptive_verifier import LayerAdaptiveVerifier
from .deepep_mode_verifier import DeepEPModeVerifier
from .eplb_verifier import EPLBVerifier
from .waterfill_verifier import WaterfillVerifier
from .lplb_verifier import LPLBVerifier
from .sglang_tp4ep4_verifier import SGLangTP4EP4Verifier
from .ray_engine_dual_host_verifier import RayEngineDualHostVerifier
from .colossalai_runtime_candidate_verifier import ColossalAIRuntimeCandidateVerifier
from .g21_fusion_governance_verifier import (
    G21EightStepPipelineGovernanceVerifier,
    G21StateABIExtensionHookVerifier,
    G21UPKFusionBindingVerifier,
)
from .g22_deepep_l20n_verifier import (
    G22BootstrapDeepEPCompatVerifier,
    G22DeepEPL20NDualNodeVerifier,
    G22DeepEPL20NInferenceVerifier,
    G22DeepEPL20NMegatrainVerifier,
    G22StateABIL20NVerifier,
    G22SystemProfileL20NVerifier,
    G22UPKL20NOptimizationVerifier,
)
from .rswa_double_layer_kv_verifier import RSWADoubleLayerKVVerifier
from .g23_trueorthokda_adapter_verifier import G23TrueOrthoKDAAdapterVerifier
from .g23_cloud_l20n_tp4_verifier import G23CloudL20NTP4Verifier
from .unified_ir_inject_verifier import UnifiedIRInjectVerifier
from .endtoend_moe_transport_verifier import EndToEndMoETransportVerifier
from .nfsordma_verifier import NFSoRDMAVerifier
from .jetspec_verifier import JetSpecVerifier
from .dspark_verifier import DSparkVerifier
from .dflash_deepseek_v4_verifier import DFlashDeepSeekV4Verifier
from .kv_cache_verifier import (
    KVCacheManagementVerifier,
    KVCacheReuseVerifier,
    KVDynamicSizingVerifier,
    KVCachePrefetchingVerifier,
)

__all__ = [
    "VerificationResult",
    "VerificationStatus",
    # Gate 1.0
    "DOPDVerifier",
    "CQ4Verifier",
    "ZeroCopyVerifier",
    "TrueOrthoKDAVerifier",
    "DeepSeekV4FlashResumeVerifier",
    "EdgeOMLXFlashMoEVerifier",
    # Gate 2.0 本体
    "LayerAdaptiveVerifier",
    "DeepEPModeVerifier",
    "EPLBVerifier",
    "WaterfillVerifier",
    "LPLBVerifier",
    "SGLangTP4EP4Verifier",
    "RayEngineDualHostVerifier",
    "ColossalAIRuntimeCandidateVerifier",
    "G21EightStepPipelineGovernanceVerifier",
    "G21UPKFusionBindingVerifier",
    "G21StateABIExtensionHookVerifier",
    "G22DeepEPL20NDualNodeVerifier",
    "G22DeepEPL20NMegatrainVerifier",
    "G22DeepEPL20NInferenceVerifier",
    "G22BootstrapDeepEPCompatVerifier",
    "G22SystemProfileL20NVerifier",
    "G22UPKL20NOptimizationVerifier",
    "G22StateABIL20NVerifier",
    "RSWADoubleLayerKVVerifier",
    "G23TrueOrthoKDAAdapterVerifier",
    "G23CloudL20NTP4Verifier",
    # Gate 2.1 投机解码
    "JetSpecVerifier",
    "DSparkVerifier",
    "DFlashDeepSeekV4Verifier",
    # Gate 2.2 KV Cache
    "KVCacheManagementVerifier",
    "KVCacheReuseVerifier",
    "KVDynamicSizingVerifier",
    "KVCachePrefetchingVerifier",
    # Gate 2.3 传输
    "UnifiedIRInjectVerifier",
    "EndToEndMoETransportVerifier",
    "NFSoRDMAVerifier",
]
