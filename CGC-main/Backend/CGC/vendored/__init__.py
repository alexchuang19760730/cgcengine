"""vendored/__init__.py — 上游开源推理加速框架 vendored 整合

本目录 vendored 整合以下 MIT 协议上游仓库（保留各自 LICENSE/NOTICE）：
  - jetspec/ : https://github.com/hao-ai-lab/JetSpec (MIT)
               并行树草稿投机解码，head-based SD framework。
  - deepspec/ : https://github.com/deepseek-ai/DeepSpec (MIT)
                DSpark 半自回归草稿模型 + 置信度调度验证。

本 vendored 整合对应 Gate 2.0 能力：
  - g21_jetspec_draft_runtime_adapter
  - g21_dspark_scheduler_runtime_adapter

Adapter 模块把上游 API 包装为 CGC runtime 可直接调用的接口。
"""

from .jetspec_adapter import (
    JetSpecRuntimeAdapter,
    JetSpecAdapterError,
    get_jetspec_adapter,
)
from .dspark_adapter import (
    DSparkRuntimeAdapter,
    DSparkAdapterError,
    get_dspark_adapter,
)
from .sglang_spec_plugin import (
    DraftProposal,
    JetSpecDraftWorker,
    DSparkDraftWorker,
    register_cgc_spec_algos,
    run_draft_verify_round,
    build_target_verify_input,
    accept_tokens_after_verify,
)

__all__ = [
    "JetSpecRuntimeAdapter",
    "JetSpecAdapterError",
    "get_jetspec_adapter",
    "DSparkRuntimeAdapter",
    "DSparkAdapterError",
    "get_dspark_adapter",
    # SGLang verify stack plugin
    "DraftProposal",
    "JetSpecDraftWorker",
    "DSparkDraftWorker",
    "register_cgc_spec_algos",
    "run_draft_verify_round",
    "build_target_verify_input",
    "accept_tokens_after_verify",
]
