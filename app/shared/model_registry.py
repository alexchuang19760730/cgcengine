"""统一模型注册表 — 跨后端/跨模型/跨平台的单一配置源.

所有模型 (Qwen3-VL / Gemma4 / DSV4 Flash) 的架构参数、EOS tokens、
首 token 校准规则、模型路径模式都从此文件定义。

使用方式:
    from app.shared.model_registry import get_model_config, ModelConfig

    cfg = get_model_config("gemma4")
    print(cfg.hidden_size, cfg.vocab_size, cfg.eos_tokens)

    # 训练脚本
    mtp = cfg.create_mtp_head()
    # edge_first_proxy
    calibration = cfg.first_token_rules

设计原则:
    - 一处定义, 处处引用 (训练/推理/proxy/CLI)
    - 跨后端: 同一 config 生成 PyTorch / MLX / SGLang 版 MTP head
    - 跨模型: 新增模型只需在此文件添加一个 ModelConfig entry
    - 跨平台: Mac (MLX) / gs01 (PyTorch) / cloud (SGLang) 共享同一 registry
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_mtp_output_base_dir() -> str:
    env_root = str(os.environ.get("CGC_MTP_OUTPUT_BASE_DIR", "") or "").strip()
    if env_root:
        return env_root.rstrip("/")
    if os.path.isdir("/data"):
        return "/data/mtp_output"
    return os.path.join(REPO_ROOT, "var", "mtp_output")


def _default_mtp_train_data_dir(model_name: str) -> str:
    env_root = str(os.environ.get("CGC_MTP_TRAIN_DATA_BASE_DIR", "") or "").strip()
    if env_root:
        return os.path.join(env_root.rstrip("/"), model_name)
    if os.path.isdir("/data"):
        return f"/data/mtp_{model_name}_shards"
    return os.path.join(REPO_ROOT, "var", "mtp_train_data", model_name)


@dataclass
class FirstTokenRule:
    """首 token 校准规则 (per-model)."""
    family: str
    markers: tuple[str, ...]
    confidence: float
    candidates: list[str]
    enabled: bool = True


@dataclass
class ModelConfig:
    """单个模型的完整配置 — MTP head + 训练 + 推理 + 校准."""

    # === 标识 ===
    name: str                          # 注册名: "gemma4", "dsv4", "qwen3vl"
    display_name: str                  # 显示名: "Gemma4-26B-A4B"
    model_type: str                    # AutoConfig.model_type: "gemma4", "deepseek_v4", "qwen3_vl"
    architectures: list[str]           # config.architectures

    # === 架构参数 (MTP head 用) ===
    hidden_size: int
    vocab_size: int
    num_heads: int
    head_dim: int
    intermediate_size: int             # MTP head MLP intermediate (可能 != base model)
    num_hidden_layers: int = 0         # decoder 层数 (用于 residency / runtime 估算)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960

    # === MTP head 专用参数 (可覆盖 base model 的 head_dim/num_heads) ===
    # DSV4 base 用 MLA head_dim=512, 但 MTP head 用标准 MHA head_dim=128
    mtp_num_heads: int = 0    # 0 = 使用 num_heads
    mtp_head_dim: int = 0     # 0 = 使用 head_dim

    # === 训练参数 ===
    eos_tokens: set[int] = field(default_factory=lambda: {1})
    base_model_path: str = ""          # 默认模型路径 (Host1/Host2)
    tokenizer_path: str = ""           # tokenizer 路径 (edge_first_proxy 用)
    default_system_prompt: str = ""    # 默认 system prompt（空=不注入）
    strip_reasoning_tags: bool = False # 是否移除 <think>...</think> 输出

    # === MoE 参数 (可选) ===
    is_moe: bool = False
    n_routed_experts: int = 0
    num_experts_per_tok: int = 0

    # === 首 token 校准 (edge_first_proxy 用) ===
    first_token_rules: list[FirstTokenRule] = field(default_factory=list)
    default_candidates: list[str] = field(default_factory=lambda: ["The"])

    # === 模型路径模式 (自动检测) ===
    path_patterns: list[str] = field(default_factory=list)

    # === 跨后端 IR ===
    backends: list[str] = field(default_factory=lambda: ["pytorch", "mlx", "sglang"])

    def create_mtp_head(self):
        """创建此模型的 MTP head (延迟导入避免 torch 依赖)."""
        import sys
        import os
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from CGC_Phase2.mtp_head.model import MTPHead, MTPHeadConfig

        config = MTPHeadConfig(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            num_heads=self.mtp_num_heads if self.mtp_num_heads > 0 else self.num_heads,
            head_dim=self.mtp_head_dim if self.mtp_head_dim > 0 else self.head_dim,
            intermediate_size=self.intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
        )
        return MTPHead(config)

    def get_mtp_output_dir(self, base_dir: str = "") -> str:
        """获取此模型 MTP 训练输出目录."""
        root = str(base_dir or _default_mtp_output_base_dir()).rstrip("/")
        return os.path.join(root, self.name)

    def get_checkpoint_path(self, base_dir: str = "") -> str:
        """获取此模型 MTP head checkpoint 的标准路径."""
        output_dir = self.get_mtp_output_dir(base_dir)
        return f"{output_dir}/mtp_head_{self.name}_decode.pt"

    def get_shard_dir(self, base_dir: str = "") -> str:
        """获取训练 shard 目录."""
        if base_dir:
            return os.path.join(str(base_dir).rstrip("/"), self.name)
        return _default_mtp_train_data_dir(self.name)

    def get_embed_head_path(self, base_dir: str = "") -> str:
        """获取训练阶段导出的 embed_head.pt 标准路径."""
        return os.path.join(self.get_shard_dir(base_dir), "embed_head.pt")

    def get_corpus_path(self, base_dir: str = "/data") -> str:
        """获取训练 corpus 路径."""
        return f"{base_dir}/mtp_corpus_{self.name}.jsonl"

    def matches_path(self, path: str) -> bool:
        """检查给定路径是否匹配此模型."""
        path_lower = path.lower()
        for pattern in self.path_patterns:
            if pattern.lower() in path_lower:
                return True
        return False

    def to_dict(self) -> dict:
        """序列化为 dict (State ABI / JSON 配置用)."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "model_type": self.model_type,
            "architectures": self.architectures,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "eos_tokens": sorted(self.eos_tokens),
            "default_system_prompt": self.default_system_prompt,
            "strip_reasoning_tags": self.strip_reasoning_tags,
            "is_moe": self.is_moe,
            "backends": self.backends,
        }


# ============================================================================
# 模型注册表 — 所有支持的模型
# ============================================================================

_GEMMA4_FIRST_TOKEN_RULES = [
    FirstTokenRule("fix", ("fix", "repair", "correct"), 0.90, ["Because", "The", "Here"]),
    FirstTokenRule("refactor", ("refactor", "restructure", "clean up", "improve"), 0.85, ["The", "Here", "I"]),
    FirstTokenRule("optimize", ("optimize", "speed up", "efficient"), 0.85, ["The", "Here", "To"]),
    FirstTokenRule("write", ("write", "implement", "create", "generate"), 0.85, ["Here", "There", "The"]),
    FirstTokenRule("explain", ("explain", "what does", "what do", "describe"), 0.82, ["This", "The", "Here"]),
    FirstTokenRule("debug", ("debug", "traceback", "exception"), 0.78, ["###", "The", "This"]),
    FirstTokenRule("review", ("review", "code review", "check this"), 0.78, ["The", "Here", "Overall"]),
    FirstTokenRule("test", ("unit test", "pytest", "jest"), 0.78, ["The", "Here", "To"]),
    FirstTokenRule("list", ("list", "enumerate", "name "), 0.75, ["Design", "Here", "The"]),
    FirstTokenRule("algo", ("algorithm", "binary search", "sort", "complexity"), 0.75, ["The", "Here", "This"]),
    FirstTokenRule("py_def", ("def ", "def\t"), 0.72, ["To", "The", "Here"]),
    FirstTokenRule("py_class", ("class ", "class\t"), 0.70, ["Since", "The", "Here"]),
    FirstTokenRule("py_import", ("import ", "from "), 0.70, ["It", "The", "Here"]),
    FirstTokenRule("py_self", ("self.", "self->"), 0.68, ["Since", "The", "It"]),
    FirstTokenRule("py_return", ("return ", "return\t"), 0.68, ["In", "The", "Here"]),
    FirstTokenRule("js_const", ("const ", "let "), 0.68, ["It", "The", "Here"]),
    FirstTokenRule("js_func", ("function ", "async function"), 0.68, ["The", "It", "Here"]),
    FirstTokenRule("js_export", ("export "), 0.65, ["In", "The", "Here"]),
    FirstTokenRule("generic_q", ("what is", "tell me about", "how does"), 0.60, ["At", "The", "It"]),
    FirstTokenRule("greeting", ("hello", "hi ", "hey", "how are you"), 0.60, ["I", "Hello", "Hi"]),
]

_DSV4_FIRST_TOKEN_RULES = [
    # DSV4 首 token 模式 (与 Gemma4 不同: DSV4 更倾向 "The" / "I" / "Here")
    FirstTokenRule("fix", ("fix", "repair", "correct"), 0.85, ["The", "Here", "I"]),
    FirstTokenRule("write", ("write", "implement", "create", "generate"), 0.80, ["Here", "The", "I"]),
    FirstTokenRule("explain", ("explain", "what does", "what do", "describe"), 0.78, ["The", "This", "Here"]),
    FirstTokenRule("debug", ("debug", "error", "traceback", "exception"), 0.75, ["The", "Here", "This"]),
    FirstTokenRule("list", ("list", "enumerate", "name "), 0.70, ["Here", "The", "1"]),
    FirstTokenRule("algo", ("algorithm", "binary search", "sort", "complexity"), 0.70, ["The", "Here", "This"]),
    FirstTokenRule("py_def", ("def ", "def\t"), 0.65, ["The", "Here", "This"]),
    FirstTokenRule("py_class", ("class ", "class\t"), 0.65, ["The", "Here", "This"]),
    FirstTokenRule("py_import", ("import ", "from "), 0.65, ["The", "Here", "This"]),
    FirstTokenRule("generic_q", ("what is", "tell me about", "how does"), 0.55, ["The", "I", "Here"]),
    FirstTokenRule("greeting", ("hello", "hi ", "hey", "how are you"), 0.55, ["I", "Hello", "Hi"]),
]

_QWEN3_MOE_FIRST_TOKEN_RULES = [
    FirstTokenRule("fix", ("fix", "repair", "correct"), 0.82, ["The", "Here", "I"]),
    FirstTokenRule("write", ("write", "implement", "create", "generate"), 0.80, ["Here", "The", "I"]),
    FirstTokenRule("explain", ("explain", "what does", "what do", "describe"), 0.78, ["The", "This", "Here"]),
    FirstTokenRule("debug", ("debug", "error", "traceback", "exception"), 0.76, ["The", "Here", "This"]),
    FirstTokenRule("review", ("review", "code review", "check this"), 0.74, ["The", "Overall", "Here"]),
    FirstTokenRule("generic_q", ("what is", "tell me about", "how does"), 0.60, ["The", "It", "Here"]),
]


_REGISTRY: dict[str, ModelConfig] = {

    "gemma4": ModelConfig(
        name="gemma4",
        display_name="Gemma4-26B-A4B",
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        hidden_size=2816,
        vocab_size=262144,
        num_heads=16,
        head_dim=256,
        intermediate_size=14336,
        num_hidden_layers=42,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=40960,
        eos_tokens={1, 106},
        base_model_path="/data/models/gemma-4-26b-a4b-it",
        tokenizer_path="models/gemma-4-mtp-head",
        is_moe=True,
        n_routed_experts=128,
        num_experts_per_tok=4,
        first_token_rules=_GEMMA4_FIRST_TOKEN_RULES,
        default_candidates=["The", "Here", "This", "It", "At", "Since", "In"],
        path_patterns=["gemma-4-26b-a4b", "26b-a4b", "a4b-it", "gemma4_a4b", "gemma4", "gemma_4"],
    ),

    "gemma4_e4b": ModelConfig(
        name="gemma4_e4b",
        display_name="Gemma4-E4B",
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        hidden_size=2560,
        vocab_size=262144,
        num_heads=20,
        head_dim=128,
        intermediate_size=10240,
        num_hidden_layers=42,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=131072,
        eos_tokens={1, 106},
        base_model_path="/data/models/gemma-4-E4B-it",
        tokenizer_path="models/gemma-4-mtp-head",
        is_moe=False,
        first_token_rules=_GEMMA4_FIRST_TOKEN_RULES,
        default_candidates=["The", "Here", "This", "It", "At", "Since", "In"],
        path_patterns=["gemma-4-e4b", "gemma4_e4b", "gemma4-e4b", "e4b-it", "e4b"],
    ),

    "dsv4": ModelConfig(
        name="dsv4",
        display_name="DeepSeek V4 Flash",
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=4096,
        vocab_size=129280,
        num_heads=64,
        head_dim=512,
        intermediate_size=11264,       # 2.75x hidden (MTP head MLP, 非 base MoE intermediate)
        rms_norm_eps=1e-6,
        rope_theta=10000,
        max_position_embeddings=1048576,
        mtp_num_heads=8,               # MTP head 用标准 MHA: 8 heads × 128 dim = 1024
        mtp_head_dim=128,              # 非 base model 的 512 (MLA)
        eos_tokens={1},
        base_model_path="/data/models/DeepSeek-V4-Flash-UD-IQ2",
        tokenizer_path="/data/models/DeepSeek-V4-Flash-UD-IQ2",
        is_moe=True,
        n_routed_experts=256,
        num_experts_per_tok=6,
        first_token_rules=_DSV4_FIRST_TOKEN_RULES,
        default_candidates=["The", "Here", "I", "This"],
        path_patterns=["deepseek-v4", "deepseek_v4", "dsv4", "DeepSeek-V4-Flash"],
    ),

    "qwen3vl": ModelConfig(
        name="qwen3vl",
        display_name="Qwen3-VL-2B",
        model_type="qwen3_vl",
        architectures=["Qwen3VLMoeForConditionalGeneration", "Qwen3VLForConditionalGeneration"],
        hidden_size=2048,
        vocab_size=151936,
        num_heads=16,
        head_dim=128,
        intermediate_size=5632,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=40960,
        eos_tokens={151644, 151645},
        base_model_path="/data/models/Qwen3-VL-2B-Instruct",
        tokenizer_path="/data/models/Qwen3-VL-2B-Instruct",
        is_moe=False,
        first_token_rules=[],  # Qwen3-VL 使用默认规则
        default_candidates=["The", "Here", "I"],
        path_patterns=["qwen3-vl", "qwen3_vl", "Qwen3-VL"],
    ),

    "huihui_moe": ModelConfig(
        name="huihui_moe",
        display_name="Huihui-MoE-0.8B-2E",
        model_type="qwen3_moe",
        architectures=["Qwen3MoeForCausalLM"],
        hidden_size=1024,
        vocab_size=151936,
        num_heads=16,
        head_dim=128,
        intermediate_size=3072,
        num_hidden_layers=28,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=40960,
        eos_tokens={151645},
        base_model_path="models/moe_test/Huihui-MoE-0.8B-2E.Q4_K_M.gguf",
        tokenizer_path="models/moe_test/Huihui-MoE-0.8B-2E_hf",
        default_system_prompt="",
        strip_reasoning_tags=True,
        is_moe=True,
        n_routed_experts=2,
        num_experts_per_tok=1,
        first_token_rules=_QWEN3_MOE_FIRST_TOKEN_RULES,
        default_candidates=["The", "Here", "I", "This"],
        path_patterns=["huihui-moe", "huihui_moe", "qwen3moe", "0.8b-2e"],
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """按注册名获取模型配置.

    支持别名:
        g4, gemma → gemma4
        e4b, gemma4_e4b → gemma4_e4b
        ds, deepseek, v4, flash → dsv4
        qwen, q3 → qwen3vl
    """
    aliases = {
        "g4": "gemma4", "gemma": "gemma4",
        "e4b": "gemma4_e4b", "gemma4_e4b": "gemma4_e4b", "gemma-e4b": "gemma4_e4b",
        "ds": "dsv4", "deepseek": "dsv4", "v4": "dsv4", "flash": "dsv4",
        "qwen": "qwen3vl", "q3": "qwen3vl",
        "huihui": "huihui_moe", "huihui-moe": "huihui_moe", "huihui_moe": "huihui_moe",
    }
    key = aliases.get(name.lower(), name.lower())
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[key]


def get_model_config_by_path(path: str) -> ModelConfig:
    """按模型路径自动检测并返回配置."""
    path_lower = str(path or "").lower()
    best_cfg: Optional[ModelConfig] = None
    best_score = -1
    for cfg in _REGISTRY.values():
        for pattern in cfg.path_patterns:
            token = str(pattern or "").lower()
            if token and token in path_lower and len(token) > best_score:
                best_cfg = cfg
                best_score = len(token)
    if best_cfg is not None:
        return best_cfg
    raise KeyError(f"Cannot detect model from path: {path}")


def get_model_config_by_type(model_type: str) -> ModelConfig:
    """按 AutoConfig.model_type 返回配置."""
    mt = model_type.lower()
    for cfg in _REGISTRY.values():
        if cfg.model_type.lower() == mt:
            return cfg
    raise KeyError(f"Unknown model_type: {model_type}")


def list_models() -> list[str]:
    """列出所有注册的模型名."""
    return list(_REGISTRY.keys())


def list_model_infos() -> list[dict]:
    """列出所有模型的摘要信息."""
    return [cfg.to_dict() for cfg in _REGISTRY.values()]


if __name__ == "__main__":
    print("=" * 70)
    print("CGC Model Registry — 跨后端/跨模型/跨平台")
    print("=" * 70)

    for name in list_models():
        cfg = get_model_config(name)
        print(f"\n  {cfg.display_name} ({cfg.name})")
        print(f"    model_type: {cfg.model_type}")
        print(f"    hidden_size: {cfg.hidden_size}, vocab: {cfg.vocab_size}")
        print(f"    heads: {cfg.num_heads} × dim {cfg.head_dim}")
        print(f"    intermediate: {cfg.intermediate_size}")
        print(f"    rope_theta: {cfg.rope_theta}")
        print(f"    EOS: {sorted(cfg.eos_tokens)}")
        print(f"    MoE: {cfg.is_moe}" + (f" ({cfg.n_routed_experts} experts, {cfg.num_experts_per_tok}/tok)" if cfg.is_moe else ""))
        print(f"    backends: {cfg.backends}")
        print(f"    first_token_rules: {len(cfg.first_token_rules)} rules")
        print(f"    checkpoint: {cfg.get_checkpoint_path()}")
        print(f"    path_patterns: {cfg.path_patterns}")

    print(f"\n  Total: {len(list_models())} models registered")

    # 测试路径检测
    print("\n  Path detection test:")
    for path in [
        "/data/models/gemma-4-26b-a4b-it",
        "/data/models/DeepSeek-V4-Flash-UD-IQ2",
        "/data/models/Qwen3-VL-2B-Instruct",
    ]:
        cfg = get_model_config_by_path(path)
        print(f"    {path} -> {cfg.name}")
