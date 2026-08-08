#!/usr/bin/env python3
"""Edge-first proxy: 四態路由(按模型架構+Mac顯存決策) + edge-first 首 token 預測。

四態路由決策樹(按架構圖):
  1. Mac 可加載完整模型?
     否 → 全雲PD(V4-Flash, 架構不支持)
     是 → 2a. 可用內存 ≥ 全模型+KV+安全緩衝? 是→全本地 llama.cpp
          否 → 2b. 可用內存 ≥ 前P層+激活? 否→降級全雲
               是 → 2c. 時延預判: Mac前P+RTT < 雲全prefill? 是→LayerSplitPD / 否→降級全雲

路由模式:
  local_full       - 全本地 llama.cpp 推理(端側主模型, 顯存夠)
  layer_split_pd   - Mac前P層 + 雲後L-P + MTP投機(MLX前P層, 顯存部分, 時延達標)
  cloud_pd         - 全雲 PD(cloud prefill+decode, V4-Flash完整雲端)
  cloud_fallback   - 降級全雲(時延不達標/MLX失敗/顯存不足)

edge-first 首 token: cloud_pd/cloud_fallback 模式下, 本地tokenizer預測首token(TTFT<10ms)
  + 雲端接續。local/layer_split 模式下由各自推理路徑產出。

TTFT 目標: <100ms(cloud_pd edge-first首token) / 更低(local全本地無RTT)
Decode 速度: 雲端 DSV4+MTP ~37tok/s(native, PD已證通) / local全本地取決Mac GPU

用法:
  python3 edge_first_proxy.py --port 30001 --cloud-url http://127.0.0.1:30000

環境變量(路由):
  EDGE_LOCAL_MAIN_MODEL_PATH: 本地 llama.cpp / GGUF 主模型路徑(空則無法 local_full)
  EDGE_LOCAL_MLX_MODEL_PATH: 本地 MLX layer-split 模型路徑(空則無法 layer_split)
  EDGE_LOCAL_MODEL_PATH: 舊兼容別名；未顯式配置時仍可作為以上兩者回退
  EDGE_LOCAL_NUM_LAYERS: 本地模型層數(默認32)
  EDGE_LOCAL_PARAMS: 本地模型參數量(0=未知, 用於權重預估)
  EDGE_LOCAL_KV_HEAD_DIM / EDGE_LOCAL_KV_HEADS: KV cache 維度(默認128/8)
  EDGE_LOCAL_MEM_SAFETY: 顯存安全餘量(默認0.8, 留20%)
  EDGE_MAC_TFLOPS: Mac GPU算力(默認30)
  EDGE_CLOUD_TFLOPS: 雲端算力(默認3000, 8×RTX PRO)
  EDGE_RTT_SEC: Mac→cloud RTT(默認0.05, 公網; VPC用0.001)
  EDGE_CLOUD_NUM_LAYERS: 雲端模型總層數(默認42, 用於雲prefill預估)

環境變量(edge-first首token):
  EDGE_MODEL_PATH: 本地GGUF小模型路徑(可選, 默認tokenizer預測)
  EDGE_FIRST_ENABLED: 1=啟用edge-first首token, 0=純轉發(默認1)
  CLOUD_URL: 雲端 sglang URL(默認 http://127.0.0.1:30000)
  DSV4_TOKENIZER_PATH: DSV4 tokenizer路徑(首token預測用)

調試:
  GET  /health       - 服務狀態 + 路由配置
  POST /route-test   - 對請求做路由決策(不推理), 返回mode/P/reason/時延預估
"""
from __future__ import annotations

import argparse
import asyncio
import codecs
import copy
import hashlib
import json
import math
import os
import sys
import time
import threading
import urllib.request
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
import uvicorn

from app.shared.task_type_contract import TASK_TYPE_PREFILL
from app.shared.task_type_contract import task_type_contract_ref
from app.shared.acceptance_tracker import get_acceptance_tracker
from app.shared.colibri_backend import build_unified_runtime_ir_v0
from app.shared.draft_registry import get_draft_registry
from app.shared.expert_data_plane import get_expert_data_plane_manager
from app.shared.hermes_router import Bootstrap, VerifyLoopBackend, get_hermes_router
from app.shared.route_decision_v2 import (
    draft_mode_acceptance_report_contract,
    report_contracts as route_report_contracts,
    route_heat_snapshot_report_contract,
    route_policy_v2_report_contract,
    single_node_candidate_matrix_report_contract,
)
from app.shared.transport_route_context import (
    LAYER_SPLIT_MODEL_PATH as _LAYER_SPLIT_MODEL_PATH,
    LOCAL_FULL_MODEL_PATH as _LOCAL_FULL_MODEL_PATH,
    ROUTE_CLOUD_FALLBACK,
    ROUTE_CLOUD_PD,
    ROUTE_LAYER_SPLIT_PD,
    ROUTE_LOCAL_FULL,
    build_transport_route_context as _build_transport_route_context,
    transport_debug_snapshot as _transport_debug_snapshot,
    transport_runtime_snapshot as _transport_runtime_snapshot,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "app"))

# 设置 AcceptanceTracker 状态转换日志
def _on_tracker_transition(old_state: str, new_state: str, rate: float):
    print(f"[edge-first] *** ACCEPTANCE TRACKER: {old_state} → {new_state} (rate={rate:.3f}) ***", file=sys.stderr)

get_acceptance_tracker().set_on_transition(_on_tracker_transition)

# Edge-first 模型（本地小模型，用於首 token）
DEFAULT_EDGE_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
DEFAULT_EDGE_MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# DSV4 tokenizer 路徑（用於 context-aware 首 token 預測，無需 GGUF）
DEFAULT_DSV4_TOKENIZER_PATH = os.environ.get(
    "DSV4_TOKENIZER_PATH",
    "/data/models/DeepSeek-V4-Flash-UD-IQ2",
)

_edge_llm = None
_edge_tokenizer = None
_edge_lock = threading.Lock()
_warmup_state: dict[str, float] = {}
_warmup_lock = threading.Lock()
_local_warmup_state: dict[str, Any] = {
    "status": "idle",
    "started_at": 0.0,
    "finished_at": 0.0,
    "error": "",
    "llama_loaded": False,
    "verify_loop_ready": False,
}
_local_warmup_lock = threading.Lock()
_expert_data_plane = get_expert_data_plane_manager()

EDGE_FIRST_PROFILE_ID = "cgc.edge_first_ttft_v1"
EDGE_FIRST_PROFILE_VERSION = "v1"
EDGE_FIRST_STATE_ABI_ID = "united_pipeline_kernel_v1"
EDGE_FIRST_OUTPUT_CONTRACT_ID = "chat_completion_stream_v1"
EDGE_FIRST_EXECUTION_PROFILE_BINDING_KEY = "cgc.execution.edge_first.prefill.v1"
EDGE_FIRST_BOOTSTRAP_CONTRACT_BINDING_KEY = "cgc.bootstrap.edge_first.prefill.v1"
EDGE_FIRST_FLOW_PARAMETER_CONTRACT_BINDING_KEY = "cgc.flow.edge_first.ttft.v1"

_PROFILE_SETTINGS_PATH = os.environ.get("EDGE_FIRST_PROFILE_SETTINGS_PATH", "")
_BOOTSTRAP_CONTRACT_PATH = os.environ.get("EDGE_FIRST_BOOTSTRAP_CONTRACT_PATH", "")
_SYSTEM_MANIFEST_PATH = os.environ.get("EDGE_FIRST_SYSTEM_MANIFEST_PATH", "")

_DEFAULT_EDGE_SPECULATION_MIN_CONFIDENCE = float(
    os.environ.get("EDGE_FIRST_SPECULATION_MIN_CONFIDENCE", "0.55") or "0.55"
)
_DEFAULT_WARMUP_TTL_SEC = float(
    os.environ.get("EDGE_FIRST_WARMUP_TTL_SEC", "600") or "600"
)
_DEFAULT_ENABLE_PROMPT_NORMALIZATION = (
    os.environ.get("EDGE_FIRST_ENABLE_PROMPT_NORMALIZATION", "1") == "1"
)
_DEFAULT_ENABLE_LOCAL_STARTUP_WARMUP = (
    os.environ.get("EDGE_LOCAL_STARTUP_WARMUP", "1") == "1"
)
_DEFAULT_DISABLE_LOCAL_MTP = os.environ.get("EDGE_DISABLE_LOCAL_MTP", "0") == "1"
_DEFAULT_ENABLE_WARMUP = os.environ.get("EDGE_FIRST_ENABLE_WARMUP", "1") == "1"
_DRAFT_MODE_MIN_SAMPLES = int(os.environ.get("EDGE_FIRST_DRAFT_MODE_MIN_SAMPLES", "8") or "8")
_DRAFT_MODE_MIN_ROI = float(os.environ.get("EDGE_FIRST_DRAFT_MODE_MIN_ROI", "0.0") or "0.0")
_DRAFT_MODE_MIN_JSON_SUCCESS = float(
    os.environ.get("EDGE_FIRST_DRAFT_MODE_MIN_JSON_SUCCESS", "0.95") or "0.95"
)
_EDGE_REPORT_OUTPUT_DIR = os.environ.get(
    "EDGE_FIRST_REPORT_OUTPUT_DIR",
    os.path.join(REPO_ROOT, "ComputeGraphCompiler-main", "Output", "edge_first_proxy_reports"),
)
_PRODUCTION_GATE_TTFT_MS = float(os.environ.get("EDGE_FIRST_PRODUCTION_GATE_TTFT_MS", "300") or "300")
_PRODUCTION_GATE_DECODE_TPS = float(os.environ.get("EDGE_FIRST_PRODUCTION_GATE_DECODE_TPS", "30") or "30")
_PRODUCTION_GATE_STABLE_ROUNDS = int(os.environ.get("EDGE_FIRST_PRODUCTION_GATE_STABLE_ROUNDS", "8") or "8")
_PRODUCTION_GATE_MIN_JSON_SUCCESS = float(
    os.environ.get("EDGE_FIRST_PRODUCTION_GATE_MIN_JSON_SUCCESS", str(_DRAFT_MODE_MIN_JSON_SUCCESS))
    or str(_DRAFT_MODE_MIN_JSON_SUCCESS)
)
_PRODUCTION_GATE_MIN_ROI = float(
    os.environ.get("EDGE_FIRST_PRODUCTION_GATE_MIN_ROI", "0.0") or "0.0"
)
_PRODUCTION_GATE_EXECUTION_SUCCESS_RATE = float(
    os.environ.get("EDGE_FIRST_PRODUCTION_GATE_EXECUTION_SUCCESS_RATE", "1.0") or "1.0"
)
_PRODUCTION_GATE_CONTENT_SUCCESS_RATE = float(
    os.environ.get("EDGE_FIRST_PRODUCTION_GATE_CONTENT_SUCCESS_RATE", "1.0") or "1.0"
)


def _load_edge_model():
    """延遲載入本地小模型（llama-cpp-python）。"""
    global _edge_llm
    with _edge_lock:
        if _edge_llm is not None:
            return _edge_llm
        model_path = os.environ.get("EDGE_MODEL_PATH", "")
        if not model_path:
            return None
        try:
            from llama_cpp import Llama
            _edge_llm = Llama(
                model_path=model_path,
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
            return _edge_llm
        except Exception as e:
            print(f"[edge-first] 載入 {model_path} 失敗: {e}", file=sys.stderr)
            return None


class _TokenizerWrapper:
    """适配 tokenizers.Tokenizer → transformers.AutoTokenizer 接口。
    避免 transformers 5.x AutoTokenizer.from_pretrained bus error。
    """
    def __init__(self, raw_tokenizer, model_path: str):
        self._tok = raw_tokenizer
        self._model_path = model_path
        self._chat_template = None

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs):
        """返回 list[int]（兼容 transformers.encode 接口）。"""
        # tokenizers.Tokenizer.encode 返回 Encoding 对象，且原生支持 add_special_tokens 参数。
        enc = self._tok.encode(text, add_special_tokens=bool(add_special_tokens))
        return enc.ids

    def decode(self, ids, **kwargs):
        return self._tok.decode(ids)

    def apply_chat_template(self, messages, tokenize: bool = True, add_generation_prompt: bool = True, **kwargs):
        """简单 chat template fallback (仅 EDGE_MODEL_PATH 模式使用)。"""
        if self._chat_template is None:
            # 从 tokenizer_config.json 读取 chat_template
            try:
                import json as _json
                import os.path as _osp
                cfg_path = _osp.join(self._model_path, "tokenizer_config.json")
                with open(cfg_path) as f:
                    cfg = _json.load(f)
                self._chat_template = cfg.get("chat_template", "")
            except Exception:
                self._chat_template = ""
        if not self._chat_template:
            # Fallback: 简单拼接
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(parts)
        else:
            # 使用 jinja2 渲染 chat_template
            try:
                from jinja2 import Template
                tpl = Template(self._chat_template)
                prompt = tpl.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)
            except Exception:
                parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                if add_generation_prompt:
                    parts.append("<|im_start|>assistant\n")
                prompt = "\n".join(parts)
        if tokenize:
            return self.encode(prompt, add_special_tokens=False)
        return prompt


def _load_edge_tokenizer():
    """載入 DSV4 tokenizer（用於 context-aware 首 token 預測）。

    這不需要載入模型，只用 tokenizer 做 prompt 分析，
    根據 prompt 類型預測首 token，TTFT < 10ms。
    詞表與雲端 DSV4 完全匹配（129280）。

    優先使用 tokenizers.Tokenizer (Rust, 無 bus error)，
    fallback 到 transformers.AutoTokenizer。
    """
    global _edge_tokenizer
    with _edge_lock:
        if _edge_tokenizer is not None:
            return _edge_tokenizer

        for tokenizer_root in _candidate_tokenizer_paths():
            # 方式 1: tokenizers.Tokenizer 直接加载 (避免 transformers 5.x bus error)
            try:
                from tokenizers import Tokenizer as _RustTokenizer
                import os.path as _osp
                tok_path = _osp.join(tokenizer_root, "tokenizer.json")
                if _osp.exists(tok_path):
                    _raw = _RustTokenizer.from_file(tok_path)
                    _edge_tokenizer = _TokenizerWrapper(_raw, tokenizer_root)
                    print(
                        f"[edge-first] tokenizer 載入成功 (tokenizers直載, root={tokenizer_root}, vocab={_edge_tokenizer.vocab_size})",
                        file=sys.stderr,
                    )
                    return _edge_tokenizer
            except Exception as e:
                print(f"[edge-first] tokenizers 直載失敗 ({tokenizer_root}): {e}, 嘗試 transformers...", file=sys.stderr)

            # 方式 2: transformers.AutoTokenizer (fallback)
            try:
                from transformers import AutoTokenizer
                _edge_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_root, trust_remote_code=True
                )
                print(
                    f"[edge-first] tokenizer 載入成功 (transformers, root={tokenizer_root}, vocab={_edge_tokenizer.vocab_size})",
                    file=sys.stderr,
                )
                return _edge_tokenizer
            except Exception as e:
                print(f"[edge-first] 載入 tokenizer 失敗 ({tokenizer_root}): {e}", file=sys.stderr)
        return None


_ACTIVE_MODEL = "gemma4"  # 默认模型, 可通过 --active-model 切换
_ACTIVE_MODEL_CFG = None  # ModelConfig, 在 _init_model_config 中设置
_PRODUCTION_TARGET_MODEL = os.environ.get("EDGE_FIRST_PRODUCTION_TARGET_MODEL", "gemma4")

def _init_model_config(model_name: str = "gemma4"):
    """从 model_registry 加载模型配置, 动态构建校准规则.

    支持: gemma4 / dsv4 / qwen3vl / huihui_moe (及别名)
    切换模型时, 首 token 校准规则和 tokenizer 路径自动更新.
    """
    global _ACTIVE_MODEL, _ACTIVE_MODEL_CFG
    global _FIRST_TOKEN_PATTERNS, _PROMPT_FAMILY_RULES, _edge_tokenizer
    global DEFAULT_DSV4_TOKENIZER_PATH

    try:
        sys.path.insert(0, REPO_ROOT)
        from app.shared.model_registry import get_model_config
        cfg = get_model_config(model_name)
        _ACTIVE_MODEL = cfg.name
        _ACTIVE_MODEL_CFG = cfg

        # 从 registry 构建 prompt family rules
        _PROMPT_FAMILY_RULES = [
            (rule.family, tuple(rule.markers), rule.confidence,
             list(rule.candidates), rule.enabled)
            for rule in cfg.first_token_rules
        ]

        # 构建 first token patterns (markers, candidates 格式)
        _FIRST_TOKEN_PATTERNS = [
            (list(rule.markers), list(rule.candidates))
            for rule in cfg.first_token_rules
        ]
        # 添加默认 fallback
        _FIRST_TOKEN_PATTERNS.append(([], list(cfg.default_candidates)))

        # 更新 tokenizer 路径
        tokenizer_path = cfg.tokenizer_path or DEFAULT_DSV4_TOKENIZER_PATH
        if tokenizer_path and not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.join(REPO_ROOT, tokenizer_path)
        DEFAULT_DSV4_TOKENIZER_PATH = tokenizer_path
        _edge_tokenizer = None

        print(f"[edge-first] Active model: {cfg.display_name} ({cfg.name}), "
              f"{len(_PROMPT_FAMILY_RULES)} rules, "
              f"tokenizer={DEFAULT_DSV4_TOKENIZER_PATH}", file=sys.stderr)
        return cfg
    except Exception as e:
        print(f"[edge-first] Failed to load model config '{model_name}': {e}, "
              f"using hardcoded defaults", file=sys.stderr)
        return None


def _resolve_production_target() -> dict[str, Any]:
    """返回单机量产验收主模配置。

    当前口径默认锚定 Gemma4-26B-A4B；后续若要切换，只改环境变量或此处即可。
    """
    try:
        from app.shared.model_registry import get_model_config
        cfg = get_model_config(_PRODUCTION_TARGET_MODEL)
        return {
            "name": cfg.name,
            "display_name": cfg.display_name,
            "priority": "P0",
            "is_moe": bool(cfg.is_moe),
            "num_experts_per_tok": int(cfg.num_experts_per_tok or 0),
            "target_params_b": 26 if cfg.name == "gemma4" else 0,
        }
    except Exception:
        return {
            "name": str(_PRODUCTION_TARGET_MODEL or "gemma4"),
            "display_name": "Gemma4-26B-A4B" if str(_PRODUCTION_TARGET_MODEL or "gemma4") == "gemma4" else str(_PRODUCTION_TARGET_MODEL or "gemma4"),
            "priority": "P0",
            "is_moe": True,
            "num_experts_per_tok": 4,
            "target_params_b": 26 if str(_PRODUCTION_TARGET_MODEL or "gemma4") == "gemma4" else 0,
        }


# 常見 prompt 模式 → 首 token 映射（基於 DSV4 實測數據 2026-07-14）
# 數據來源：24 個 prompt 樣本，記錄 DSV4 實際首 token
# 優化後預期準確率：18/24 = 75%（之前 12/24 = 50%）
_FIRST_TOKEN_PATTERNS = [
    # === Gemma4 26B 校准 (2026-07-26 实测) ===
    # 代码修复类（实测: Because）
    (["fix", "repair", "correct"], ["Because", "The", "Here"]),
    # 代碼編寫類（实测: Here/There 各50%）
    (["write", "implement", "create", "generate"], ["Here", "There", "The"]),
    # 代碼解釋類（实测: This）
    (["explain", "what does", "what do", "describe"], ["This", "The", "Here"]),
    # 除錯類（实测: ###）
    (["debug", "error", "traceback", "exception"], ["###", "The", "This"]),
    # 列表類（实测: Design）
    (["list", "enumerate", "name "], ["Design", "Here", "The"]),
    # 算法類（实测: The）
    (["algorithm", "binary search", "sort", "complexity"], ["The", "Here", "This"]),
    # === 代码补全场景 (Gemma4 回应解释) ===
    (["def "], ["To", "The", "Here"]),
    (["class "], ["Since", "The", "Here"]),
    (["import ", "from "], ["It", "The", "Here"]),
    (["self."], ["Since", "The", "It"]),
    (["return "], ["In", "The", "Here"]),
    (["const ", "let "], ["It", "The", "Here"]),
    (["function "], ["The", "It", "Here"]),
    (["export "], ["In", "The", "Here"]),
    # === 通用 ===
    (["what is", "tell me about", "how does"], ["At", "The", "It"]),
    (["hello", "hi ", "hey"], ["I", "Hello", "Hi"]),
    # 預設（Gemma4 实测: The 最常見, 其次 At/It/Here/Since）
    ([], ["The", "Here", "This", "It", "At", "Since", "In"]),
]

_PROMPT_FAMILY_RULES = [
    (
        "cgc_g6_claude",
        (
            "you are claude code",
            "post /api/users/create",
            "database is locked",
        ),
        0.97,
        ["The", "This", "Here"],
        True,
    ),
    (
        "cgc_g6_run",
        (
            "you are a software engineer working on a python codebase",
            "identify all concurrency bugs",
            "race conditions",
        ),
        0.95,
        ["This", "The", "Here"],
        True,
    ),
    # === Gemma4 26B 校准规则 (2026-07-26 实测, 2026-07-30 更新) ===
    ("fix", ("fix", "repair", "correct"), 0.90, ["Because", "The", "Here"], True),
    ("refactor", ("refactor", "restructure", "clean up", "improve"), 0.85, ["The", "Here", "I"], True),
    ("optimize", ("optimize", "speed up", "efficient"), 0.85, ["The", "Here", "To"], True),
    ("write", ("write", "implement", "create", "generate"), 0.85, ["Here", "There", "The"], True),
    ("explain", ("explain", "what does", "what do", "describe"), 0.82, ["This", "The", "Here"], True),
    ("debug", ("debug", "traceback", "exception"), 0.78, ["###", "The", "This"], True),
    ("review", ("review", "code review", "check this"), 0.78, ["The", "Here", "Overall"], True),
    ("test", ("unit test", "pytest", "jest"), 0.78, ["The", "Here", "To"], True),
    ("list", ("list", "enumerate", "name "), 0.75, ["Design", "Here", "The"], True),
    ("algo", ("algorithm", "binary search", "sort", "complexity"), 0.75, ["The", "Here", "This"], True),
    # === 代码补全场景 (Gemma4 回应解释而非补全) ===
    ("py_def", ("def ", "def\t"), 0.72, ["To", "The", "Here"], True),
    ("py_class", ("class ", "class\t"), 0.70, ["Since", "The", "Here"], True),
    ("py_import", ("import ", "from "), 0.70, ["It", "The", "Here"], True),
    ("py_self", ("self.", "self->"), 0.68, ["Since", "The", "It"], True),
    ("py_return", ("return ", "return\t"), 0.68, ["In", "The", "Here"], True),
    ("js_const", ("const ", "let "), 0.68, ["It", "The", "Here"], True),
    ("js_func", ("function ", "async function"), 0.68, ["The", "It", "Here"], True),
    ("js_export", ("export "), 0.65, ["In", "The", "Here"], True),
    # === 通用聊天 (Gemma4 校准) ===
    ("generic_q", ("what is", "tell me about", "how does"), 0.60, ["At", "The", "It"], True),
    ("greeting", ("hello", "hi ", "hey", "how are you"), 0.60, ["I", "Hello", "Hi"], True),
]


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


# 无代码 prompt 的首 token 候选 (基于真实云端 Gemma4 temperature=0 确定性数据)
# 大多数无代码 prompt → 模型回复 "haven't provided the code yet!" → first token "haven"
# 部分家族有确定性差异，按家族区分候选
_NO_CODE_CANDIDATES_DEFAULT = ["haven", "I", "You"]
_NO_CODE_CANDIDATES_BY_FAMILY: dict[str, list[str]] = {
    "fix": ["haven", "I", "You"],
    "debug": ["haven", "I", "You"],
    "write": ["haven", "I", "You"],
    "explain": ["haven", "I", "You"],
    "refactor": ["haven", "I", "You"],
    "review": ["haven", "I", "You"],
    "list": ["haven", "I", "You"],
    "algo": ["haven", "I", "You"],
    # 以下家族在 temperature=0 下确定性返回不同首 token (小写)
    "optimize": ["optimize", "Optimize", "haven"],  # 模型返回 "optimize..."
    "test": ["you", "You", "haven"],  # 模型返回 "you didn't..." 或 "you need to..."
    # 代码补全家族 (无代码时也走 haven)
    "py_def": ["haven", "I", "You"],
    "py_class": ["haven", "I", "You"],
    "py_import": ["haven", "I", "You"],
    "py_self": ["haven", "I", "You"],
    "py_return": ["haven", "I", "You"],
    "js_const": ["haven", "I", "You"],
    "js_func": ["haven", "I", "You"],
    "js_export": ["haven", "I", "You"],
    "generic_q": ["haven", "I", "You"],
}

# 含代码 prompt 的首 token 候选 (基于真实云端 Gemma4 temperature=0 数据)
# 含代码 prompt 的首 token 高度依赖具体代码内容，per-family 准确率有限
# 但 parallel preflight 保证 miss penalty=0，所以即使 MISS 也不影响 TTFT
_CODE_CANDIDATES_BY_FAMILY: dict[str, list[str]] = {
    "fix": ["The", "Because", "error"],       # 模型: "The code...", "Because...", "error in..."
    "debug": ["The", "###", "This"],           # 模型: "The Error...", "### The Cause...", "This error..."
    "review": ["The", "code", "This"],         # 模型: "The code...", "code is...", "This code..."
    "refactor": ["the", "The", "I"],           # 模型: "the original...", "The code...", "I would..."
    "optimize": ["The", "To", "Here"],         # 模型: "The first...", "To optimize...", "Here are..."
    "write": ["Here", "The", "There"],         # 模型: "Here's...", "The following...", "There are..."
    "explain": ["This", "The", "Here"],        # 模型: "This code...", "The function...", "Here's..."
    "test": ["The", "You", "Here"],            # 模型: "The test...", "You can...", "Here's..."
    "list": ["The", "Here", "Design"],         # 模型: "The following...", "Here are...", "Design..."
    "algo": ["The", "Here", "This"],           # 模型: "The algorithm...", "Here's...", "This approach..."
    "py_def": ["The", "To", "Here"],
    "py_class": ["The", "Since", "Here"],
    "py_import": ["The", "It", "Here"],
    "py_self": ["The", "Since", "It"],
    "py_return": ["The", "In", "Here"],
    "js_const": ["The", "It", "Here"],
    "js_func": ["The", "It", "Here"],
    "js_export": ["The", "In", "Here"],
    "generic_q": ["The", "At", "It"],
}

# 内容感知细化: 根据代码内容重排候选 token
# 基于 temperature=0 确定性解码的真实云端数据
_CODE_CONTENT_PATTERNS: list[tuple[str, list[str], list[str]]] = [
    # (family, patterns_to_match_in_lower_msg, reordered_candidates)
    # fix + 明显运行时错误 (类型不匹配/未定义) → 模型说 "error in..."
    ("fix", ["result =", "'"], ["error", "The", "Because"]),
    ("fix", ["result =", '"'], ["error", "The", "Because"]),
    # fix + traceback → 模型说 "The error..."
    ("fix", ["traceback", 'file "'], ["The", "Because", "error"]),
    # review + "review this code" → 模型说 "code looks..."
    ("review", ["review this code"], ["code", "The", "This"]),
    # debug + "debug this error" (无traceback) → 模型说 "The Error..."
    ("debug", ["debug this error"], ["The", "###", "This"]),
    # refactor + markdown code block → 模型说 "the original..."
    ("refactor", ["```"], ["the", "The", "I"]),
]


def _refine_code_candidates(family: str, raw_msg: str, base_candidates: list[str]) -> list[str]:
    """根据代码内容细化候选 token 顺序.

    temperature=0 下模型确定性返回, 但首 token 依赖具体代码内容.
    此函数检查代码内容模式, 重排候选以提升首次预测准确率.
    """
    if not raw_msg:
        return base_candidates
    msg_lower = raw_msg.lower()
    for pat_family, patterns, reordered in _CODE_CONTENT_PATTERNS:
        if family != pat_family:
            continue
        if all(p in msg_lower for p in patterns):
            return list(reordered)
    return base_candidates

# 代码检测标记 — 出现任一标记则认为 prompt 包含实际代码
# 只保留在普通英语中极不可能出现的标记
_CODE_INDICATORS = (
    "```", "{", "}", ";", "def ", "class ", "import ", "const ",
    "var ", "=>", "->", "print(", "console.log", "printf(",
    "system.out", "fmt.", "sprintf", "malloc", "sizeof", "#include",
    "public ", "private ", "protected ", "void ",
    "endif", "endfor", "endwhile", "elseif", "elif",
    "!= ", "== ", "<= ", ">= ", "&&", "||", "++", "--",
    "traceback", 'file "', "at line", "stack trace",
    "error:", "exception:", "nameerror", "typeerror", "valueerror",
    "attributeerror", "keyerror", "indexerror",
)
# 弱标记: 需要与其他标记组合才判定为代码
_CODE_WEAK_INDICATORS = (
    "function(", "async function", "let ", "return ",
    "string ", "bool ", "float ", "double ", "auto ",
    "fn ", "func ", "static ", "echo ",
)


def _has_code(user_msg: str) -> bool:
    """检测 prompt 是否包含实际代码片段."""
    if not user_msg:
        return False
    # 长消息更可能包含代码
    if len(user_msg) > 500:
        return True
    msg_lower = user_msg.lower()
    # 强标记: 出现任一即判定为代码
    for indicator in _CODE_INDICATORS:
        if indicator in msg_lower:
            return True
    # 弱标记: 需要至少 2 个不同的弱标记才判定为代码
    weak_count = sum(1 for ind in _CODE_WEAK_INDICATORS if ind in msg_lower)
    if weak_count >= 2:
        return True
    # 检查多行缩进 (代码通常有缩进行)
    lines = user_msg.split("\n")
    if len(lines) > 2:
        indented = sum(1 for line in lines[1:] if line and line[0] in (" ", "\t"))
        if indented >= 2:
            return True
    return False


def _normalize_message_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    text = content.replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    compact: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run <= 2:
                compact.append("")
            continue
        blank_run = 0
        compact.append(line)
    return "\n".join(compact).strip()


def _extract_user_message(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", "") or "")
    return ""


def _classify_prompt_family(messages: list[dict[str, Any]]) -> dict[str, Any]:
    user_msg = _normalize_text(_extract_user_message(messages))
    raw_msg = _extract_user_message(messages)
    family = "generic"
    confidence = 0.35
    candidates = _FIRST_TOKEN_PATTERNS[-1][1]
    allow_speculation = False

    for name, markers, score, token_candidates, enabled in _PROMPT_FAMILY_RULES:
        if any(marker in user_msg for marker in markers):
            family = name
            confidence = score
            candidates = token_candidates
            allow_speculation = enabled
            break

    # 无代码 prompt → 按 family 使用不同的首 token 候选
    # temperature=0 确定性解码下，同一 family + 无代码 → 模型始终返回相同首 token
    _SKIP_NO_CODE_FAMILIES = {"generic", "greeting"}
    if family not in _SKIP_NO_CODE_FAMILIES and not _has_code(raw_msg):
        candidates = list(_NO_CODE_CANDIDATES_BY_FAMILY.get(family, _NO_CODE_CANDIDATES_DEFAULT))
        confidence = max(confidence, 0.85)
    elif family not in _SKIP_NO_CODE_FAMILIES and _has_code(raw_msg):
        # 含代码 prompt → 使用真实云端数据校准的候选
        # 含代码 prompt 首 token 高度依赖具体代码，准确率有限但 parallel preflight 保证 miss penalty=0
        code_cands = _CODE_CANDIDATES_BY_FAMILY.get(family)
        if code_cands:
            # 内容感知细化: 根据代码内容重排候选以提升首次准确率
            candidates = list(_refine_code_candidates(family, raw_msg, code_cands))
            confidence = max(confidence, 0.65)  # 含代码 prompt 置信度较低

    prompt_hash = hashlib.sha1(user_msg.encode("utf-8")).hexdigest()[:16] if user_msg else ""
    return {
        "family": family,
        "confidence": confidence,
        "candidates": list(candidates),
        "allow_speculation": allow_speculation,
        "prompt_hash": prompt_hash,
    }


def _extract_frontier_id(body: dict[str, Any], request: Optional[Request] = None) -> tuple[str, str]:
    candidate_keys = (
        "conversation_id",
        "session_id",
        "thread_id",
        "request_id",
        "trace_id",
    )
    search_spaces: list[dict[str, Any]] = [body]
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    extra_body = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
    if metadata:
        search_spaces.append(metadata)
    if extra_body:
        search_spaces.append(extra_body)
    for bucket in search_spaces:
        for key in candidate_keys:
            value = str(bucket.get(key) or "").strip()
            if value:
                return value, value
    if request is not None:
        for header_name in ("x-cgc-conversation-id", "x-session-id", "x-thread-id", "x-request-id"):
            value = str(request.headers.get(header_name) or "").strip()
            if value:
                return value, value
    prompt_hash = str(body.get("_edge_prompt_hash") or "").strip()
    if prompt_hash:
        return prompt_hash, f"prompt:{prompt_hash}"
    return "", ""


def _attach_frontier_context(
    family_info: dict[str, Any],
    body: dict[str, Any],
    request: Optional[Request] = None,
) -> dict[str, Any]:
    enriched = dict(family_info or {})
    frontier_id, request_uid = _extract_frontier_id(body, request)
    if not frontier_id:
        prompt_hash = str(enriched.get("prompt_hash") or "").strip()
        frontier_id = prompt_hash
        request_uid = prompt_hash
    enriched["frontier_id"] = frontier_id
    enriched["request_uid"] = request_uid
    return enriched


def _edge_profile_binding_ref(route_family: str) -> dict[str, Any]:
    binding_ref = {
        "profile_id": EDGE_FIRST_PROFILE_ID,
        "profile_version": EDGE_FIRST_PROFILE_VERSION,
        "task_type": TASK_TYPE_PREFILL,
        "initiator": "cgc_edge_first_proxy",
        "state_abi": EDGE_FIRST_STATE_ABI_ID,
        "output_contract": EDGE_FIRST_OUTPUT_CONTRACT_ID,
        "execution_profile_binding_key": EDGE_FIRST_EXECUTION_PROFILE_BINDING_KEY,
        "bootstrap_contract_binding_key": EDGE_FIRST_BOOTSTRAP_CONTRACT_BINDING_KEY,
        "flow_parameter_contract_binding_key": EDGE_FIRST_FLOW_PARAMETER_CONTRACT_BINDING_KEY,
        "route_family": route_family,
        "task_type_contract_ref": task_type_contract_ref(),
    }
    if str(_PROFILE_SETTINGS_PATH or "").strip():
        binding_ref["profile_settings_path"] = str(_PROFILE_SETTINGS_PATH)
    if str(_BOOTSTRAP_CONTRACT_PATH or "").strip():
        binding_ref["bootstrap_contract_path"] = str(_BOOTSTRAP_CONTRACT_PATH)
    if str(_SYSTEM_MANIFEST_PATH or "").strip():
        binding_ref["system_manifest_path"] = str(_SYSTEM_MANIFEST_PATH)
    return binding_ref


def _edge_system_profile_ref(route_family: str) -> dict[str, Any]:
    return {
        "profile_id": EDGE_FIRST_PROFILE_ID,
        "profile_version": EDGE_FIRST_PROFILE_VERSION,
        "source": "cgc_edge_first_proxy",
        "route_family": route_family,
        "source_path": str(_SYSTEM_MANIFEST_PATH or ""),
    }


def _prepare_cloud_payload(body: dict[str, Any], family_info: dict[str, Any]) -> dict[str, Any]:
    payload = dict(body or {})
    route_family = str(family_info.get("family") or "generic")
    # temperature 策略:
    # - 客户端未指定 temperature → 设为 0 (确定性, 提升 cache 准确率)
    # - 客户端显式指定 temperature > 0 → 尊重客户端选择 (general chat 场景)
    #   投机改用 top-k fuzzy match, parallel preflight 保证 miss penalty=0
    if "temperature" not in payload:
        payload["temperature"] = 0.0
    normalized_messages: list[dict[str, Any]] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        normalized_message = dict(message)
        if _DEFAULT_ENABLE_PROMPT_NORMALIZATION:
            if str(message.get("role") or "") == "assistant":
                content = message.get("content")
                normalized_message["content"] = content.replace("\r\n", "\n") if isinstance(content, str) else content
            else:
                normalized_message["content"] = _normalize_message_content(message.get("content"))
        normalized_messages.append(normalized_message)
    payload["messages"] = normalized_messages

    binding_ref = _edge_profile_binding_ref(route_family)
    system_profile_ref = _edge_system_profile_ref(route_family)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    payload["metadata"] = {
        **dict(metadata),
        "task_type": TASK_TYPE_PREFILL,
        "initiator": "cgc_edge_first_proxy",
        "state_abi": EDGE_FIRST_STATE_ABI_ID,
        "output_contract": EDGE_FIRST_OUTPUT_CONTRACT_ID,
        "route_family": route_family,
        "edge_first_confidence": family_info.get("confidence"),
        "edge_first_prompt_hash": family_info.get("prompt_hash"),
        "edge_first_frontier_id": family_info.get("frontier_id"),
        "profile_binding_ref": binding_ref,
        "system_profile_ref": system_profile_ref,
        "task_type_contract_ref": task_type_contract_ref(),
    }
    extra_body = payload.get("extra_body") if isinstance(payload.get("extra_body"), dict) else {}
    payload["extra_body"] = {
        **dict(extra_body),
        "task_type": TASK_TYPE_PREFILL,
        "profile_binding_ref": binding_ref,
        "system_profile_ref": system_profile_ref,
        "route_family": route_family,
        "edge_first_frontier_id": family_info.get("frontier_id"),
    }
    payload["task_type"] = TASK_TYPE_PREFILL
    return payload


def _cloud_request_headers(request_headers: Any, family_info: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in dict(request_headers or {}).items():
        lowered = str(key).lower()
        if lowered in {"host", "content-length"}:
            continue
        headers[str(key)] = str(value)
    headers["x-edge-first-route-family"] = str(family_info.get("family") or "generic")
    headers["x-edge-first-prompt-hash"] = str(family_info.get("prompt_hash") or "")
    headers["x-edge-first-frontier-id"] = str(family_info.get("frontier_id") or "")
    headers["x-cgc-task-type"] = TASK_TYPE_PREFILL
    return headers


# 探索计数器: 当 tracker 阻止投机时, 每 N 次请求探索一次 (epsilon-greedy)
_exploration_counter = 0
_exploration_lock = threading.Lock()
_EXPLORATION_RATE = 10  # 每 10 次被阻止的请求, 探索 1 次


def _should_speculate(family_info: dict[str, Any]) -> bool:
    """是否执行投机 — 由 AcceptanceTracker 三态状态机动态决策.

    ENABLED:  confidence >= 0.55 (宽松)
    DEGRADED: confidence >= 0.70 (保守, 只对高 confidence family 投机)
    DISABLED: 不投机 (parallel preflight 仍工作, miss penalty=0ms)

    探索机制: 当 tracker 阻止投机时, 每 N 次请求探索一次,
    以检测条件是否已改善 (epsilon-greedy).
    """
    family = str(family_info.get("family") or "generic")
    tracker = get_acceptance_tracker()

    # tracker 可以完全禁止投机 (DISABLED 状态)
    if not tracker.should_speculate(family):
        # 探索: 偶尔投机以检测恢复条件
        global _exploration_counter
        with _exploration_lock:
            _exploration_counter += 1
            if _exploration_counter >= _EXPLORATION_RATE:
                _exploration_counter = 0
                print(f"[edge-first] EXPLORATION: trying speculation despite tracker state={tracker.get_state()}", file=sys.stderr)
                return bool(
                    family_info.get("allow_speculation")
                    and float(family_info.get("confidence") or 0.0) >= 0.55  # 用 ENABLED 的 threshold 探索
                )
        return False

    # tracker 动态调整 confidence threshold
    min_confidence = tracker.get_min_confidence()
    return bool(
        family_info.get("allow_speculation")
        and float(family_info.get("confidence") or 0.0) >= min_confidence
    )


def _maybe_schedule_warmup(cloud_endpoint: str, payload: dict[str, Any], headers: dict[str, str], family_info: dict[str, Any]) -> None:
    if not _DEFAULT_ENABLE_WARMUP:
        return
    route_family = str(family_info.get("family") or "generic")
    prompt_hash = str(family_info.get("prompt_hash") or "")
    if not route_family or not prompt_hash:
        return
    cache_key = f"{route_family}:{prompt_hash}"
    now = time.monotonic()
    with _warmup_lock:
        last = _warmup_state.get(cache_key, 0.0)
        if now - last < _DEFAULT_WARMUP_TTL_SEC:
            return
        _warmup_state[cache_key] = now

    warm_payload = {
        **payload,
        "stream": False,
        "max_tokens": min(int(payload.get("max_tokens", 32) or 32), 8),
        "temperature": 0.0,
    }

    def _runner():
        try:
            req = urllib.request.Request(
                cloud_endpoint,
                data=json.dumps(warm_payload).encode("utf-8"),
                headers={**headers, "Content-Type": "application/json", "x-edge-first-warmup": "1"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                resp.read()
            print(f"[edge-first] warmup ok family={route_family} hash={prompt_hash}", file=sys.stderr)
        except Exception as exc:
            print(f"[edge-first] warmup miss family={route_family} hash={prompt_hash}: {exc}", file=sys.stderr)

    threading.Thread(target=_runner, daemon=True).start()


# === 首 token 预测缓存 (多级缓存) ===
# L1: exact prompt_hash → first_token (重复 prompt → 100% 准确, TTFT ~0ms)
# L2: prefix_hash (前 256 chars normalized) → first_token (相似 prompt共享)
# L3: context_tail_hash (末 128 chars) → first_token (代码补全尾部模式匹配)
# L4: semantic trigram Jaccard similarity → first_token (语义相似 prompt)
# L5: family pattern (规则匹配, 最低优先级)
_first_token_cache: dict[str, str] = {}          # L1: exact
_prefix_token_cache: dict[str, str] = {}          # L2: prefix
_tail_token_cache: dict[str, str] = {}            # L3: context tail
_semantic_cache: list[dict] = []                  # L4: semantic [{trigrams: set, token: str, prompt_hash: str}]
_first_token_cache_lock = threading.Lock()
_FIRST_TOKEN_CACHE_MAX = 2000  # LRU 上限 (每级)
_SEMANTIC_CACHE_MAX = 1000     # L4 语义缓存上限
_SEMANTIC_THRESHOLD = 0.75     # Jaccard 相似度阈值


def _compute_trigrams(text: str) -> frozenset:
    """计算文本的字符 trigram 集合 (用于语义相似度匹配).

    标准化: 小写 + 去多余空白, 然后取 3-gram.
    """
    normalized = " ".join(str(text or "").lower().split())
    if len(normalized) < 3:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(normalized[i:i+3] for i in range(len(normalized) - 2))


def _jaccard_similarity(set_a: frozenset, set_b: frozenset) -> float:
    """计算两个集合的 Jaccard 相似度."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _check_semantic_cache(user_msg: str) -> tuple[Optional[str], float]:
    """检查语义缓存 (L4).

    遍历语义缓存, 找到 Jaccard 相似度最高的条目.
    如果相似度 >= _SEMANTIC_THRESHOLD, 返回对应的 first_token.

    Returns:
        (first_token, similarity) or (None, 0.0)
    """
    if not user_msg:
        return None, 0.0
    query_trigrams = _compute_trigrams(user_msg)
    if not query_trigrams:
        return None, 0.0

    best_token = None
    best_sim = 0.0

    with _first_token_cache_lock:
        for entry in _semantic_cache:
            sim = _jaccard_similarity(query_trigrams, entry["trigrams"])
            if sim > best_sim:
                best_sim = sim
                best_token = entry["token"]
                if sim >= 0.95:
                    break  # 几乎完全匹配, 直接返回

    if best_sim >= _SEMANTIC_THRESHOLD:
        return best_token, best_sim
    return None, 0.0


def _record_semantic(user_msg: str, first_token: str, prompt_hash: str) -> None:
    """记录到语义缓存 (L4).

    只在 L1-L3 都 miss 时才记录 (避免重复).
    """
    if not user_msg or not first_token:
        return
    trigrams = _compute_trigrams(user_msg)
    if not trigrams:
        return

    with _first_token_cache_lock:
        # 避免重复 (检查是否已有高相似度条目)
        for entry in _semantic_cache:
            if _jaccard_similarity(trigrams, entry["trigrams"]) >= 0.90:
                # 更新已有条目 (用最新的 prompt_hash)
                entry["token"] = first_token
                entry["prompt_hash"] = prompt_hash
                return

        # 添加新条目
        if len(_semantic_cache) >= _SEMANTIC_CACHE_MAX:
            # 删除最旧的一半
            del _semantic_cache[:_SEMANTIC_CACHE_MAX // 2]

        _semantic_cache.append({
            "trigrams": trigrams,
            "token": first_token,
            "prompt_hash": prompt_hash,
        })

# === temperature>0 适配 ===
# temperature=0: 确定性解码, 单 token 精确匹配 (code completion 场景, 75% accept)
# 0 < temp <= 1.0: 非确定性, 但分布集中 → top-k 预测 (general chat 场景)
# temp > 1.0: 高度随机, 禁用投机 (creative 场景, 依赖 parallel preflight)
_TEMP_BAND_DETERMINISTIC = "deterministic"   # temp == 0
_TEMP_BAND_LOW = "low"                        # 0 < temp <= 0.7
_TEMP_BAND_MEDIUM = "medium"                  # 0.7 < temp <= 1.0
_TEMP_BAND_HIGH = "high"                      # temp > 1.0 (禁用投机)
_TEMP_SPECULATION_DISABLE_THRESHOLD = 1.0     # temp > 1.0 → 不投机
_TEMP_TOPK_SIZE = 5                           # top-k 候选数量


def _get_temperature_band(temperature: float) -> str:
    """根据 temperature 值返回温度带.

    - deterministic (temp=0): 确定性解码, 单 token 精确匹配
    - low (0 < temp <= 0.7): 分布较集中, top-k 有效
    - medium (0.7 < temp <= 1.0): 分布较分散, top-k 仍可能命中
    - high (temp > 1.0): 高度随机, 禁用投机
    """
    t = float(temperature)
    if t <= 0.0:
        return _TEMP_BAND_DETERMINISTIC
    elif t <= 0.7:
        return _TEMP_BAND_LOW
    elif t <= _TEMP_SPECULATION_DISABLE_THRESHOLD:
        return _TEMP_BAND_MEDIUM
    else:
        return _TEMP_BAND_HIGH


def _should_speculate_for_temperature(temperature: float, family_info: dict) -> bool:
    """temperature>0 时是否值得投机.

    high band (>1.0): 不投机 (太随机)
    medium band (0.7-1.0): 投机但用 fuzzy hit 判定
    low band (0-0.7): 投机, top-k 命中率高
    deterministic (0): 投机, 精确匹配
    """
    band = _get_temperature_band(temperature)
    if band == _TEMP_BAND_HIGH:
        return False
    return True


def _predict_first_token_topk(messages: list, temperature: float, k: int = None) -> tuple[Optional[str], list[str]]:
    """temperature>0 适配: 返回 (首选 token, top-k 候选列表).

    对于 temperature=0: 返回 (predicted_token, [predicted_token])
    对于 temperature>0: 返回 (most_likely_token, [top_k_candidates])

    策略:
    1. 先查 L1-L4 缓存 (如果命中, 说明该 prompt 之前出现过, 首选 token 可靠)
    2. 如果缓存 miss, 从 family pattern 取 top-k 候选
    3. 对于 temperature>0, 额外收集该 family 的所有历史首 token 作为候选

    Returns:
        (first_choice_token, all_candidate_tokens)
        first_choice_token: 最可能的首 token (用于投机发送)
        all_candidate_tokens: top-k 候选列表 (用于 fuzzy hit 判定)
    """
    if k is None:
        k = _TEMP_TOPK_SIZE

    family_info = _classify_prompt_family(messages)
    user_msg = _extract_user_message(messages)
    _prompt_hash = str(family_info.get("prompt_hash") or "")

    # 1. 检查缓存 (L1-L4)
    cached_token = None
    if _prompt_hash:
        with _first_token_cache_lock:
            cached_token = _first_token_cache.get(_prompt_hash)
    if not cached_token and user_msg:
        _prefix_hash = _compute_prefix_hash(user_msg)
        if _prefix_hash:
            with _first_token_cache_lock:
                cached_token = _prefix_token_cache.get(_prefix_hash)
    if not cached_token and user_msg:
        _tail_hash = _compute_tail_hash(user_msg)
        if _tail_hash:
            with _first_token_cache_lock:
                cached_token = _tail_token_cache.get(_tail_hash)
    if not cached_token and user_msg:
        sem_token, _ = _check_semantic_cache(user_msg)
        cached_token = sem_token

    # 2. 收集 family pattern 候选
    pattern_candidates = list(family_info.get("candidates") or _FIRST_TOKEN_PATTERNS[-1][1])

    # 3. 构建 top-k 列表
    all_candidates: list[str] = []
    if cached_token:
        all_candidates.append(cached_token)
    for c in pattern_candidates:
        if c not in all_candidates:
            all_candidates.append(c)
        if len(all_candidates) >= k:
            break

    # 补充通用候选 (确保至少 k 个)
    _FILLER_CANDIDATES = ["The", "I", "Here", "This", "You", "haven", "To", "In", "It", "Sure"]
    for c in _FILLER_CANDIDATES:
        if c not in all_candidates:
            all_candidates.append(c)
        if len(all_candidates) >= k:
            break

    first_choice = cached_token or (all_candidates[0] if all_candidates else "The")
    return first_choice, all_candidates[:k]


# === 统计追踪 ===
_stats_lock = threading.Lock()
_REQUEST_PATH_COLD = "cold_path"
_REQUEST_PATH_WARM_HOT = "warm_hot_path"


def _new_request_path_metrics() -> dict[str, Any]:
    return {
        "requests": 0,
        "completed_requests": 0,
        "failed_requests": 0,
        "content_completed_requests": 0,
        "empty_output_requests": 0,
        "ttft_ms_sum": 0.0,
        "ttft_ms_count": 0,
        "ttft_ms_min": 999999.0,
        "ttft_ms_max": 0.0,
        "decode_tokens_sum": 0,
        "decode_elapsed_ms_sum": 0.0,
        "decode_sample_count": 0,
        "stable_consecutive_successes": 0,
        "stable_rounds_passed_max": 0,
    }


def _new_stats_state() -> dict[str, Any]:
    return {
        "total_requests": 0,
        "completed_requests": 0,
        "failed_requests": 0,
        "content_completed_requests": 0,
        "empty_output_requests": 0,
        "speculated": 0,           # 尝试投机次数
        "cache_hit_l1": 0,         # L1 exact hit
        "cache_hit_l2": 0,         # L2 prefix hit
        "cache_hit_l3": 0,         # L3 tail hit
        "cache_hit_l4": 0,         # L4 semantic hit
        "cache_miss": 0,           # 全 miss, 用 pattern
        "speculation_correct": 0,  # 投机命中 (首token == cloud首token)
        "speculation_wrong": 0,    # 投机未命中
        "speculation_fuzzy_hit": 0, # fuzzy hit: cloud 首 token 在 top-k 候选中 (temp>0)
        "speculation_fuzzy_miss": 0, # fuzzy miss: cloud 首 token 不在 top-k (temp>0)
        "temp_disabled": 0,        # temperature>1.0 禁用投机次数
        "ttft_ms_sum": 0.0,        # 累计 TTFT (ms)
        "ttft_ms_count": 0,        # TTFT 样本数
        "ttft_ms_min": 999999.0,
        "ttft_ms_max": 0.0,
        "decode_tokens_sum": 0,
        "decode_elapsed_ms_sum": 0.0,
        "decode_sample_count": 0,
        "stable_consecutive_successes": 0,
        "stable_rounds_passed_max": 0,
        "family_counts": {},       # family → count
        "temp_band_counts": {},    # temperature band → count
        "path_metrics": {
            _REQUEST_PATH_COLD: _new_request_path_metrics(),
            _REQUEST_PATH_WARM_HOT: _new_request_path_metrics(),
        },
    }


def _normalize_request_path_kind(path_kind: Any) -> str:
    return _REQUEST_PATH_WARM_HOT if str(path_kind or "") == _REQUEST_PATH_WARM_HOT else _REQUEST_PATH_COLD


def _classify_request_path_kind(expert_session: Optional[Any]) -> str:
    if expert_session is None:
        return _REQUEST_PATH_COLD
    plan = getattr(expert_session, "plan", None)
    if plan is None or not bool(getattr(plan, "enabled", False)):
        return _REQUEST_PATH_COLD
    cold_bytes = max(int(getattr(plan, "cold_bytes", 0) or 0), 0)
    cache_hits = max(int(getattr(expert_session, "cache_hits", 0) or 0), 0)
    prefetch_hits = max(int(getattr(expert_session, "prefetch_hits", 0) or 0), 0)
    loaded_keys = len(list(getattr(expert_session, "loaded_keys", []) or []))
    if cold_bytes > 0:
        return _REQUEST_PATH_COLD
    if cache_hits > 0 or prefetch_hits > 0 or loaded_keys > 0:
        return _REQUEST_PATH_WARM_HOT
    return _REQUEST_PATH_COLD


def _record_ttft_sample(ttft_ms: float, *, path_kind: str) -> None:
    _record_stats(
        ttft_sample={
            "ttft_ms": ttft_ms,
            "path_kind": path_kind,
        }
    )


def _finalize_request_metrics(bucket: dict[str, Any]) -> dict[str, Any]:
    d = dict(bucket)
    if d["ttft_ms_count"] > 0:
        d["ttft_ms_avg"] = round(d["ttft_ms_sum"] / d["ttft_ms_count"], 1)
    else:
        d["ttft_ms_avg"] = 0.0
    d["ttft_ms_min"] = round(d["ttft_ms_min"], 1) if d["ttft_ms_min"] < 999999 else 0.0
    d["ttft_ms_max"] = round(d["ttft_ms_max"], 1)
    if d["decode_elapsed_ms_sum"] > 0 and d["decode_tokens_sum"] > 0:
        d["decode_tps_avg"] = round(d["decode_tokens_sum"] / d["decode_elapsed_ms_sum"] * 1000, 1)
    else:
        d["decode_tps_avg"] = 0.0
    total_finished = int(d.get("completed_requests", 0)) + int(d.get("failed_requests", 0))
    d["request_success_rate"] = round(int(d.get("completed_requests", 0)) / total_finished, 3) if total_finished > 0 else 0.0
    d["execution_success_rate"] = d["request_success_rate"]
    d["content_success_rate"] = round(int(d.get("content_completed_requests", 0)) / total_finished, 3) if total_finished > 0 else 0.0
    d.pop("ttft_ms_sum", None)
    d.pop("decode_elapsed_ms_sum", None)
    return d


_stats = _new_stats_state()
_draft_mode_stats_lock = threading.Lock()
_draft_mode_stats: dict[str, dict[str, Any]] = {}
_report_snapshot_lock = threading.Lock()
_last_route_policy_snapshot: dict[str, Any] = route_policy_v2_report_contract()
_last_heat_snapshot: dict[str, Any] = route_heat_snapshot_report_contract()
_last_single_node_candidate_matrix: dict[str, Any] = single_node_candidate_matrix_report_contract()


def _record_stats(**kwargs):
    """更新统计数据 (线程安全)."""
    with _stats_lock:
        for key, val in kwargs.items():
            if key == "family":
                _stats["family_counts"][val] = _stats["family_counts"].get(val, 0) + 1
            elif key == "temp_band":
                _stats["temp_band_counts"][val] = _stats["temp_band_counts"].get(val, 0) + 1
            elif key == "ttft_sample":
                sample = val if isinstance(val, dict) else {}
                ms = max(float(sample.get("ttft_ms") or 0.0), 0.0)
                path_kind = _normalize_request_path_kind(sample.get("path_kind"))
                _stats["ttft_ms_sum"] += ms
                _stats["ttft_ms_count"] += 1
                _stats["ttft_ms_min"] = min(_stats["ttft_ms_min"], ms)
                _stats["ttft_ms_max"] = max(_stats["ttft_ms_max"], ms)
                path_bucket = _stats["path_metrics"].setdefault(path_kind, _new_request_path_metrics())
                path_bucket["ttft_ms_sum"] += ms
                path_bucket["ttft_ms_count"] += 1
                path_bucket["ttft_ms_min"] = min(path_bucket["ttft_ms_min"], ms)
                path_bucket["ttft_ms_max"] = max(path_bucket["ttft_ms_max"], ms)
            elif key == "decode_sample":
                sample = val if isinstance(val, dict) else {}
                tokens = max(int(sample.get("tokens") or 0), 0)
                elapsed_ms = max(float(sample.get("elapsed_ms") or 0.0), 0.0)
                success = bool(sample.get("success", False))
                content_success = bool(sample.get("content_success", False))
                path_kind = _normalize_request_path_kind(sample.get("path_kind"))
                path_bucket = _stats["path_metrics"].setdefault(path_kind, _new_request_path_metrics())
                path_bucket["requests"] += 1
                if tokens > 0 and elapsed_ms > 0:
                    _stats["decode_tokens_sum"] += tokens
                    _stats["decode_elapsed_ms_sum"] += elapsed_ms
                    _stats["decode_sample_count"] += 1
                    path_bucket["decode_tokens_sum"] += tokens
                    path_bucket["decode_elapsed_ms_sum"] += elapsed_ms
                    path_bucket["decode_sample_count"] += 1
                if success:
                    _stats["completed_requests"] += 1
                    path_bucket["completed_requests"] += 1
                    if content_success:
                        _stats["content_completed_requests"] += 1
                        path_bucket["content_completed_requests"] += 1
                    else:
                        _stats["empty_output_requests"] += 1
                        path_bucket["empty_output_requests"] += 1
                    _stats["stable_consecutive_successes"] += 1
                    _stats["stable_rounds_passed_max"] = max(
                        _stats["stable_rounds_passed_max"],
                        _stats["stable_consecutive_successes"],
                    )
                    path_bucket["stable_consecutive_successes"] += 1
                    path_bucket["stable_rounds_passed_max"] = max(
                        path_bucket["stable_rounds_passed_max"],
                        path_bucket["stable_consecutive_successes"],
                    )
                else:
                    _stats["failed_requests"] += 1
                    path_bucket["failed_requests"] += 1
                    _stats["stable_consecutive_successes"] = 0
                    path_bucket["stable_consecutive_successes"] = 0
            elif key in _stats:
                _stats[key] += val


def _get_stats() -> dict:
    """获取统计数据快照."""
    with _stats_lock:
        d = copy.deepcopy(_stats)
    d = _finalize_request_metrics(d)
    total_hits = d["cache_hit_l1"] + d["cache_hit_l2"] + d["cache_hit_l3"] + d.get("cache_hit_l4", 0)
    total_cacheable = total_hits + d["cache_miss"]
    d["cache_hit_rate"] = round(total_hits / total_cacheable, 3) if total_cacheable > 0 else 0
    speculated = d["speculated"]
    d["speculation_rate"] = round(d["speculation_correct"] / speculated, 3) if speculated > 0 else 0
    # fuzzy hit rate (temperature>0 专用)
    fuzzy_total = d.get("speculation_fuzzy_hit", 0) + d.get("speculation_fuzzy_miss", 0)
    d["fuzzy_hit_rate"] = round(d.get("speculation_fuzzy_hit", 0) / fuzzy_total, 3) if fuzzy_total > 0 else 0
    # 综合命中率: exact + fuzzy
    total_exact = d.get("speculation_correct", 0) + d.get("speculation_wrong", 0)
    d["combined_hit_rate"] = round(
        (d.get("speculation_correct", 0) + d.get("speculation_fuzzy_hit", 0)) /
        (total_exact + fuzzy_total), 3
    ) if (total_exact + fuzzy_total) > 0 else 0
    d["path_metrics"] = {
        _REQUEST_PATH_COLD: _finalize_request_metrics(
            dict((d.get("path_metrics") or {}).get(_REQUEST_PATH_COLD) or _new_request_path_metrics())
        ),
        _REQUEST_PATH_WARM_HOT: _finalize_request_metrics(
            dict((d.get("path_metrics") or {}).get(_REQUEST_PATH_WARM_HOT) or _new_request_path_metrics())
        ),
    }
    return d


def _new_draft_mode_bucket() -> dict[str, Any]:
    return {
        "requests": 0,
        "speculative_requests": 0,
        "hits": 0,
        "misses": 0,
        "speculation_cost_ms": 0.0,
        "speculation_benefit_ms": 0.0,
        "json_successes": 0,
        "json_failures": 0,
        "auto_disabled": False,
        "disable_reason": "",
        "last_update_ts": 0.0,
    }


def _get_draft_mode_bucket(draft_mode: str) -> dict[str, Any]:
    with _draft_mode_stats_lock:
        return dict(_draft_mode_stats.setdefault(draft_mode, _new_draft_mode_bucket()))


def _draft_mode_roi(bucket: dict[str, Any]) -> float:
    cost = float(bucket.get("speculation_cost_ms", 0.0) or 0.0)
    benefit = float(bucket.get("speculation_benefit_ms", 0.0) or 0.0)
    if cost <= 0:
        return 0.0
    return round((benefit - cost) / cost, 3)


def _draft_mode_json_success_rate(bucket: dict[str, Any]) -> float:
    total = int(bucket.get("json_successes", 0)) + int(bucket.get("json_failures", 0))
    if total <= 0:
        return 1.0
    return round(int(bucket.get("json_successes", 0)) / total, 3)


def _snapshot_draft_mode_stats() -> dict[str, Any]:
    with _draft_mode_stats_lock:
        raw = {k: dict(v) for k, v in _draft_mode_stats.items()}
    result: dict[str, Any] = {}
    for mode, bucket in raw.items():
        bucket["roi"] = _draft_mode_roi(bucket)
        bucket["json_success_rate"] = _draft_mode_json_success_rate(bucket)
        result[mode] = bucket
    return result


def _detect_response_contract(body: dict[str, Any], family_info: dict[str, Any]) -> tuple[str, str]:
    response_format = body.get("response_format") if isinstance(body.get("response_format"), dict) else {}
    response_type = str(response_format.get("type") or "").lower()
    if isinstance(body.get("tools"), list) and body.get("tools"):
        return "tool", "tool_call"
    if response_type in {"json_object", "json_schema"}:
        return "json", "schema" if response_type == "json_schema" else "json"
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    user_msg = _normalize_text(_extract_user_message(messages))
    if any(marker in user_msg for marker in ["json only", "strict json", "respond in json", "json"]):
        return "json", "json"
    family = str(family_info.get("family") or "generic")
    if family in {"json", "tool", "schema"}:
        return ("tool", "tool_call") if family == "tool" else ("json", "json")
    return "plain", "off"


def _resolve_local_runtime_model_cfg():
    model_path = str(_LOCAL_FULL_MODEL_PATH or "").strip()
    if not model_path:
        return None
    try:
        from app.shared.model_registry import get_model_config_by_path
        return get_model_config_by_path(model_path)
    except Exception:
        return None


def _resolve_hermes_model_name(body: dict[str, Any]) -> str:
    raw_model = str(body.get("model") or "").strip().lower()
    local_cfg = _resolve_local_runtime_model_cfg()
    local_name = str(getattr(local_cfg, "name", "") or "").strip().lower()
    if local_name:
        if not raw_model:
            return local_name
        if "gemma" in raw_model and local_name.startswith("gemma4"):
            return local_name
        if ("deepseek" in raw_model or "dsv4" in raw_model) and local_name.startswith("dsv4"):
            return local_name
        if "qwen" in raw_model and local_name.startswith("qwen3vl"):
            return local_name
        if raw_model == local_name:
            return local_name
    try:
        from app.shared.model_registry import get_model_config
        if raw_model:
            return str(get_model_config(raw_model).name)
    except Exception:
        pass
    if "gemma" in raw_model:
        return "gemma4"
    if "deepseek" in raw_model or "dsv4" in raw_model:
        return "dsv4"
    if "qwen" in raw_model:
        return "qwen3vl"
    return str(_ACTIVE_MODEL or "gemma4")


def _local_mtp_runtime_status(
    *,
    model_name: str,
    route_mode: str,
    draft_n_tokens: int,
) -> dict[str, Any]:
    status = {
        "available": False,
        "reason": "",
        "draft_n_tokens": int(draft_n_tokens or 0),
        "draft_path": "",
        "draft_path_exists": False,
        "verify_loop_configured": False,
        "executor_support": True,
        "mtp_checkpoint": "",
        "mtp_checkpoint_exists": False,
        "assistant_model_path": "",
        "assistant_model_exists": False,
        "speculator_backend": "none",
        "startup_warmup_status": "",
        "startup_warmup_error": "",
        "training_target": "",
        "expected_checkpoint_path": "",
        "expected_embed_head_path": "",
    }
    if draft_n_tokens <= 0:
        status["reason"] = "local_mtp_not_requested"
        return status
    if str(route_mode or "") != ROUTE_LOCAL_FULL:
        status["reason"] = "non_local_full_route"
        return status
    if _DEFAULT_DISABLE_LOCAL_MTP:
        status["executor_support"] = False
        status["reason"] = "local_mtp_unavailable:forced_disabled_by_env"
        return status

    draft_registry = get_draft_registry()
    entry = draft_registry.get_entry(model_name)
    if entry is None:
        entry = draft_registry.register_from_registry(model_name)
    if entry is not None:
        status["draft_path"] = str(entry.draft_path or "")
        status["draft_path_exists"] = bool(entry.draft_path) and os.path.exists(entry.draft_path)

    model_info = dict(Bootstrap.DRAFT_MODELS.get(model_name) or {})
    verify_cfg = dict(model_info.get("verify_loop_config") or {})
    resolved_cfg = _resolve_active_model_cfg(model_name)
    if resolved_cfg is not None:
        status["training_target"] = str(getattr(resolved_cfg, "name", "") or "")
        try:
            status["expected_checkpoint_path"] = str(resolved_cfg.get_checkpoint_path())
            status["expected_embed_head_path"] = str(resolved_cfg.get_embed_head_path())
        except Exception:
            pass

    status["verify_loop_configured"] = bool(verify_cfg)
    status["mtp_checkpoint"] = str(status["expected_checkpoint_path"] or verify_cfg.get("mtp_checkpoint") or "")
    status["mtp_checkpoint_exists"] = bool(status["mtp_checkpoint"]) and os.path.exists(status["mtp_checkpoint"])
    status["assistant_model_path"] = str(verify_cfg.get("assistant_model_path") or "")
    status["assistant_model_exists"] = bool(status["assistant_model_path"]) and os.path.exists(status["assistant_model_path"])
    status["speculator_backend"] = (
        "official_assistant_proxy"
        if status["assistant_model_exists"]
        else ("verify_loop_ngram" if status["verify_loop_configured"] else "none")
    )

    reasons: list[str] = []
    if not status["draft_path"]:
        reasons.append("draft_path_missing")
    elif not status["draft_path_exists"]:
        reasons.append(f"draft_path_not_found:{status['draft_path']}")
    if not status["verify_loop_configured"]:
        reasons.append("verify_loop_config_missing")
    if status["verify_loop_configured"] and not status["mtp_checkpoint_exists"] and not status["assistant_model_exists"]:
        reasons.append("trained_mtp_checkpoint_missing:using_ngram_fallback")
    warmup_status = dict(_local_warmup_state)
    status["startup_warmup_status"] = str(warmup_status.get("status") or "")
    status["startup_warmup_error"] = str(warmup_status.get("error") or "")
    if status["verify_loop_configured"] and _DEFAULT_ENABLE_LOCAL_STARTUP_WARMUP:
        if str(warmup_status.get("status") or "") == "running":
            status["executor_support"] = False
            reasons.append("local_startup_warmup_running")
        elif not bool(warmup_status.get("verify_loop_ready")):
            status["executor_support"] = False
            warmup_error = str(warmup_status.get("error") or "").strip()
            reasons.append(warmup_error or "verify_loop_runtime_not_ready")
    status["available"] = status["verify_loop_configured"] and status["executor_support"]
    reason_prefix = "local_mtp_available:" if status["available"] else "local_mtp_unavailable:"
    status["reason"] = reason_prefix + ",".join(reasons)
    # #region debug-point A:local-mtp-runtime-status
    try:
        _dbg_p = os.path.join(REPO_ROOT, ".dbg", "dense-streaming-measure.env")
        _dbg_u, _dbg_s = "http://127.0.0.1:7777/event", "dense-streaming-measure"
        try:
            with open(_dbg_p, "r", encoding="utf-8") as _dbg_f:
                _dbg_c = _dbg_f.read()
            _dbg_u = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SERVER_URL=")), _dbg_u)
            _dbg_s = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SESSION_ID=")), _dbg_s)
        except Exception:
            pass
        urllib.request.urlopen(
            urllib.request.Request(
                _dbg_u,
                data=json.dumps({
                    "sessionId": _dbg_s,
                    "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                    "hypothesisId": "A",
                    "location": "app/servers/edge_first_proxy.py:_local_mtp_runtime_status",
                    "msg": "[DEBUG] evaluated local_mtp runtime availability",
                    "data": {
                        "model_name": str(model_name),
                        "route_mode": str(route_mode),
                        "draft_n_tokens": int(draft_n_tokens or 0),
                        "available": bool(status.get("available")),
                        "reason": str(status.get("reason") or ""),
                        "verify_loop_configured": bool(status.get("verify_loop_configured")),
                        "executor_support": bool(status.get("executor_support")),
                        "speculator_backend": str(status.get("speculator_backend") or ""),
                    },
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.35,
        ).read()
    except Exception:
        pass
    # #endregion
    return status


def _build_draft_policy_from_hermes(
    body: dict[str, Any],
    family_info: dict[str, Any],
    route_context: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    prompt = _extract_user_message(body.get("messages") or [])
    model_name = _resolve_hermes_model_name(body)
    tracker = get_acceptance_tracker()
    try:
        # 传入 expert_data_plane 真实运行态, 让 Hermes colibri 字段不再用
        # accept_rate 占位 (P0 修复)
        expert_runtime = _expert_data_plane.runtime_snapshot()
        matrix_v2, decision_v2 = get_hermes_router().decide_v2(
            model_name=model_name,
            prompt=prompt,
            cache_hit=False,
            online=True,
            mtp_available=True,
            mtp_accept_rate=tracker.get_accept_rate(str(family_info.get("family") or "generic")),
            route_context=route_context,
            expert_runtime=expert_runtime,
        )
        enabled = decision_v2.grammar_mode != "off" or decision_v2.draft_n_tokens > 0
        disable_reason = ""
        mtp_runtime = _local_mtp_runtime_status(
            model_name=model_name,
            route_mode=str(decision_v2.mode or ""),
            draft_n_tokens=int(decision_v2.draft_n_tokens or 0),
        )
        if decision_v2.fallback_policy == "disable_speculation":
            enabled = False
            disable_reason = "hermes_fallback:disable_speculation"
        elif int(decision_v2.draft_n_tokens or 0) > 0 and not bool(mtp_runtime.get("available")):
            enabled = False
            disable_reason = str(mtp_runtime.get("reason") or "local_mtp_unavailable")
        draft_mode = "grammar_json" if decision_v2.grammar_mode in {"json", "schema"} else ("hybrid" if decision_v2.grammar_mode == "tool_call" else ("mtp" if decision_v2.draft_n_tokens > 0 else "off"))
        if not enabled and str(disable_reason).startswith("local_mtp_unavailable"):
            draft_mode = "off"
        policy = {
            "draft_mode": draft_mode,
            "response_contract": decision_v2.response_contract,
            "grammar_mode": decision_v2.grammar_mode,
            "enabled": enabled,
            "disable_reason": disable_reason,
            "roi": float(decision_v2.speculation_expected_roi or 0.0),
            "json_success_rate": 1.0,
            "source": "hermes_v2",
            "route_mode": decision_v2.mode,
            "reason": f"{decision_v2.reason} | {disable_reason}".strip(" |") if disable_reason else decision_v2.reason,
            "mtp_runtime": mtp_runtime,
        }
        return policy, matrix_v2.to_dict(), decision_v2.to_dict()
    except Exception as exc:
        fallback = _build_draft_mode_policy(body, family_info)
        fallback["source"] = "local_fallback"
        fallback["reason"] = f"hermes_error:{exc}"
        fallback_mode = str((route_context or {}).get("mode_hint") or (route_context or {}).get("mode") or ROUTE_CLOUD_PD)
        fallback_policy = {
            "mode": fallback_mode,
            "pivot_layer": int((route_context or {}).get("P") or 0),
            "reason": str((route_context or {}).get("reason") or fallback["reason"]),
            "response_contract": str(fallback.get("response_contract") or "plain"),
            "grammar_mode": str(fallback.get("grammar_mode") or "off"),
            "fallback_policy": "cloud_only",
            "policy_source": "local_fallback",
        }
        return fallback, None, fallback_policy


def _build_draft_mode_policy(body: dict[str, Any], family_info: dict[str, Any]) -> dict[str, Any]:
    response_contract, grammar_mode = _detect_response_contract(body, family_info)
    draft_mode = "grammar_json" if response_contract == "json" else ("hybrid" if response_contract == "tool" else "mtp")
    bucket = _get_draft_mode_bucket(draft_mode)
    roi = _draft_mode_roi(bucket)
    json_success_rate = _draft_mode_json_success_rate(bucket)
    enabled = True
    disable_reason = ""
    if bucket.get("auto_disabled"):
        enabled = False
        disable_reason = str(bucket.get("disable_reason") or "auto_disabled")
    elif int(bucket.get("speculative_requests", 0)) >= _DRAFT_MODE_MIN_SAMPLES and roi < _DRAFT_MODE_MIN_ROI:
        enabled = False
        disable_reason = f"roi_below_threshold:{roi}"
    elif response_contract == "json" and int(bucket.get("json_successes", 0)) + int(bucket.get("json_failures", 0)) >= _DRAFT_MODE_MIN_SAMPLES and json_success_rate < _DRAFT_MODE_MIN_JSON_SUCCESS:
        enabled = False
        disable_reason = f"json_success_below_threshold:{json_success_rate}"
    return {
        "draft_mode": draft_mode,
        "response_contract": response_contract,
        "grammar_mode": grammar_mode,
        "enabled": enabled,
        "disable_reason": disable_reason,
        "roi": roi,
        "json_success_rate": json_success_rate,
    }


def _record_draft_mode_request(policy: dict[str, Any]) -> None:
    draft_mode = str(policy.get("draft_mode") or "off")
    with _draft_mode_stats_lock:
        bucket = _draft_mode_stats.setdefault(draft_mode, _new_draft_mode_bucket())
        bucket["requests"] += 1
        bucket["last_update_ts"] = time.time()


def _record_draft_mode_outcome(
    policy: dict[str, Any],
    *,
    speculative: bool,
    hit: bool,
    spec_elapsed_ms: float = 0.0,
    local_ttft_ms: float = 0.0,
    cloud_ttft_ms: float = 0.0,
    json_success: Optional[bool] = None,
) -> None:
    draft_mode = str(policy.get("draft_mode") or "off")
    response_contract = str(policy.get("response_contract") or "plain")
    with _draft_mode_stats_lock:
        bucket = _draft_mode_stats.setdefault(draft_mode, _new_draft_mode_bucket())
        bucket["last_update_ts"] = time.time()
        if speculative:
            bucket["speculative_requests"] += 1
            bucket["speculation_cost_ms"] += max(float(spec_elapsed_ms or 0.0), 0.0)
            if hit:
                bucket["hits"] += 1
                bucket["speculation_benefit_ms"] += max(float(cloud_ttft_ms or 0.0) - float(local_ttft_ms or 0.0), 0.0)
            else:
                bucket["misses"] += 1
        if json_success is True:
            bucket["json_successes"] += 1
        elif json_success is False:
            bucket["json_failures"] += 1

        roi = _draft_mode_roi(bucket)
        json_success_rate = _draft_mode_json_success_rate(bucket)
        if bucket["speculative_requests"] >= _DRAFT_MODE_MIN_SAMPLES and roi < _DRAFT_MODE_MIN_ROI:
            bucket["auto_disabled"] = True
            bucket["disable_reason"] = f"roi_below_threshold:{roi}"
        elif response_contract == "json" and (bucket["json_successes"] + bucket["json_failures"]) >= _DRAFT_MODE_MIN_SAMPLES and json_success_rate < _DRAFT_MODE_MIN_JSON_SUCCESS:
            bucket["auto_disabled"] = True
            bucket["disable_reason"] = f"json_success_below_threshold:{json_success_rate}"
    _persist_live_report_snapshots()


def _extract_response_text_from_payload(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    if isinstance(message.get("content"), str):
        return str(message.get("content") or "")
    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
    return str(delta.get("content") or "")


def _resolve_active_model_cfg(model_name: str = ""):
    candidate = str(model_name or _ACTIVE_MODEL or "").strip().lower()
    local_cfg = _resolve_local_runtime_model_cfg()
    local_name = str(getattr(local_cfg, "name", "") or "").strip().lower()
    if local_name:
        if not candidate or candidate == local_name:
            return local_cfg
        if local_name.startswith("gemma4") and candidate in {"gemma4", "g4", "gemma"}:
            return local_cfg
        if local_name.startswith("dsv4") and candidate in {"dsv4", "ds", "deepseek", "v4", "flash"}:
            return local_cfg
        if local_name.startswith("qwen3vl") and candidate in {"qwen3vl", "qwen", "q3"}:
            return local_cfg
    if _ACTIVE_MODEL_CFG is not None and (not candidate or candidate == str(getattr(_ACTIVE_MODEL_CFG, "name", "") or "").lower()):
        return _ACTIVE_MODEL_CFG
    try:
        from app.shared.model_registry import get_model_config
        if candidate:
            return get_model_config(candidate)
    except Exception:
        pass
    return _ACTIVE_MODEL_CFG


def _normalized_messages_for_model(messages: list[dict[str, Any]], model_name: str = "") -> list[dict[str, Any]]:
    normalized = [dict(msg or {}) for msg in (messages or [])]
    cfg = _resolve_active_model_cfg(model_name)
    default_system_prompt = str(getattr(cfg, "default_system_prompt", "") or "").strip() if cfg is not None else ""
    needs_no_think = _should_strip_reasoning_tags(model_name)
    if needs_no_think:
        has_no_think = any("/no_think" in str((msg or {}).get("content") or "") for msg in normalized)
        if not has_no_think:
            injected = False
            for msg in normalized:
                if str((msg or {}).get("role") or "").strip().lower() == "system":
                    msg["content"] = f"/no_think\n{str(msg.get('content') or '').lstrip()}"
                    injected = True
                    break
            if not injected:
                no_think_message = "/no_think"
                if default_system_prompt:
                    no_think_message = f"/no_think\n{default_system_prompt}"
                    default_system_prompt = ""
                normalized = [{"role": "system", "content": no_think_message}, *normalized]
    if not default_system_prompt:
        return normalized
    if any(str((msg or {}).get("role") or "").strip().lower() == "system" for msg in normalized):
        return normalized
    return [{"role": "system", "content": default_system_prompt}, *normalized]


def _should_strip_reasoning_tags(model_name: str = "") -> bool:
    cfg = _resolve_active_model_cfg(model_name)
    return bool(getattr(cfg, "strip_reasoning_tags", False)) if cfg is not None else False


class _ReasoningTagFilter:
    """增量移除 <think>...</think> 段，兼容 chunk 边界切分。"""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self._trim_leading_ws = False

    def _emit(self, text: str) -> str:
        chunk = str(text or "")
        if self._trim_leading_ws:
            chunk = chunk.lstrip()
            if chunk:
                self._trim_leading_ws = False
        return chunk

    def feed(self, text: str) -> str:
        self._buffer += str(text or "")
        out: list[str] = []
        open_keep = len(self._OPEN) - 1
        close_keep = len(self._CLOSE) - 1
        while self._buffer:
            if self._inside:
                close_idx = self._buffer.find(self._CLOSE)
                if close_idx < 0:
                    if len(self._buffer) > close_keep:
                        self._buffer = self._buffer[-close_keep:]
                    break
                self._buffer = self._buffer[close_idx + len(self._CLOSE):]
                self._inside = False
                self._trim_leading_ws = True
                continue

            open_idx = self._buffer.find(self._OPEN)
            if open_idx < 0:
                if len(self._buffer) <= open_keep:
                    break
                emit = self._emit(self._buffer[:-open_keep])
                self._buffer = self._buffer[-open_keep:]
                if emit:
                    out.append(emit)
                continue

            emit = self._emit(self._buffer[:open_idx])
            if emit:
                out.append(emit)
            self._buffer = self._buffer[open_idx + len(self._OPEN):]
            self._inside = True
        return "".join(out)

    def finish(self) -> str:
        if self._inside:
            self._buffer = ""
            return ""
        tail = self._buffer
        self._buffer = ""
        for marker in ("<think", "</think"):
            idx = tail.find(marker)
            if idx >= 0:
                tail = tail[:idx]
        return self._emit(tail)


def _sanitize_output_text_for_model(text: str, *, model_name: str = "") -> str:
    if not _should_strip_reasoning_tags(model_name):
        return str(text or "")
    sanitizer = _ReasoningTagFilter()
    return f"{sanitizer.feed(str(text or ''))}{sanitizer.finish()}"


def _sanitize_payload_content_for_model(payload: dict[str, Any], *, model_name: str = "") -> dict[str, Any]:
    if not _should_strip_reasoning_tags(model_name):
        return payload
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = _sanitize_output_text_for_model(str(message.get("content") or ""), model_name=model_name)
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            delta["content"] = _sanitize_output_text_for_model(str(delta.get("content") or ""), model_name=model_name)
    return payload


def _strip_markdown_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _is_valid_json_text(text: str) -> bool:
    cleaned = _strip_markdown_fence(text)
    if not cleaned:
        return False
    try:
        json.loads(cleaned)
        return True
    except Exception:
        return False


def _estimate_output_tokens(text: str) -> int:
    """估算输出 token 数，优先使用当前 active model tokenizer。"""
    cleaned = str(text or "")
    if not cleaned.strip():
        return 0
    try:
        tokenizer = _load_edge_tokenizer()
        if tokenizer is not None:
            return max(len(tokenizer.encode(cleaned, add_special_tokens=False)), 0)
    except Exception:
        pass
    # fallback：没有 tokenizer 时退回到粗粒度词数估算，至少保持 >0 统计链不断。
    return max(len(cleaned.split()), 1)


def _header_safe(value: Any) -> str:
    """将动态 header 值约束到 latin-1，避免 Starlette 响应头编码失败。"""
    return str(value or "").encode("latin-1", "replace").decode("latin-1")


def _record_request_completion(
    *,
    response_text: str,
    request_started_at: float,
    first_token_ttft_ms: float = 0.0,
    success: bool,
    content_success: Optional[bool] = None,
    path_kind: str = _REQUEST_PATH_COLD,
) -> None:
    total_elapsed_ms = max((time.monotonic() - request_started_at) * 1000, 0.0)
    first_token_ms = max(float(first_token_ttft_ms or 0.0), 0.0)
    decode_elapsed_ms = max(total_elapsed_ms - first_token_ms, 0.0)
    output_tokens = _estimate_output_tokens(response_text)
    resolved_content_success = bool(str(response_text or "").strip()) if content_success is None else bool(content_success)
    _record_stats(
        decode_sample={
            "tokens": output_tokens,
            "elapsed_ms": decode_elapsed_ms,
            "success": success,
            "content_success": resolved_content_success,
            "path_kind": path_kind,
        }
    )


def _inspect_sse_chunk(chunk: bytes) -> tuple[bool, str, bool]:
    try:
        line = chunk.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            return False, "", False
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            return False, "", True
        obj = json.loads(data_str)
        choices = obj.get("choices", [])
        if not choices:
            return False, "", False
        finish_reason = str((choices[0] or {}).get("finish_reason") or "")
        content = str((choices[0].get("delta") or {}).get("content") or "")
        return bool(content), content, finish_reason == "stop"
    except Exception:
        return False, "", False


class _StreamFrontierAdvancer:
    """Share one decode-aware frontier counting policy across streaming branches."""

    def __init__(self, *, expert_session: Optional[Any]) -> None:
        self._expert_session = expert_session
        self._advance_interval = max(int(os.environ.get("EDGE_EXPERT_ADVANCE_TOKEN_INTERVAL", "8") or "8"), 1)
        self._token_count = 0
        self._counted_token_total = 0
        self._response_parts: list[str] = []
        self._tokenizer = None
        if expert_session is not None:
            try:
                self._tokenizer = _load_edge_tokenizer()
            except Exception:
                self._tokenizer = None

    def _count_tokens_total(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, math.ceil(len(text) / 4))

    def observe_text(self, text: str) -> None:
        if not text:
            return
        self._response_parts.append(text)
        if self._expert_session is None:
            return
        counted_total = self._count_tokens_total("".join(self._response_parts))
        delta_tokens = max(0, counted_total - self._counted_token_total)
        self._counted_token_total = counted_total
        self._token_count += delta_tokens
        while self._token_count >= self._advance_interval:
            self._token_count -= self._advance_interval
            try:
                _expert_data_plane.advance_window(self._expert_session)
            except Exception:
                break

    def response_text(self) -> str:
        return "".join(self._response_parts)


async def _instrument_passthrough_stream(
    stream_iter,
    *,
    request_started_at: float,
    draft_policy: dict[str, Any],
    family_info: dict[str, Any],
    route_mode: str,
    transport_runtime: Optional[dict[str, Any]] = None,
    transport_route: Optional[dict[str, Any]] = None,
    expert_session: Optional[Any] = None,
    speculation_info: Optional[dict[str, Any]] = None,
):
    first_token_ttft_ms = 0.0
    path_kind = _classify_request_path_kind(expert_session)
    stream_failed = False
    stream_completed = False
    saw_content = False
    frontier_advancer = _StreamFrontierAdvancer(expert_session=expert_session)

    try:
        if hasattr(stream_iter, "__aiter__"):
            async for chunk in stream_iter:
                has_content, content, terminal = _inspect_sse_chunk(chunk)
                if has_content and first_token_ttft_ms <= 0:
                    first_token_ttft_ms = max((time.monotonic() - request_started_at) * 1000, 0.0)
                    _record_ttft_sample(first_token_ttft_ms, path_kind=path_kind)
                if content:
                    saw_content = True
                    frontier_advancer.observe_text(content)
                if terminal:
                    stream_completed = True
                yield chunk
        else:
            for chunk in stream_iter:
                has_content, content, terminal = _inspect_sse_chunk(chunk)
                if has_content and first_token_ttft_ms <= 0:
                    first_token_ttft_ms = max((time.monotonic() - request_started_at) * 1000, 0.0)
                    _record_ttft_sample(first_token_ttft_ms, path_kind=path_kind)
                if content:
                    saw_content = True
                    frontier_advancer.observe_text(content)
                if terminal:
                    stream_completed = True
                yield chunk
    except Exception:
        stream_failed = True
        raise
    finally:
        response_text = frontier_advancer.response_text()
        execution_success = (not stream_failed) and (stream_completed or saw_content)
        content_success = bool(response_text.strip())
        json_success = None
        if draft_policy["response_contract"] == "json":
            json_success = _is_valid_json_text(response_text)
        speculation_info = speculation_info or {}
        _record_draft_mode_outcome(
            draft_policy,
            speculative=bool(speculation_info.get("speculative", False)),
            hit=bool(speculation_info.get("hit", False)),
            spec_elapsed_ms=float(speculation_info.get("spec_elapsed_ms", 0.0) or 0.0),
            local_ttft_ms=float(speculation_info.get("local_ttft_ms", 0.0) or 0.0),
            cloud_ttft_ms=float(speculation_info.get("cloud_ttft_ms", 0.0) or 0.0),
            json_success=json_success,
        )
        _record_request_completion(
            response_text=response_text,
            request_started_at=request_started_at,
            first_token_ttft_ms=first_token_ttft_ms,
            success=execution_success,
            content_success=content_success,
            path_kind=path_kind,
        )
        _expert_data_plane.complete_request(
            expert_session,
            success=execution_success,
            response_text=response_text,
        )
        _refresh_acceptance_live_reports(
            family_info=family_info,
            route_mode=route_mode,
            transport_runtime=transport_runtime,
            transport_route=transport_route,
        )


def _refresh_acceptance_live_reports(
    *,
    family_info: dict[str, Any],
    route_mode: str,
    transport_runtime: Optional[dict[str, Any]] = None,
    transport_route: Optional[dict[str, Any]] = None,
) -> None:
    _update_heat_snapshot(
        family_info,
        route_mode,
        transport_runtime=transport_runtime,
        transport_route=transport_route,
    )


def _update_route_policy_snapshot(
    feature_schema: dict[str, Any],
    final_policy: dict[str, Any],
    guard_overrides: list[dict[str, Any]],
    hermes_policy: Optional[dict[str, Any]] = None,
    transport_runtime: Optional[dict[str, Any]] = None,
    transport_route: Optional[dict[str, Any]] = None,
) -> None:
    global _last_route_policy_snapshot
    snapshot = route_policy_v2_report_contract()
    snapshot["feature_schema"] = feature_schema
    snapshot["transport_runtime"] = transport_runtime or {}
    snapshot["transport_route"] = transport_route or {}
    snapshot["hermes_policy"] = hermes_policy or {}
    snapshot["final_policy"] = final_policy
    snapshot["guard_overrides"] = guard_overrides
    with _report_snapshot_lock:
        _last_route_policy_snapshot = snapshot
    _persist_live_report_snapshots()


def _frontier_thread_snapshot(
    family_info: dict[str, Any],
    expert_snapshot: dict[str, Any],
    route_mode: str,
) -> dict[str, Any]:
    last_plan = expert_snapshot.get("last_plan") or {}
    frontier_cursor_head = list(expert_snapshot.get("frontier_cursor_head") or [])
    frontier_key = str(last_plan.get("frontier_key") or "")
    if not frontier_key and frontier_cursor_head:
        frontier_key = str((frontier_cursor_head[0] or {}).get("key") or "")
    frontier_cursor = None
    for item in frontier_cursor_head:
        if str((item or {}).get("key") or "") == frontier_key:
            frontier_cursor = int((item or {}).get("cursor") or 0)
            break
    if frontier_cursor is None and frontier_cursor_head:
        frontier_cursor = int((frontier_cursor_head[0] or {}).get("cursor") or 0)
    return {
        "frontier_id": str(family_info.get("frontier_id") or ""),
        "request_uid": str(family_info.get("request_uid") or ""),
        "route_family": str(family_info.get("family") or "generic"),
        "prompt_hash": str(family_info.get("prompt_hash") or ""),
        "route_mode": str(route_mode or ""),
        "frontier_key": frontier_key,
        "execution_success": bool(last_plan.get("execution_success", last_plan.get("success"))),
        "content_success": bool(last_plan.get("content_success", False)),
        "layer_cursor": int(last_plan.get("layer_cursor") or 0),
        "next_layer_cursor": int(last_plan.get("next_layer_cursor") or 0),
        "current_layer_ids": list(last_plan.get("current_layer_ids") or []),
        "next_layer_ids": list(last_plan.get("next_layer_ids") or []),
        "frontier_advanced_to": int(last_plan.get("frontier_advanced_to") or 0),
        "frontier_cursor": int(frontier_cursor or 0),
    }


def _build_routing_summary(
    *,
    transport_route: Optional[dict[str, Any]],
    route_mode: str,
) -> dict[str, Any]:
    actual_route = str((transport_route or {}).get("mode") or route_mode or "unknown")
    degrade_suggested = bool((transport_route or {}).get("degrade_suggested"))
    degrade_target_mode = str((transport_route or {}).get("degrade_target_mode") or "")
    mode_switch_reason = str((transport_route or {}).get("mode_switch_reason") or (transport_route or {}).get("reason") or "")
    memory_pressure = str((transport_route or {}).get("memory_pressure") or "unknown")
    admissible_degrade = bool(
        degrade_suggested
        and actual_route == degrade_target_mode
        and actual_route in {ROUTE_LAYER_SPLIT_PD, ROUTE_CLOUD_PD, ROUTE_CLOUD_FALLBACK}
    )
    if actual_route in {ROUTE_LOCAL_FULL, ROUTE_LAYER_SPLIT_PD, ROUTE_CLOUD_PD, ROUTE_CLOUD_FALLBACK}:
        routing_status = "ADMISSIBLE_DEGRADE" if admissible_degrade else "PASS"
    else:
        routing_status = "FAIL"
    return {
        "routing_status": routing_status,
        "admissible_degrade": admissible_degrade,
        "transport_route_mode": actual_route,
        "degrade_target_mode": degrade_target_mode,
        "mode_switch_reason": mode_switch_reason,
        "memory_pressure": memory_pressure,
    }


def _build_path_bottlenecks(
    *,
    candidate_snapshot: dict[str, Any],
    cold_path_status: str,
    warm_hot_path_status: str,
    cold_failed_gate_summaries: list[dict[str, Any]],
    failed_gate_summaries: list[dict[str, Any]],
    cold_ttft_ms_avg: float,
    warm_hot_ttft_ms_avg: float,
    warm_hot_decode_tps_avg: float,
    warm_hot_evaluated_requests: int,
    dense_non_expert_baseline: bool,
) -> dict[str, Any]:
    expert_snapshot = dict(candidate_snapshot.get("expert_data_plane") or {})
    last_plan = dict(expert_snapshot.get("last_plan") or {})
    frontier_thread = dict(candidate_snapshot.get("frontier_thread") or {})
    frontier_cursor_head = list(expert_snapshot.get("frontier_cursor_head") or [])
    frontier_head = dict(frontier_cursor_head[0] or {}) if frontier_cursor_head else {}

    cold_primary = dict(cold_failed_gate_summaries[0]) if cold_failed_gate_summaries else {}
    if cold_primary:
        cold_bottleneck = {
            "status": cold_path_status,
            "primary_gate": str(cold_primary.get("name") or "unknown"),
            "summary": str(cold_primary.get("summary") or ""),
            "actual": cold_primary.get("actual"),
            "threshold": cold_primary.get("threshold"),
            "ttft_ms_avg": round(float(cold_ttft_ms_avg or 0.0), 1),
        }
    else:
        cold_bottleneck = {
            "status": cold_path_status,
            "primary_gate": "none",
            "summary": "cold_path has no active bottleneck",
            "actual": round(float(cold_ttft_ms_avg or 0.0), 1),
            "threshold": _PRODUCTION_GATE_TTFT_MS,
            "ttft_ms_avg": round(float(cold_ttft_ms_avg or 0.0), 1),
        }

    warm_hot_failures = [dict(item) for item in failed_gate_summaries if str(item.get("name") or "").startswith("warm_hot_path.")]
    warm_hot_primary = dict(warm_hot_failures[0]) if warm_hot_failures else {}
    warm_hot_plan_enabled = bool(last_plan.get("enabled"))
    warm_hot_plan_reason = str(last_plan.get("reason") or "")
    if dense_non_expert_baseline:
        warm_hot_bottleneck = {
            "status": warm_hot_path_status,
            "primary_gate": "warm_hot_path.not_applicable",
            "summary": "warm_hot_path not applicable for dense/non-expert baseline; expert catalog is intentionally absent",
            "actual": {
                "expert_plan_enabled": warm_hot_plan_enabled,
                "expert_plan_reason": warm_hot_plan_reason,
                "frontier_advanced_to": int(frontier_thread.get("frontier_advanced_to") or 0),
                "frontier_cursor_count": int(expert_snapshot.get("frontier_cursor_count", 0) or 0),
            },
            "threshold": None,
        }
    elif warm_hot_evaluated_requests <= 0:
        if not warm_hot_plan_enabled:
            summary = f"warm_hot_path unreachable: expert_data_plane last_plan disabled ({warm_hot_plan_reason or 'unknown'})"
            primary_gate = "warm_hot_path.unreachable"
        elif int(frontier_thread.get("frontier_advanced_to") or 0) <= 0 and int(expert_snapshot.get("frontier_cursor_count", 0) or 0) <= 0:
            summary = "warm_hot_path unreachable: no frontier advance observed, reusable warm state not established"
            primary_gate = "warm_hot_path.frontier_not_advanced"
        else:
            summary = "warm_hot_path unreachable: no cache/prefetch hit observed, requests stayed on cold classification"
            primary_gate = "warm_hot_path.no_warm_signal"
        warm_hot_bottleneck = {
            "status": warm_hot_path_status,
            "primary_gate": primary_gate,
            "summary": summary,
            "actual": {
                "expert_plan_enabled": warm_hot_plan_enabled,
                "expert_plan_reason": warm_hot_plan_reason,
                "frontier_advanced_to": int(frontier_thread.get("frontier_advanced_to") or 0),
                "frontier_cursor_count": int(expert_snapshot.get("frontier_cursor_count", 0) or 0),
            },
            "threshold": "warm_hot_path_evaluated_requests>0",
        }
    elif warm_hot_primary:
        warm_hot_bottleneck = {
            "status": warm_hot_path_status,
            "primary_gate": str(warm_hot_primary.get("name") or "unknown"),
            "summary": str(warm_hot_primary.get("summary") or ""),
            "actual": warm_hot_primary.get("actual"),
            "threshold": warm_hot_primary.get("threshold"),
        }
    else:
        warm_hot_bottleneck = {
            "status": warm_hot_path_status,
            "primary_gate": "none",
            "summary": "warm_hot_path has no active bottleneck",
            "actual": {
                "ttft_ms_avg": round(float(warm_hot_ttft_ms_avg or 0.0), 1),
                "decode_tps_avg": round(float(warm_hot_decode_tps_avg or 0.0), 1),
            },
            "threshold": None,
        }

    warm_hot_bottleneck["diagnostic_signals"] = {
        "expert_plan_enabled": warm_hot_plan_enabled,
        "expert_plan_reason": warm_hot_plan_reason,
        "frontier_id": str(frontier_thread.get("frontier_id") or ""),
        "frontier_key": str(frontier_thread.get("frontier_key") or ""),
        "frontier_advanced_to": int(frontier_thread.get("frontier_advanced_to") or 0),
        "frontier_cursor_count": int(expert_snapshot.get("frontier_cursor_count", 0) or 0),
        "frontier_cursor_head_key": str(frontier_head.get("key") or ""),
        "warm_hot_evaluated_requests": int(warm_hot_evaluated_requests or 0),
        "warm_hot_ttft_ms_avg": round(float(warm_hot_ttft_ms_avg or 0.0), 1),
        "warm_hot_decode_tps_avg": round(float(warm_hot_decode_tps_avg or 0.0), 1),
    }
    return {
        "cold_path": cold_bottleneck,
        "warm_hot_path": warm_hot_bottleneck,
    }


def _update_heat_snapshot(
    family_info: dict[str, Any],
    route_mode: str,
    transport_runtime: Optional[dict[str, Any]] = None,
    transport_route: Optional[dict[str, Any]] = None,
) -> None:
    global _last_heat_snapshot, _last_single_node_candidate_matrix
    stats = _get_stats()
    expert_snapshot = _expert_data_plane.runtime_snapshot()
    last_plan = expert_snapshot.get("last_plan") or {}
    canonical_transport_mode = str((transport_route or {}).get("mode") or route_mode or "")
    frontier_thread = _frontier_thread_snapshot(
        family_info=family_info,
        expert_snapshot=expert_snapshot,
        route_mode=route_mode,
    )
    snapshot = route_heat_snapshot_report_contract()
    snapshot.update({
        "expert_hit_rate_ema": round(float(expert_snapshot.get("cache_hit_rate", 0.0)), 3),
        "hot_expert_ratio": round(
            float(expert_snapshot.get("pinned_count", 0)) / max(int(expert_snapshot.get("expert_count", 0) or 0), 1),
            3,
        ) if expert_snapshot.get("expert_count") else 0.0,
        "warm_pin_gb": round(float(expert_snapshot.get("pinned_gb", 0.0)), 3),
        "predicted_bytes_to_read_mb": round(float(last_plan.get("predicted_bytes_to_read_mb", 0.0)), 3),
        "predicted_cold_bytes_mb": round(float(last_plan.get("predicted_cold_bytes_mb", 0.0)), 3),
        "prefetch_hit_rate_ema": round(float(expert_snapshot.get("prefetch_hit_rate", 0.0)), 3),
        "transport_runtime": transport_runtime or {},
        "transport_route": transport_route or {},
        "frontier_thread": frontier_thread,
        "expert_data_plane": expert_snapshot,
        "storage_topology": {
            "route_family": str(family_info.get("family") or "generic"),
            # route_mode 在 heat report 中固定锚定 transport snapshot，
            # 避免 storage_topology 和 transport_route 出现同名异义漂移。
            "route_mode": canonical_transport_mode,
            "final_route_mode": route_mode,
            "route_mode_consistent": canonical_transport_mode == str(route_mode or canonical_transport_mode),
            "memory_pressure": str((transport_route or {}).get("memory_pressure") or "unknown"),
            "moe_streaming_admissible": bool((transport_route or {}).get("moe_streaming_admissible")),
            "degrade_suggested": bool((transport_route or {}).get("degrade_suggested")),
            "degrade_target_mode": str((transport_route or {}).get("degrade_target_mode") or ""),
            "mode_switch_reason": str((transport_route or {}).get("mode_switch_reason") or ""),
            "sticky_active": bool((transport_route or {}).get("sticky_active")),
            "frontier_id": str(frontier_thread.get("frontier_id") or ""),
            "frontier_key": str(frontier_thread.get("frontier_key") or ""),
            "execution_success": bool(frontier_thread.get("execution_success")),
            "content_success": bool(frontier_thread.get("content_success")),
            "warmup_cache_size": len(_warmup_state),
            "resident_expert_count": int(expert_snapshot.get("resident_count", 0) or 0),
            "pinned_expert_count": int(expert_snapshot.get("pinned_count", 0) or 0),
            "prefetch_inflight": int(expert_snapshot.get("prefetch_inflight", 0) or 0),
        },
    })
    candidate_matrix = single_node_candidate_matrix_report_contract()
    production_target = _resolve_production_target()
    candidate_model_name = str(production_target.get("name") or _ACTIVE_MODEL)
    candidate_display_name = str(production_target.get("display_name") or candidate_model_name)
    candidate_matrix["hardware_profile"] = {
        "active_model": _ACTIVE_MODEL,
        "engine": "edge_first_proxy",
        "route_mode": route_mode,
        "memory_pressure": str((transport_route or {}).get("memory_pressure") or "unknown"),
        "moe_streaming_admissible": bool((transport_route or {}).get("moe_streaming_admissible")),
        "degrade_target_mode": str((transport_route or {}).get("degrade_target_mode") or ""),
        "mode_switch_reason": str((transport_route or {}).get("mode_switch_reason") or ""),
        "sticky_active": bool((transport_route or {}).get("sticky_active")),
        "frontier_id": str(frontier_thread.get("frontier_id") or ""),
        "frontier_key": str(frontier_thread.get("frontier_key") or ""),
        "execution_success": bool(frontier_thread.get("execution_success")),
        "content_success": bool(frontier_thread.get("content_success")),
    }
    candidate_matrix["production_target"] = production_target
    candidate_matrix["transport_runtime"] = transport_runtime or {}
    candidate_matrix["transport_route"] = transport_route or {}
    candidate_matrix["routing_summary"] = _build_routing_summary(
        transport_route=transport_route,
        route_mode=route_mode,
    )
    candidate_matrix["frontier_thread"] = frontier_thread
    candidate_matrix["expert_data_plane"] = expert_snapshot
    candidate_matrix["candidates"] = [
        {
            "model": candidate_model_name,
            "priority": str(production_target.get("priority") or "P0"),
            "status": "PARTIAL",
            "ttft_ms": stats.get("ttft_ms_avg", 0.0),
            "decode_tps": 0.0,
            "stable_rounds_passed": 0,
            "notes": (
                f"target={candidate_display_name}, family={str(family_info.get('family') or 'generic')}, "
                f"frontier={str(frontier_thread.get('frontier_id') or frontier_thread.get('frontier_key') or 'none')}, "
                f"exec_success={1 if frontier_thread.get('execution_success') else 0}, "
                f"content_success={1 if frontier_thread.get('content_success') else 0}, "
                f"memory_pressure={str((transport_route or {}).get('memory_pressure') or 'unknown')}, "
                f"moe_streaming_admissible={1 if (transport_route or {}).get('moe_streaming_admissible') else 0}, "
                f"degrade_target={str((transport_route or {}).get('degrade_target_mode') or 'none')}, "
                f"switch_reason={str((transport_route or {}).get('mode_switch_reason') or 'none')}, "
                f"experts={len(last_plan.get('current_keys') or [])}, "
                f"prefetch={len(last_plan.get('next_keys') or [])}, "
                f"route_swap_pct={expert_snapshot.get('route_swap_pct', 0.0)}, "
                f"warm_start={expert_snapshot.get('warm_start_loaded', 0)}, "
                f"catalog={expert_snapshot.get('catalog_source', 'none')}"
            ),
        }
    ]
    with _report_snapshot_lock:
        _last_heat_snapshot = snapshot
        _last_single_node_candidate_matrix = candidate_matrix
    _persist_live_report_snapshots()


def _draft_mode_acceptance_snapshot() -> dict[str, Any]:
    snapshot = draft_mode_acceptance_report_contract()
    modes = _snapshot_draft_mode_stats()
    if not modes:
        return snapshot
    preferred_mode = "grammar_json" if "grammar_json" in modes else sorted(modes.keys())[0]
    bucket = modes[preferred_mode]
    snapshot.update({
        "draft_mode": preferred_mode,
        "response_contract": "json" if preferred_mode == "grammar_json" else "plain",
        "accept_rate_ema": round(get_acceptance_tracker().get_accept_rate(), 3),
        "grammar_accept_rate_ema": round(bucket.get("hits", 0) / max(bucket.get("speculative_requests", 1), 1), 3) if bucket.get("speculative_requests", 0) else 0.0,
        "json_success_rate": bucket.get("json_success_rate", 1.0),
        "recent_speculation_roi": bucket.get("roi", 0.0),
        "auto_disabled": bool(bucket.get("auto_disabled", False)),
        "disable_reason": str(bucket.get("disable_reason") or ""),
        "modes": modes,
    })
    return snapshot


def _production_evidence_chain_ready() -> bool:
    required_files = [
        "route_policy_v2.live.json",
        "route_heat_snapshot.live.json",
        "draft_mode_acceptance.live.json",
    ]
    output_dir = _ensure_report_output_dir()
    for filename in required_files:
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path) or os.path.getsize(path) <= 2:
            return False
    return True


def _select_roi_mode_snapshot(modes: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not modes:
        return "off", {}
    ranked = sorted(
        modes.items(),
        key=lambda item: (
            int(item[1].get("speculative_requests", 0)),
            int(item[1].get("requests", 0)),
            item[0] != "off",
        ),
        reverse=True,
    )
    return ranked[0]


def _build_production_gate_snapshot(candidate_snapshot: dict[str, Any]) -> dict[str, Any]:
    stats = _get_stats()
    modes = _snapshot_draft_mode_stats()
    draft_snapshot = _draft_mode_acceptance_snapshot()
    production_target = dict(candidate_snapshot.get("production_target") or _resolve_production_target())
    candidate_snapshot["production_target"] = production_target
    candidate_snapshot["hardware_profile"] = dict(candidate_snapshot.get("hardware_profile") or {
        "active_model": _ACTIVE_MODEL,
        "engine": "edge_first_proxy",
    })
    routing_summary = _build_routing_summary(
        transport_route=dict(candidate_snapshot.get("transport_route") or {}),
        route_mode=str((candidate_snapshot.get("hardware_profile") or {}).get("route_mode") or ""),
    )
    candidate_snapshot["routing_summary"] = routing_summary
    if not candidate_snapshot.get("candidates"):
        candidate_snapshot["candidates"] = [{
            "model": str(production_target.get("name") or _ACTIVE_MODEL),
            "priority": str(production_target.get("priority") or "P0"),
            "status": "PARTIAL",
            "ttft_ms": 0.0,
            "decode_tps": 0.0,
            "stable_rounds_passed": 0,
            "notes": f"target={str(production_target.get('display_name') or production_target.get('name') or _ACTIVE_MODEL)}",
        }]
    roi_mode, roi_bucket = _select_roi_mode_snapshot(modes)
    roi_samples = int(roi_bucket.get("speculative_requests", 0) or 0)
    roi_value = float(roi_bucket.get("roi", draft_snapshot.get("recent_speculation_roi", 0.0)) or 0.0)
    grammar_bucket = modes.get("grammar_json", {})
    grammar_samples = int(grammar_bucket.get("json_successes", 0) or 0) + int(grammar_bucket.get("json_failures", 0) or 0)
    grammar_json_success = float(grammar_bucket.get("json_success_rate", 0.0) or 0.0) if grammar_bucket else 0.0
    evidence_chain_ready = _production_evidence_chain_ready()
    completed_requests = int(stats.get("completed_requests", 0) or 0)
    failed_requests = int(stats.get("failed_requests", 0) or 0)
    stable_rounds_passed = int(stats.get("stable_consecutive_successes", 0) or 0)
    ttft_ms_avg = float(stats.get("ttft_ms_avg", 0.0) or 0.0)
    decode_tps_avg = float(stats.get("decode_tps_avg", 0.0) or 0.0)
    execution_success_rate = float(stats.get("execution_success_rate", 0.0) or 0.0)
    content_success_rate = float(stats.get("content_success_rate", 0.0) or 0.0)
    transport_route = dict(candidate_snapshot.get("transport_route") or {})
    expert_snapshot = dict(candidate_snapshot.get("expert_data_plane") or {})
    last_plan = dict(expert_snapshot.get("last_plan") or {})
    dense_non_expert_baseline = bool(
        not bool(transport_route.get("moe_candidate"))
        and not bool(last_plan.get("enabled"))
        and str(last_plan.get("reason") or "") in {"no_expert_catalog", "expert_streaming_disabled", "fit_memory_bypass"}
    )
    empty_output_requests = int(stats.get("empty_output_requests", 0) or 0)
    content_completed_requests = int(stats.get("content_completed_requests", 0) or 0)
    path_metrics = dict(stats.get("path_metrics") or {})
    cold_stats = dict(path_metrics.get(_REQUEST_PATH_COLD) or {})
    warm_hot_stats = dict(path_metrics.get(_REQUEST_PATH_WARM_HOT) or {})
    cold_completed_requests = int(cold_stats.get("completed_requests", 0) or 0)
    cold_failed_requests = int(cold_stats.get("failed_requests", 0) or 0)
    cold_evaluated_requests = cold_completed_requests + cold_failed_requests
    cold_ttft_ms_avg = float(cold_stats.get("ttft_ms_avg", 0.0) or 0.0)
    cold_execution_success_rate = float(cold_stats.get("execution_success_rate", 0.0) or 0.0)
    cold_content_success_rate = float(cold_stats.get("content_success_rate", 0.0) or 0.0)
    cold_empty_output_requests = int(cold_stats.get("empty_output_requests", 0) or 0)
    cold_content_completed_requests = int(cold_stats.get("content_completed_requests", 0) or 0)
    warm_hot_completed_requests = int(warm_hot_stats.get("completed_requests", 0) or 0)
    warm_hot_failed_requests = int(warm_hot_stats.get("failed_requests", 0) or 0)
    warm_hot_evaluated_requests = warm_hot_completed_requests + warm_hot_failed_requests
    warm_hot_ttft_ms_avg = float(warm_hot_stats.get("ttft_ms_avg", 0.0) or 0.0)
    warm_hot_decode_tps_avg = float(warm_hot_stats.get("decode_tps_avg", 0.0) or 0.0)
    warm_hot_stable_rounds = int(warm_hot_stats.get("stable_consecutive_successes", 0) or 0)
    warm_hot_execution_success_rate = float(warm_hot_stats.get("execution_success_rate", 0.0) or 0.0)
    warm_hot_content_success_rate = float(warm_hot_stats.get("content_success_rate", 0.0) or 0.0)
    warm_hot_empty_output_requests = int(warm_hot_stats.get("empty_output_requests", 0) or 0)
    warm_hot_content_completed_requests = int(warm_hot_stats.get("content_completed_requests", 0) or 0)

    cold_path_gates = {
        "ttft_le_300ms": {
            "pass": cold_completed_requests > 0 and cold_ttft_ms_avg <= _PRODUCTION_GATE_TTFT_MS,
            "actual": cold_ttft_ms_avg,
            "threshold": _PRODUCTION_GATE_TTFT_MS,
            "completed_requests": cold_completed_requests,
            "evaluated_requests": cold_evaluated_requests,
        },
        "execution_success_rate": {
            "pass": cold_completed_requests > 0 and cold_execution_success_rate >= _PRODUCTION_GATE_EXECUTION_SUCCESS_RATE,
            "actual": cold_execution_success_rate,
            "threshold": _PRODUCTION_GATE_EXECUTION_SUCCESS_RATE,
            "completed_requests": cold_completed_requests,
            "failed_requests": cold_failed_requests,
        },
        "content_success_rate": {
            "pass": cold_completed_requests > 0 and cold_content_success_rate >= _PRODUCTION_GATE_CONTENT_SUCCESS_RATE,
            "actual": cold_content_success_rate,
            "threshold": _PRODUCTION_GATE_CONTENT_SUCCESS_RATE,
            "content_completed_requests": cold_content_completed_requests,
            "empty_output_requests": cold_empty_output_requests,
        },
    }
    warm_hot_path_gates = {
        "ttft_le_300ms": {
            "pass": warm_hot_completed_requests > 0 and warm_hot_ttft_ms_avg <= _PRODUCTION_GATE_TTFT_MS,
            "actual": warm_hot_ttft_ms_avg,
            "threshold": _PRODUCTION_GATE_TTFT_MS,
            "completed_requests": warm_hot_completed_requests,
            "evaluated_requests": warm_hot_evaluated_requests,
        },
        "decode_ge_30_tps": {
            "pass": warm_hot_completed_requests > 0 and warm_hot_decode_tps_avg >= _PRODUCTION_GATE_DECODE_TPS,
            "actual": warm_hot_decode_tps_avg,
            "threshold": _PRODUCTION_GATE_DECODE_TPS,
            "completed_requests": warm_hot_completed_requests,
        },
        "stable_rounds": {
            "pass": warm_hot_stable_rounds >= _PRODUCTION_GATE_STABLE_ROUNDS and warm_hot_failed_requests == 0,
            "actual": warm_hot_stable_rounds,
            "threshold": _PRODUCTION_GATE_STABLE_ROUNDS,
            "failed_requests": warm_hot_failed_requests,
        },
        "execution_success_rate": {
            "pass": warm_hot_completed_requests > 0 and warm_hot_execution_success_rate >= _PRODUCTION_GATE_EXECUTION_SUCCESS_RATE,
            "actual": warm_hot_execution_success_rate,
            "threshold": _PRODUCTION_GATE_EXECUTION_SUCCESS_RATE,
            "completed_requests": warm_hot_completed_requests,
            "failed_requests": warm_hot_failed_requests,
        },
        "content_success_rate": {
            "pass": warm_hot_completed_requests > 0 and warm_hot_content_success_rate >= _PRODUCTION_GATE_CONTENT_SUCCESS_RATE,
            "actual": warm_hot_content_success_rate,
            "threshold": _PRODUCTION_GATE_CONTENT_SUCCESS_RATE,
            "content_completed_requests": warm_hot_content_completed_requests,
            "empty_output_requests": warm_hot_empty_output_requests,
        },
    }
    shared_gates = {
        "speculation_roi_positive": {
            "pass": roi_samples >= _DRAFT_MODE_MIN_SAMPLES and roi_value > _PRODUCTION_GATE_MIN_ROI,
            "actual": roi_value,
            "threshold": _PRODUCTION_GATE_MIN_ROI,
            "samples": roi_samples,
            "mode": roi_mode,
        },
        "json_grammar_success": {
            "pass": grammar_samples >= _DRAFT_MODE_MIN_SAMPLES and grammar_json_success >= _PRODUCTION_GATE_MIN_JSON_SUCCESS,
            "actual": grammar_json_success,
            "threshold": _PRODUCTION_GATE_MIN_JSON_SUCCESS,
            "samples": grammar_samples,
        },
        "json_evidence_chain": {
            "pass": evidence_chain_ready,
            "actual": evidence_chain_ready,
            "threshold": True,
        },
    }
    if dense_non_expert_baseline:
        for payload in warm_hot_path_gates.values():
            payload["pass"] = True
            payload["applicable"] = False
            payload["skip_reason"] = "dense_non_expert_baseline"
        for gate_name in ("speculation_roi_positive", "json_grammar_success"):
            payload = shared_gates[gate_name]
            payload["pass"] = True
            payload["applicable"] = False
            payload["skip_reason"] = "dense_non_expert_baseline"

    def _scope_failed_gate_names(scope: str, gates: dict[str, dict[str, Any]]) -> list[str]:
        return [f"{scope}.{name}" for name, payload in gates.items() if not bool(payload.get("pass"))]

    def _append_scope_failures(
        *,
        scope: str,
        gates: dict[str, dict[str, Any]],
        summaries: list[dict[str, Any]],
    ) -> None:
        for gate_name, gate_payload in gates.items():
            if bool(gate_payload.get("pass")):
                continue
            if scope == _REQUEST_PATH_COLD and gate_name == "execution_success_rate":
                summary = (
                    f"cold_path execution_success_rate={cold_execution_success_rate:.3f} < "
                    f"{_PRODUCTION_GATE_EXECUTION_SUCCESS_RATE:.3f} "
                    f"(completed={cold_completed_requests}, failed={cold_failed_requests})"
                )
            elif scope == _REQUEST_PATH_COLD and gate_name == "content_success_rate":
                summary = (
                    f"cold_path content_success_rate={cold_content_success_rate:.3f} < "
                    f"{_PRODUCTION_GATE_CONTENT_SUCCESS_RATE:.3f} "
                    f"(content_completed={cold_content_completed_requests}, empty_output={cold_empty_output_requests})"
                )
            elif scope == _REQUEST_PATH_COLD and gate_name == "ttft_le_300ms":
                summary = (
                    f"cold_path ttft_ms_avg={cold_ttft_ms_avg:.1f} <= {_PRODUCTION_GATE_TTFT_MS:.1f} "
                    f"(completed_requests={cold_completed_requests})"
                )
            elif scope == _REQUEST_PATH_WARM_HOT and gate_name == "execution_success_rate":
                summary = (
                    f"warm_hot_path execution_success_rate={warm_hot_execution_success_rate:.3f} < "
                    f"{_PRODUCTION_GATE_EXECUTION_SUCCESS_RATE:.3f} "
                    f"(completed={warm_hot_completed_requests}, failed={warm_hot_failed_requests})"
                )
            elif scope == _REQUEST_PATH_WARM_HOT and gate_name == "content_success_rate":
                summary = (
                    f"warm_hot_path content_success_rate={warm_hot_content_success_rate:.3f} < "
                    f"{_PRODUCTION_GATE_CONTENT_SUCCESS_RATE:.3f} "
                    f"(content_completed={warm_hot_content_completed_requests}, empty_output={warm_hot_empty_output_requests})"
                )
            elif scope == _REQUEST_PATH_WARM_HOT and gate_name == "stable_rounds":
                summary = (
                    f"warm_hot_path stable_rounds={warm_hot_stable_rounds} < {_PRODUCTION_GATE_STABLE_ROUNDS} "
                    f"(failed_requests={warm_hot_failed_requests})"
                )
            elif scope == _REQUEST_PATH_WARM_HOT and gate_name == "ttft_le_300ms":
                summary = (
                    f"warm_hot_path ttft_ms_avg={warm_hot_ttft_ms_avg:.1f} <= {_PRODUCTION_GATE_TTFT_MS:.1f} "
                    f"(completed_requests={warm_hot_completed_requests})"
                )
            elif scope == _REQUEST_PATH_WARM_HOT and gate_name == "decode_ge_30_tps":
                summary = (
                    f"warm_hot_path decode_tps_avg={warm_hot_decode_tps_avg:.1f} >= {_PRODUCTION_GATE_DECODE_TPS:.1f} "
                    f"(completed_requests={warm_hot_completed_requests})"
                )
            elif scope == "shared" and gate_name == "speculation_roi_positive":
                summary = (
                    f"speculation_roi={roi_value:.3f} with samples={roi_samples} "
                    f"(need samples>={_DRAFT_MODE_MIN_SAMPLES} and roi>{_PRODUCTION_GATE_MIN_ROI:.3f})"
                )
            elif scope == "shared" and gate_name == "json_grammar_success":
                summary = (
                    f"json_grammar_success_rate={grammar_json_success:.3f} with samples={grammar_samples} "
                    f"(need samples>={_DRAFT_MODE_MIN_SAMPLES} and success>={_PRODUCTION_GATE_MIN_JSON_SUCCESS:.3f})"
                )
            elif scope == "shared" and gate_name == "json_evidence_chain":
                summary = "json_evidence_chain missing required live reports"
            else:
                actual = gate_payload.get("actual")
                threshold = gate_payload.get("threshold")
                summary = f"{scope}.{gate_name}: actual={actual}, threshold={threshold}"
            summaries.append({
                "name": f"{scope}.{gate_name}",
                "summary": summary,
                "actual": gate_payload.get("actual"),
                "threshold": gate_payload.get("threshold"),
            })

    def _scope_status(
        *,
        gates: dict[str, dict[str, Any]],
        evaluated_requests: int,
        requires_requests: bool,
    ) -> str:
        if requires_requests and evaluated_requests <= 0:
            return "NOT_EVALUATED"
        return "PASS" if all(bool(payload.get("pass")) for payload in gates.values()) else "FAIL"

    blocking_scope = ["shared"] if dense_non_expert_baseline else ["warm_hot_path", "shared"]
    production_gates = {
        _REQUEST_PATH_COLD: cold_path_gates,
        _REQUEST_PATH_WARM_HOT: warm_hot_path_gates,
        "shared": shared_gates,
        "blocking_scope": list(blocking_scope),
    }
    blocking_failed_gate_names = _scope_failed_gate_names(_REQUEST_PATH_WARM_HOT, warm_hot_path_gates) + _scope_failed_gate_names("shared", shared_gates)
    cold_failed_gate_names = _scope_failed_gate_names(_REQUEST_PATH_COLD, cold_path_gates)
    failed_gate_summaries: list[dict[str, Any]] = []
    cold_failed_gate_summaries: list[dict[str, Any]] = []
    _append_scope_failures(scope=_REQUEST_PATH_WARM_HOT, gates=warm_hot_path_gates, summaries=failed_gate_summaries)
    _append_scope_failures(scope="shared", gates=shared_gates, summaries=failed_gate_summaries)
    _append_scope_failures(scope=_REQUEST_PATH_COLD, gates=cold_path_gates, summaries=cold_failed_gate_summaries)
    cold_path_status = _scope_status(gates=cold_path_gates, evaluated_requests=cold_evaluated_requests, requires_requests=True)
    warm_hot_path_status = "NOT_APPLICABLE" if dense_non_expert_baseline else _scope_status(
        gates=warm_hot_path_gates,
        evaluated_requests=warm_hot_evaluated_requests,
        requires_requests=True,
    )
    shared_status = _scope_status(gates=shared_gates, evaluated_requests=1, requires_requests=False)
    overall_passed = cold_path_status == "PASS" and shared_status == "PASS" and (
        dense_non_expert_baseline or warm_hot_path_status == "PASS"
    )
    path_bottlenecks = _build_path_bottlenecks(
        candidate_snapshot=candidate_snapshot,
        cold_path_status=cold_path_status,
        warm_hot_path_status=warm_hot_path_status,
        cold_failed_gate_summaries=cold_failed_gate_summaries,
        failed_gate_summaries=failed_gate_summaries,
        cold_ttft_ms_avg=cold_ttft_ms_avg,
        warm_hot_ttft_ms_avg=warm_hot_ttft_ms_avg,
        warm_hot_decode_tps_avg=warm_hot_decode_tps_avg,
        warm_hot_evaluated_requests=warm_hot_evaluated_requests,
        dense_non_expert_baseline=dense_non_expert_baseline,
    )

    candidate_entry = dict((candidate_snapshot.get("candidates") or [{}])[0])
    candidate_entry["model"] = str(candidate_entry.get("model") or production_target.get("name") or _ACTIVE_MODEL)
    candidate_entry["priority"] = str(candidate_entry.get("priority") or production_target.get("priority") or "P0")
    if not str(candidate_entry.get("notes") or "").strip():
        candidate_entry["notes"] = f"target={str(production_target.get('display_name') or candidate_entry['model'])}"
    candidate_entry["ttft_ms"] = warm_hot_ttft_ms_avg if warm_hot_evaluated_requests > 0 else ttft_ms_avg
    candidate_entry["decode_tps"] = warm_hot_decode_tps_avg if warm_hot_evaluated_requests > 0 else decode_tps_avg
    candidate_entry["stable_rounds_passed"] = warm_hot_stable_rounds if warm_hot_evaluated_requests > 0 else stable_rounds_passed
    candidate_entry["status"] = "PASS" if overall_passed else "FAIL"
    candidate_entry["notes"] = (
        f"{candidate_entry.get('notes', '')}; completed={completed_requests}; failed={failed_requests}; "
        f"warm_hot_ttft={warm_hot_ttft_ms_avg}; cold_ttft={cold_ttft_ms_avg}; "
        f"warm_hot_decode={warm_hot_decode_tps_avg}; warm_hot_exec_rate={warm_hot_execution_success_rate}; "
        f"cold_exec_rate={cold_execution_success_rate}; roi_mode={roi_mode}; grammar_samples={grammar_samples}; "
        f"dense_non_expert_baseline={1 if dense_non_expert_baseline else 0}"
    ).strip("; ")
    candidate_snapshot["candidates"] = [candidate_entry]
    candidate_snapshot["acceptance_summary"] = {
        "blocking_scope": list(blocking_scope),
        "dense_non_expert_baseline": dense_non_expert_baseline,
        "ttft_ms_avg": ttft_ms_avg,
        "decode_tps_avg": decode_tps_avg,
        "stable_rounds_passed": stable_rounds_passed,
        "completed_requests": completed_requests,
        "failed_requests": failed_requests,
        "request_success_rate": float(stats.get("request_success_rate", 0.0) or 0.0),
        "execution_success_rate": execution_success_rate,
        "content_success_rate": content_success_rate,
        "content_completed_requests": content_completed_requests,
        "empty_output_requests": empty_output_requests,
        "speculation_roi": roi_value,
        "speculation_roi_mode": roi_mode,
        "speculation_roi_samples": roi_samples,
        "json_grammar_success_rate": grammar_json_success,
        "json_grammar_samples": grammar_samples,
        "evidence_chain_ready": evidence_chain_ready,
        "cold_path": {
            "evaluated_requests": cold_evaluated_requests,
            "completed_requests": cold_completed_requests,
            "failed_requests": cold_failed_requests,
            "ttft_ms_avg": cold_ttft_ms_avg,
            "execution_success_rate": cold_execution_success_rate,
            "content_success_rate": cold_content_success_rate,
            "content_completed_requests": cold_content_completed_requests,
            "empty_output_requests": cold_empty_output_requests,
            "status": cold_path_status,
        },
        "warm_hot_path": {
            "evaluated_requests": warm_hot_evaluated_requests,
            "completed_requests": warm_hot_completed_requests,
            "failed_requests": warm_hot_failed_requests,
            "ttft_ms_avg": warm_hot_ttft_ms_avg,
            "decode_tps_avg": warm_hot_decode_tps_avg,
            "stable_rounds_passed": warm_hot_stable_rounds,
            "execution_success_rate": warm_hot_execution_success_rate,
            "content_success_rate": warm_hot_content_success_rate,
            "content_completed_requests": warm_hot_content_completed_requests,
            "empty_output_requests": warm_hot_empty_output_requests,
            "status": warm_hot_path_status,
        },
        "path_bottlenecks": path_bottlenecks,
    }
    candidate_snapshot["production_gates"] = production_gates
    candidate_snapshot["production_readiness"] = {
        "status": "GO" if overall_passed else "NO_GO",
        "all_passed": overall_passed,
        "routing_status": routing_summary["routing_status"],
        "admissible_degrade": routing_summary["admissible_degrade"],
        "blocking_scope": list(blocking_scope),
        "dense_non_expert_baseline": dense_non_expert_baseline,
        "cold_path_status": cold_path_status,
        "warm_hot_path_status": warm_hot_path_status,
        "shared_status": shared_status,
        "failed_gates": blocking_failed_gate_names,
        "failed_gate_summaries": failed_gate_summaries,
        "cold_path_failed_gates": cold_failed_gate_names,
        "cold_path_failed_gate_summaries": cold_failed_gate_summaries,
        "evaluated_requests": completed_requests + failed_requests,
        "evaluated_requests_by_path": {
            _REQUEST_PATH_COLD: cold_evaluated_requests,
            _REQUEST_PATH_WARM_HOT: warm_hot_evaluated_requests,
        },
    }
    return candidate_snapshot


def _ensure_report_output_dir() -> str:
    os.makedirs(_EDGE_REPORT_OUTPUT_DIR, exist_ok=True)
    return _EDGE_REPORT_OUTPUT_DIR


def _persist_report_json(filename: str, payload: dict[str, Any]) -> str:
    output_dir = _ensure_report_output_dir()
    path = os.path.join(output_dir, filename)
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    try:
        os.replace(tmp_path, path)
    except FileNotFoundError:
        # 某些首次导入场景下，临时文件路径可能在目录初始化竞态里丢失；
        # 回退为直接写正式文件，保证 live snapshot 不缺席。
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    return path


def _persist_live_report_snapshots() -> None:
    with _report_snapshot_lock:
        route_snapshot = copy.deepcopy(_last_route_policy_snapshot)
        heat_snapshot = copy.deepcopy(_last_heat_snapshot)
        candidate_snapshot = copy.deepcopy(_last_single_node_candidate_matrix)
    candidate_snapshot = _build_production_gate_snapshot(candidate_snapshot)
    _persist_report_json("route_policy_v2.live.json", route_snapshot)
    _persist_report_json("route_heat_snapshot.live.json", heat_snapshot)
    _persist_report_json("single_node_candidate_matrix.live.json", candidate_snapshot)
    _persist_report_json("draft_mode_acceptance.live.json", _draft_mode_acceptance_snapshot())


_persist_live_report_snapshots()


def _compute_prefix_hash(user_msg: str, length: int = 256) -> str:
    """计算 prompt 前 N 字符的 hash (L2 cache key).
    标准化: 去除行号、时间戳等可变部分, 保留代码结构.
    """
    normalized = _normalize_text(user_msg)[:length]
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _compute_tail_hash(user_msg: str, length: int = 128) -> str:
    """计算 prompt 末 N 字符的 hash (L3 cache key).
    代码补全场景: 光标前的最后几行决定首 token.
    """
    normalized = _normalize_text(user_msg)
    tail = normalized[-length:] if len(normalized) > length else normalized
    return hashlib.sha1(tail.encode("utf-8")).hexdigest()[:16]


def _predict_first_token(messages: list) -> Optional[str]:
    """基於 prompt 模式預測首 token（多级缓存 + context-aware 規則）。

    查找顺序:
      L1: exact prompt_hash (100% 准确)
      L2: prefix hash (前256 chars, 相似prompt共享)
      L3: context tail hash (末128 chars, 代码补全尾部模式)
      L4: semantic trigram Jaccard similarity (语义相似 prompt)
      L5: family pattern 規則匹配 (top-3 候選)
      L6: fallback "The"

    TTFT < 10ms（純 hash + 規則匹配 + cache lookup）。
    """
    family_info = _classify_prompt_family(messages)
    _prompt_hash = str(family_info.get("prompt_hash") or "")
    user_msg = _extract_user_message(messages)

    # L1: exact cache
    if _prompt_hash:
        with _first_token_cache_lock:
            _cached = _first_token_cache.get(_prompt_hash)
        if _cached:
            _record_stats(cache_hit_l1=1)
            return _cached

    # L2: prefix cache
    _prefix_hash = _compute_prefix_hash(user_msg) if user_msg else ""
    if _prefix_hash:
        with _first_token_cache_lock:
            _cached = _prefix_token_cache.get(_prefix_hash)
        if _cached:
            _record_stats(cache_hit_l2=1)
            return _cached

    # L3: context tail cache
    _tail_hash = _compute_tail_hash(user_msg) if user_msg else ""
    if _tail_hash:
        with _first_token_cache_lock:
            _cached = _tail_token_cache.get(_tail_hash)
        if _cached:
            _record_stats(cache_hit_l3=1)
            return _cached

    # L4: semantic cache (trigram Jaccard similarity)
    sem_token, sem_sim = _check_semantic_cache(user_msg)
    if sem_token:
        _record_stats(cache_hit_l4=1)
        return sem_token

    _record_stats(cache_miss=1)
    tokenizer = _load_edge_tokenizer()
    if tokenizer is None:
        return "The"

    candidates = list(family_info.get("candidates") or _FIRST_TOKEN_PATTERNS[-1][1])

    # 驗證候選 token 是否為單 token（用 tokenizer）
    for token_text in candidates:
        try:
            ids = tokenizer.encode(token_text, add_special_tokens=False)
            if len(ids) == 1:
                return token_text
        except Exception:
            continue

    return candidates[0] if candidates else "The"


def _record_first_token(prompt_hash: str, first_token: str, user_msg: str = "") -> None:
    """记录 cloud 实际返回的首 token 到多级缓存（供下次预测）。

    L1: exact prompt_hash (100% 准确)
    L2: prefix hash (相似 prompt 共享)
    L3: context tail hash (代码补全尾部模式)
    L4: semantic trigram (语义相似 prompt 共享)
    """
    if not first_token:
        return
    with _first_token_cache_lock:
        # L1: exact
        if prompt_hash:
            if len(_first_token_cache) >= _FIRST_TOKEN_CACHE_MAX:
                _keys = list(_first_token_cache.keys())
                for k in _keys[:len(_keys) // 2]:
                    del _first_token_cache[k]
            _first_token_cache[prompt_hash] = first_token
        # L2: prefix
        if user_msg:
            p_hash = _compute_prefix_hash(user_msg)
            if p_hash:
                if len(_prefix_token_cache) >= _FIRST_TOKEN_CACHE_MAX:
                    _keys = list(_prefix_token_cache.keys())
                    for k in _keys[:len(_keys) // 2]:
                        del _prefix_token_cache[k]
                _prefix_token_cache[p_hash] = first_token
            # L3: tail
            t_hash = _compute_tail_hash(user_msg)
            if t_hash:
                if len(_tail_token_cache) >= _FIRST_TOKEN_CACHE_MAX:
                    _keys = list(_tail_token_cache.keys())
                    for k in _keys[:len(_keys) // 2]:
                        del _tail_token_cache[k]
                _tail_token_cache[t_hash] = first_token

    # L4: semantic (在锁外执行, 避免长时间持锁)
    if user_msg and prompt_hash:
        _record_semantic(user_msg, first_token, prompt_hash)


def _edge_generate_first_token(messages: list, max_tokens: int = 1, model_name: str = "") -> Optional[str]:
    """生成首 token（context-aware 預測或本地小模型）。

    優先使用 DSV4 tokenizer 的 context-aware 預測（TTFT < 10ms）。
    如果設定了 EDGE_MODEL_PATH，則使用本地小模型（llama-cpp-python）。

    返回首 token 文本，失敗返回 None。
    """
    # 模式 1: DSV4 tokenizer context-aware 預測（預設，無需 GGUF）
    if not os.environ.get("EDGE_MODEL_PATH"):
        t0 = time.monotonic()
        text = _predict_first_token(messages)
        elapsed = time.monotonic() - t0
        print(f"[edge-first] 首 token (tokenizer): '{text}' ({elapsed*1000:.0f}ms)",
              file=sys.stderr)
        return text

    # 模式 2: 本地小模型（llama-cpp-python）
    llm = _load_edge_model()
    if llm is None:
        return _predict_first_token(messages)

    try:
        # 構造 prompt（使用 DSV4 chat template）
        tokenizer = _load_edge_tokenizer()
        template_kwargs: dict[str, Any] = {}
        if _should_strip_reasoning_tags(model_name):
            template_kwargs["enable_thinking"] = False
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                _normalized_messages_for_model(messages, model_name=model_name),
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        else:
            # fallback: 簡單拼接
            prompt_parts = []
            for msg in _normalized_messages_for_model(messages, model_name=model_name):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(prompt_parts)

        t0 = time.monotonic()
        resp = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|im_end|>", "\n"],
            echo=False,
        )
        elapsed = time.monotonic() - t0
        text = resp.get("choices", [{}])[0].get("text", "").strip()
        print(f"[edge-first] 首 token (llm): '{text}' ({elapsed*1000:.0f}ms)",
              file=sys.stderr)
        return text if text else None
    except Exception as e:
        print(f"[edge-first] LLM 生成失敗: {e}", file=sys.stderr)
        return _predict_first_token(messages)


# === 异步连接池 (aiohttp, 支持真并发流式) ===
import aiohttp as _aiohttp

_aiohttp_session: Optional[_aiohttp.ClientSession] = None


async def _get_cloud_session() -> _aiohttp.ClientSession:
    """获取共享 aiohttp ClientSession (支持真并发流式, keep-alive)."""
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        _aiohttp_session = _aiohttp.ClientSession(
            connector=_aiohttp.TCPConnector(limit=0, keepalive_timeout=30),
            timeout=_aiohttp.ClientTimeout(total=120),
        )
    return _aiohttp_session


async def _cloud_stream(url: str, payload: dict, headers: dict):
    """異步轉發請求到雲端 sglang，streaming 返回 (aiohttp, 支持真並發)."""
    session = await _get_cloud_session()
    resp = await session.post(
        url,
        json=payload,
        headers={**headers, "Content-Type": "application/json"},
    )
    try:
        async for chunk in resp.content.iter_any():
            if chunk:
                yield chunk
    finally:
        resp.release()


_local_llama_cache = None
_local_llama_cache_path = None
_local_llama_cache_n_gpu_layers = -1  # P3+ 持续监控: 记录当前 n_gpu_layers
_local_llama_lock = threading.Lock()
_local_mlx_cache = None
_local_mlx_cache_path = None
_local_mlx_lock = threading.Lock()
_local_verify_backends: dict[str, VerifyLoopBackend] = {}
_local_verify_lock = threading.Lock()


def _local_llama_route_state() -> dict[str, Any]:
    loaded = bool(_local_llama_cache is not None and _local_llama_cache_path == _LOCAL_FULL_MODEL_PATH)
    n_gpu_layers = int(_local_llama_cache_n_gpu_layers if _local_llama_cache_n_gpu_layers is not None else -1)
    return {
        "local_llama_loaded": loaded,
        "local_llama_n_gpu_layers": n_gpu_layers,
        "local_llama_full_resident_loaded": bool(loaded and n_gpu_layers == -1),
    }


def _augment_transport_runtime_snapshot() -> dict[str, Any]:
    runtime = dict(_transport_runtime_snapshot())
    runtime.update(_local_llama_route_state())
    return runtime


def _align_transport_route_with_local_runtime(route_context: dict[str, Any]) -> dict[str, Any]:
    aligned = dict(route_context or {})
    runtime_state = _local_llama_route_state()
    aligned.update(runtime_state)
    if runtime_state["local_llama_full_resident_loaded"]:
        # Trust the already-loaded full-resident runtime for fit-in-memory A/B comparisons,
        # even when the control plane still suggests degrade.
        aligned["desired_mode"] = ROUTE_LOCAL_FULL
        aligned["mode"] = ROUTE_LOCAL_FULL
        aligned["mode_hint"] = ROUTE_LOCAL_FULL
        aligned["reason"] = "local_llama_full_resident_loaded"
        aligned["mode_switch_reason"] = "local_llama_full_resident_loaded"
        aligned["full_resident_admissible"] = True
        aligned["full_resident_relaxed_admissible"] = True
        aligned["runtime_full_resident_override"] = True
    else:
        aligned["runtime_full_resident_override"] = False
    return aligned


def _align_transport_route_with_runtime_unit_plan(
    route_context: dict[str, Any],
    expert_preview: Optional[dict[str, Any]],
) -> dict[str, Any]:
    aligned = dict(route_context or {})
    runtime_unit_plan = dict((expert_preview or {}).get("runtime_unit_plan") or {})
    summary = dict(runtime_unit_plan.get("summary") or {})
    aligned["runtime_unit_plan_summary"] = summary

    if bool(aligned.get("runtime_full_resident_override")) or not bool(runtime_unit_plan.get("enabled")):
        aligned["runtime_unit_route_override"] = False
        return aligned

    current_unit_count = int(summary.get("current_unit_count") or 0)
    predicted_read_bytes = int(float(summary.get("predicted_bytes_to_read_mb") or 0.0) * 1024**2)
    predicted_cold_bytes = int(float(summary.get("predicted_cold_bytes_mb") or 0.0) * 1024**2)
    tier_counts = {
        str(key): int(value or 0)
        for key, value in dict(summary.get("tier_counts") or {}).items()
    }
    current_tier_counts = {
        str(key): int(value or 0)
        for key, value in dict(summary.get("current_tier_counts") or {}).items()
    }
    resident_budget_bytes = int(summary.get("resident_budget_bytes") or 0)
    resident_bytes = int(summary.get("resident_bytes") or 0)
    pin_budget_bytes = int(summary.get("pin_budget_bytes") or 0)
    pinned_bytes = int(summary.get("pinned_bytes") or 0)
    resident_headroom_bytes = max(resident_budget_bytes - resident_bytes, 0)
    pin_headroom_bytes = max(pin_budget_bytes - pinned_bytes, 0)
    current_nvme_units = int(current_tier_counts.get("nvme", 0))
    current_local_units = max(current_unit_count - current_nvme_units, 0)
    local_capable = bool(
        aligned.get("local_full_model_configured")
        or aligned.get("local_streaming_backend_configured")
    )
    layer_split_capable = bool(
        aligned.get("layer_split_model_configured")
        or aligned.get("local_streaming_backend_configured")
    )

    selected_mode = str(aligned.get("mode") or aligned.get("desired_mode") or ROUTE_CLOUD_PD)
    selected_reason = ""

    if current_unit_count <= 0:
        selected_mode = ROUTE_CLOUD_PD if (local_capable or layer_split_capable) else ROUTE_CLOUD_FALLBACK
        selected_reason = "runtime_unit_plan_empty_current"
    elif (
        local_capable
        and current_nvme_units == 0
        and predicted_cold_bytes <= max(32 * 1024**2, resident_headroom_bytes)
    ):
        selected_mode = ROUTE_LOCAL_FULL
        selected_reason = "runtime_unit_plan_local_full"
    elif (
        local_capable
        and predicted_cold_bytes > 0
        and resident_headroom_bytes >= predicted_cold_bytes
    ):
        selected_mode = ROUTE_LOCAL_FULL
        selected_reason = "runtime_unit_plan_local_warmable"
    elif layer_split_capable and current_local_units > 0:
        selected_mode = ROUTE_LAYER_SPLIT_PD
        selected_reason = "runtime_unit_plan_layer_split"
    elif layer_split_capable and (int(tier_counts.get("resident_ram", 0)) + int(tier_counts.get("pinned_ram", 0))) > 0:
        selected_mode = ROUTE_LAYER_SPLIT_PD
        selected_reason = "runtime_unit_plan_mixed_tiers"
    else:
        selected_mode = ROUTE_CLOUD_PD if (local_capable or layer_split_capable) else ROUTE_CLOUD_FALLBACK
        selected_reason = "runtime_unit_plan_cloud_pd"

    evidence = {
        "current_unit_count": current_unit_count,
        "predicted_bytes_to_read_mb": round(predicted_read_bytes / 1024**2, 3),
        "predicted_cold_bytes_mb": round(predicted_cold_bytes / 1024**2, 3),
        "tier_counts": tier_counts,
        "current_tier_counts": current_tier_counts,
        "resident_budget_bytes": resident_budget_bytes,
        "resident_bytes": resident_bytes,
        "resident_headroom_bytes": resident_headroom_bytes,
        "pin_budget_bytes": pin_budget_bytes,
        "pinned_bytes": pinned_bytes,
        "pin_headroom_bytes": pin_headroom_bytes,
        "selected_mode": selected_mode,
        "selected_reason": selected_reason,
    }
    aligned["runtime_unit_route_evidence"] = evidence

    if selected_mode != str(aligned.get("mode") or aligned.get("desired_mode") or ""):
        aligned["desired_mode"] = selected_mode
        aligned["mode"] = selected_mode
        aligned["mode_hint"] = selected_mode
        aligned["reason"] = selected_reason
        aligned["mode_switch_reason"] = selected_reason
        aligned["runtime_unit_route_override"] = True
    else:
        aligned["runtime_unit_route_override"] = False
    return aligned


def _preview_request_for_route(
    *,
    body: dict[str, Any],
    family_info: dict[str, Any],
    route_mode: str,
) -> dict[str, Any]:
    return _expert_data_plane.preview_request(
        model_name=str(body.get("model") or _ACTIVE_MODEL),
        family_info=family_info,
        draft_policy=None,
        route_mode=str(route_mode or ROUTE_CLOUD_PD),
    )


def _candidate_tokenizer_paths() -> list[str]:
    paths: list[str] = []
    for candidate in (
        DEFAULT_DSV4_TOKENIZER_PATH,
        os.environ.get("EDGE_TOKENIZER_FALLBACK_PATH", ""),
        os.path.join(REPO_ROOT, "models", "gemma-4-mtp-head"),
    ):
        path = str(candidate or "").strip()
        if not path:
            continue
        if path not in paths:
            paths.append(path)
    return paths


def _local_llama_runtime_options() -> dict[str, Any]:
    cpu_count = max(os.cpu_count() or 4, 4)
    runtime_options = {
        "n_ctx": int(os.environ.get("EDGE_LOCAL_N_CTX", "4096") or "4096"),
        "n_batch": int(os.environ.get("EDGE_LOCAL_N_BATCH", "1024") or "1024"),
        "n_ubatch": int(os.environ.get("EDGE_LOCAL_N_UBATCH", "512") or "512"),
        "n_threads": int(os.environ.get("EDGE_LOCAL_N_THREADS", str(cpu_count)) or str(cpu_count)),
        "n_threads_batch": int(os.environ.get("EDGE_LOCAL_N_THREADS_BATCH", str(cpu_count)) or str(cpu_count)),
        "flash_attn": os.environ.get("EDGE_LOCAL_FLASH_ATTN", "1") == "1",
        "offload_kqv": os.environ.get("EDGE_LOCAL_OFFLOAD_KQV", "1") == "1",
        "use_mmap": os.environ.get("EDGE_LOCAL_USE_MMAP", "1") == "1",
        "use_mlock": os.environ.get("EDGE_LOCAL_USE_MLOCK", "0") == "1",
    }
    model_path_lower = str(_LOCAL_FULL_MODEL_PATH or "").strip().lower()
    forced_n_gpu_raw = str(os.environ.get("EDGE_LOCAL_N_GPU_LAYERS", "") or "").strip()
    forced_n_gpu = None
    if forced_n_gpu_raw:
        try:
            forced_n_gpu = int(forced_n_gpu_raw)
        except ValueError:
            forced_n_gpu = None
    stability_guard_enabled = os.environ.get("EDGE_LOCAL_DISABLE_STABILITY_GUARD", "0") != "1"
    if (
        stability_guard_enabled
        and forced_n_gpu == -1
        and "e4b" in model_path_lower
    ):
        original_batch = int(runtime_options["n_batch"])
        original_ubatch = int(runtime_options["n_ubatch"])
        runtime_options["n_batch"] = min(original_batch, 256)
        runtime_options["n_ubatch"] = min(original_ubatch, 128, int(runtime_options["n_batch"]))
        if (
            int(runtime_options["n_batch"]) != original_batch
            or int(runtime_options["n_ubatch"]) != original_ubatch
        ):
            print(
                f"[edge-router] stability guard: clamp local llama batch for E4B "
                f"(n_gpu_layers=-1, n_batch {original_batch}->{runtime_options['n_batch']}, "
                f"n_ubatch {original_ubatch}->{runtime_options['n_ubatch']})",
                file=sys.stderr,
            )
    return runtime_options
 

def _candidate_local_n_gpu_layers(*, model_size_gb: float, total_layers: int, residency_hint: dict[str, Any]) -> list[int]:
    forced_raw = str(os.environ.get("EDGE_LOCAL_N_GPU_LAYERS", "") or "").strip()
    if forced_raw:
        try:
            forced = int(forced_raw)
            return [forced]
        except ValueError:
            pass
    suggested = int(residency_hint.get("suggested_n_gpu_layers", -1))
    can_full_resident = bool(residency_hint.get("can_full_resident", True))
    candidates: list[int] = []
    aggressive_gpu = os.environ.get("EDGE_LOCAL_AGGRESSIVE_GPU", "1") == "1"
    expert_occupied = float(residency_hint.get("expert_occupied_gb", 0.0) or 0.0)
    mostly_gpu = max(total_layers - 4, 1) if total_layers > 0 else None
    if aggressive_gpu and can_full_resident and expert_occupied <= 0.25:
        candidates.append(-1)
    if mostly_gpu is not None and suggested == 0 and mostly_gpu not in candidates:
        candidates.append(mostly_gpu)
    if suggested not in candidates:
        candidates.append(suggested)
    if aggressive_gpu and not can_full_resident and expert_occupied <= 0.25 and -1 not in candidates:
        candidates.append(-1)
    if mostly_gpu is not None and suggested not in (-1, total_layers, 0) and mostly_gpu not in candidates:
        candidates.append(mostly_gpu)
    if 0 not in candidates:
        candidates.append(0)
    return candidates


def _build_prompt_from_messages(messages: list[dict[str, Any]], model_name: str = "") -> str:
    normalized_messages = _normalized_messages_for_model(messages, model_name=model_name)
    template_kwargs: dict[str, Any] = {}
    if _should_strip_reasoning_tags(model_name):
        template_kwargs["enable_thinking"] = False
    tokenizer = _load_edge_tokenizer()
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            normalized_messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
    prompt_parts = []
    for msg in normalized_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def _get_local_verify_backend(
    *,
    model_name: str,
    n_gpu_layers: Optional[int] = None,
) -> Optional[VerifyLoopBackend]:
    if not _LOCAL_FULL_MODEL_PATH:
        return None
    model_info = dict(Bootstrap.DRAFT_MODELS.get(model_name) or {})
    verify_cfg = dict(model_info.get("verify_loop_config") or {})
    if not verify_cfg:
        return None
    resolved_cfg = _resolve_active_model_cfg(model_name)
    runtime_options = _local_llama_runtime_options()
    resolved_n_gpu_layers = int(
        n_gpu_layers
        if n_gpu_layers is not None
        else verify_cfg.get(
            "n_gpu_layers",
            _local_llama_cache_n_gpu_layers if _local_llama_cache_n_gpu_layers is not None else -1,
        )
    )
    resolved_n_ctx = int(verify_cfg.get("n_ctx", runtime_options["n_ctx"]) or runtime_options["n_ctx"])
    resolved_n_batch = int(verify_cfg.get("n_batch", runtime_options["n_batch"]) or runtime_options["n_batch"])
    resolved_n_ubatch = int(verify_cfg.get("n_ubatch", runtime_options["n_ubatch"]) or runtime_options["n_ubatch"])
    resolved_use_mlx = bool(verify_cfg.get("use_mlx", False))
    resolved_use_cgc_ir = bool(verify_cfg.get("use_cgc_ir", False))
    resolved_use_ggml = bool(verify_cfg.get("use_ggml", False))
    key = (
        f"{model_name}|{_LOCAL_FULL_MODEL_PATH}"
        f"|ngl={resolved_n_gpu_layers}|ctx={resolved_n_ctx}"
        f"|batch={resolved_n_batch}|ubatch={resolved_n_ubatch}"
        f"|mlx={int(resolved_use_mlx)}|cgc={int(resolved_use_cgc_ir)}|ggml={int(resolved_use_ggml)}"
    )
    with _local_verify_lock:
        backend = _local_verify_backends.get(key)
        if backend is not None:
            return backend
        def _resolve_verify_artifact_path(raw_path: str, *, kind: str) -> str:
            candidate = str(raw_path or "").strip()
            if candidate:
                if not os.path.isabs(candidate):
                    candidate = os.path.join(REPO_ROOT, candidate)
                if os.path.exists(candidate):
                    return candidate

            target_name = str(getattr(resolved_cfg, "name", model_name) or model_name).strip()
            if resolved_cfg is not None:
                try:
                    default_path = (
                        resolved_cfg.get_checkpoint_path()
                        if kind == "checkpoint"
                        else resolved_cfg.get_embed_head_path()
                    )
                    default_path = str(default_path or "").strip()
                    if default_path and os.path.exists(default_path):
                        return default_path
                except Exception:
                    pass

            return candidate

        resolved_checkpoint = _resolve_verify_artifact_path(
            str(verify_cfg.get("mtp_checkpoint") or ""),
            kind="checkpoint",
        )
        resolved_embed_head = _resolve_verify_artifact_path(
            str(verify_cfg.get("embed_head_path") or ""),
            kind="embed_head",
        )
        backend = VerifyLoopBackend.get_or_create(
            model_key=key,
            model_path=_LOCAL_FULL_MODEL_PATH,
            mtp_checkpoint=resolved_checkpoint,
            embed_head_path=resolved_embed_head,
            assistant_model_path=str(verify_cfg.get("assistant_model_path") or ""),
            hidden_size=int(getattr(resolved_cfg, "hidden_size", model_info.get("draft_hidden_size", 1024)) or model_info.get("draft_hidden_size", 1024)),
            vocab_size=int(getattr(resolved_cfg, "vocab_size", model_info.get("draft_vocab_size", 262144)) or model_info.get("draft_vocab_size", 262144)),
            num_heads=int(verify_cfg.get("num_heads", 16) or 16),
            head_dim=int(verify_cfg.get("head_dim", 256) or 256),
            intermediate_size=int(verify_cfg.get("intermediate_size", 8192) or 8192),
            n_ctx=resolved_n_ctx,
            n_batch=resolved_n_batch,
            n_ubatch=resolved_n_ubatch,
            n_threads=int(runtime_options["n_threads"]),
            n_threads_batch=int(runtime_options["n_threads_batch"]),
            flash_attn=bool(runtime_options["flash_attn"]),
            offload_kqv=bool(runtime_options["offload_kqv"]),
            use_mmap=bool(runtime_options["use_mmap"]),
            use_mlock=bool(runtime_options["use_mlock"]),
            n_gpu_layers=resolved_n_gpu_layers,
            use_mlx=resolved_use_mlx,
            use_cgc_ir=resolved_use_cgc_ir,
            use_ggml=resolved_use_ggml,
        )
        _local_verify_backends[key] = backend
        return backend


def _load_local_llama():
    """延遲加載本地 llama.cpp 主模型。失敗返回 None。

    P3+ 持续性 dense backbone residency 管理:
    - 首次加载: 从 expert_data_plane 获取建议, 设置 baseline
    - Cache hit: 检查 expert 占用是否变化超阈值, 如果是则 reload
    """
    global _local_llama_cache, _local_llama_cache_path, _local_llama_cache_n_gpu_layers
    with _local_llama_lock:
        runtime_options = _local_llama_runtime_options()
        if _local_llama_cache is not None and _local_llama_cache_path == _LOCAL_FULL_MODEL_PATH:
            # P3+ 持续监控: 检查是否需要 reload (expert 占用变化超阈值)
            reload_hint = None
            try:
                reload_hint = _expert_data_plane.check_dense_residency_reload()
            except Exception:
                pass
            if reload_hint is not None:
                new_n_gpu = int(reload_hint.get("n_gpu_layers", -1))
                if new_n_gpu != _local_llama_cache_n_gpu_layers:
                    print(
                        f"[edge-router] dense residency reload 触发: "
                        f"{reload_hint.get('reason', '')} "
                        f"n_gpu_layers {_local_llama_cache_n_gpu_layers} → {new_n_gpu}",
                        file=sys.stderr,
                    )
                    try:
                        from llama_cpp import Llama
                        _local_llama_cache = Llama(
                            model_path=_LOCAL_FULL_MODEL_PATH,
                            n_ctx=runtime_options["n_ctx"],
                            n_batch=runtime_options["n_batch"],
                            n_ubatch=runtime_options["n_ubatch"],
                            n_gpu_layers=new_n_gpu,
                            n_threads=runtime_options["n_threads"],
                            n_threads_batch=runtime_options["n_threads_batch"],
                            flash_attn=runtime_options["flash_attn"],
                            offload_kqv=runtime_options["offload_kqv"],
                            use_mmap=runtime_options["use_mmap"],
                            use_mlock=runtime_options["use_mlock"],
                            verbose=False,
                            logits_all=False,
                        )
                        _local_llama_cache_n_gpu_layers = new_n_gpu
                        print(
                            f"[edge-router] dense residency reload 完成: "
                            f"n_gpu_layers={new_n_gpu}",
                            file=sys.stderr,
                        )
                    except Exception as e:
                        print(f"[edge-router] dense residency reload 失败 (保留旧模型): {e}", file=sys.stderr)
            return _local_llama_cache
        if not _LOCAL_FULL_MODEL_PATH:
            return None
        try:
            from llama_cpp import Llama

            # P3: dense backbone resident 与 plan 联动
            # 估算模型大小 (从文件大小), 读 expert_data_plane 建议
            model_size_gb = 0.0
            try:
                model_size_gb = os.path.getsize(_LOCAL_FULL_MODEL_PATH) / 1024**3
            except Exception:
                pass

            # 环境变量可显式覆盖；否则回退到 registry 已知层数，避免误判成 CPU-only
            total_layers = int(os.environ.get("EDGE_LOCAL_MODEL_LAYERS", "0") or "0")
            if total_layers <= 0 and _ACTIVE_MODEL_CFG is not None:
                total_layers = int(getattr(_ACTIVE_MODEL_CFG, "num_hidden_layers", 0) or 0)

            n_gpu_layers = -1  # 默认全 GPU
            residency_hint = {}
            try:
                residency_hint = _expert_data_plane.recommend_dense_residency(
                    model_size_gb=model_size_gb,
                    total_layers=total_layers,
                )
                suggested = int(residency_hint.get("suggested_n_gpu_layers", -1))
                n_gpu_layers = suggested
                if not residency_hint.get("can_full_resident", True):
                    print(
                        f"[edge-router] dense backbone 部分 offload: "
                        f"{residency_hint.get('reason', '')} "
                        f"n_gpu_layers={n_gpu_layers} "
                        f"offload_ratio={residency_hint.get('offload_ratio', 0.0):.0%}",
                        file=sys.stderr,
                    )
                # P3+ 设置 baseline (持续监控的起点)
                _expert_data_plane.set_dense_residency_baseline(
                    model_size_gb=model_size_gb,
                    total_layers=total_layers,
                    model_path=_LOCAL_FULL_MODEL_PATH,
                )
            except Exception as e:
                print(f"[edge-router] recommend_dense_residency 失败 (用默认全GPU): {e}", file=sys.stderr)

            last_error: Optional[Exception] = None
            for candidate_n_gpu in _candidate_local_n_gpu_layers(
                model_size_gb=model_size_gb,
                total_layers=total_layers,
                residency_hint=residency_hint,
            ):
                try:
                    _local_llama_cache = Llama(
                        model_path=_LOCAL_FULL_MODEL_PATH,
                        n_ctx=runtime_options["n_ctx"],
                        n_batch=runtime_options["n_batch"],
                        n_ubatch=runtime_options["n_ubatch"],
                        n_gpu_layers=candidate_n_gpu,
                        n_threads=runtime_options["n_threads"],
                        n_threads_batch=runtime_options["n_threads_batch"],
                        flash_attn=runtime_options["flash_attn"],
                        offload_kqv=runtime_options["offload_kqv"],
                        use_mmap=runtime_options["use_mmap"],
                        use_mlock=runtime_options["use_mlock"],
                        verbose=False,
                        logits_all=False,
                    )
                    _local_llama_cache_path = _LOCAL_FULL_MODEL_PATH
                    _local_llama_cache_n_gpu_layers = candidate_n_gpu
                    print(
                        f"[edge-router] llama.cpp 主模型加載成功: {_LOCAL_FULL_MODEL_PATH} "
                        f"(size={model_size_gb:.1f}GB, n_gpu_layers={candidate_n_gpu}, "
                        f"n_ctx={runtime_options['n_ctx']}, n_batch={runtime_options['n_batch']}, "
                        f"flash_attn={runtime_options['flash_attn']}, "
                        f"expert_occupied={residency_hint.get('expert_occupied_gb', 0.0):.1f}GB)",
                        file=sys.stderr,
                    )
                    # #region debug-point A:llama-load-ok
                    try:
                        _dbg_p = os.path.join(REPO_ROOT, ".dbg", "local-full-decode.env")
                        _dbg_u, _dbg_s = "http://127.0.0.1:7777/event", "local-full-decode"
                        try:
                            with open(_dbg_p, "r", encoding="utf-8") as _dbg_f:
                                _dbg_c = _dbg_f.read()
                            _dbg_u = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SERVER_URL=")), _dbg_u)
                            _dbg_s = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SESSION_ID=")), _dbg_s)
                        except Exception:
                            pass
                        urllib.request.urlopen(
                            urllib.request.Request(
                                _dbg_u,
                                data=json.dumps({
                                    "sessionId": _dbg_s,
                                    "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                                    "hypothesisId": "A",
                                    "location": "app/servers/edge_first_proxy.py:_load_local_llama",
                                    "msg": "[DEBUG] local llama load success",
                                    "data": {
                                        "model_path": _LOCAL_FULL_MODEL_PATH,
                                        "model_size_gb": round(float(model_size_gb or 0.0), 3),
                                        "candidate_n_gpu_layers": int(candidate_n_gpu),
                                        "n_ctx": int(runtime_options["n_ctx"]),
                                        "n_batch": int(runtime_options["n_batch"]),
                                        "n_ubatch": int(runtime_options["n_ubatch"]),
                                        "flash_attn": bool(runtime_options["flash_attn"]),
                                        "offload_kqv": bool(runtime_options["offload_kqv"]),
                                        "can_full_resident": bool(residency_hint.get("can_full_resident", True)),
                                        "suggested_n_gpu_layers": int(residency_hint.get("suggested_n_gpu_layers", -1) or -1),
                                        "resident_reason": str(residency_hint.get("reason") or ""),
                                    },
                                }).encode("utf-8"),
                                headers={"Content-Type": "application/json"},
                            ),
                            timeout=1,
                        ).read()
                    except Exception:
                        pass
                    # #endregion
                    return _local_llama_cache
                except Exception as load_error:
                    last_error = load_error
                    print(
                        f"[edge-router] llama.cpp 加載候選失敗: n_gpu_layers={candidate_n_gpu} err={load_error}",
                        file=sys.stderr,
                    )
            if last_error is not None:
                raise last_error
            return None
        except Exception as e:
            print(f"[edge-router] llama.cpp 主模型加載失敗 {_LOCAL_FULL_MODEL_PATH}: {e}", file=sys.stderr)
            return None


def _load_local_mlx():
    """延遲加載 MLX layer-split 模型。失敗返回 None。"""
    global _local_mlx_cache, _local_mlx_cache_path
    with _local_mlx_lock:
        if _local_mlx_cache is not None and _local_mlx_cache_path == _LAYER_SPLIT_MODEL_PATH:
            return _local_mlx_cache
        if not _LAYER_SPLIT_MODEL_PATH:
            return None
        try:
            from mlx_lm import load
            _local_mlx_cache = load(_LAYER_SPLIT_MODEL_PATH)
            _local_mlx_cache_path = _LAYER_SPLIT_MODEL_PATH
            print(f"[edge-router] MLX layer-split 加載成功: {_LAYER_SPLIT_MODEL_PATH}", file=sys.stderr)
            return _local_mlx_cache
        except Exception as e:
            print(f"[edge-router] MLX layer-split 加載失敗 {_LAYER_SPLIT_MODEL_PATH}: {e}", file=sys.stderr)
            return None


def _local_llama_stream(body: dict):
    """全本地 llama.cpp streaming 推理。失敗回退雲(由調用者處理)。"""
    llm = _load_local_llama()
    if llm is None:
        raise RuntimeError("llama_cpp_load_failed")
    messages = body.get("messages", [])
    model_name = str(body.get("model") or _ACTIVE_MODEL or "")
    prompt = _build_prompt_from_messages(messages, model_name=model_name)
    max_tokens = int(body.get("max_tokens", 32) or 32)
    sanitizer = _ReasoningTagFilter() if _should_strip_reasoning_tags(model_name) else None
    prompt_token_est = max(len(prompt) // 4, 1) if prompt else 0
    try:
        _dbg_tok = _load_edge_tokenizer()
        if _dbg_tok is not None:
            prompt_token_est = max(len(_dbg_tok.encode(prompt, add_special_tokens=False)), 0)
    except Exception:
        pass
    try:
        # #region debug-point B:local-llama-stream-enter
        try:
            _dbg_p = os.path.join(REPO_ROOT, ".dbg", "local-full-decode.env")
            _dbg_u, _dbg_s = "http://127.0.0.1:7777/event", "local-full-decode"
            try:
                with open(_dbg_p, "r", encoding="utf-8") as _dbg_f:
                    _dbg_c = _dbg_f.read()
                _dbg_u = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SERVER_URL=")), _dbg_u)
                _dbg_s = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SESSION_ID=")), _dbg_s)
            except Exception:
                pass
            urllib.request.urlopen(
                urllib.request.Request(
                    _dbg_u,
                    data=json.dumps({
                        "sessionId": _dbg_s,
                        "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                        "hypothesisId": "B",
                        "location": "app/servers/edge_first_proxy.py:_local_llama_stream",
                        "msg": "[DEBUG] enter local llama stream",
                        "data": {
                            "model_name": model_name,
                            "message_count": len(messages),
                            "prompt_chars": len(prompt),
                            "prompt_token_est": int(prompt_token_est),
                            "max_tokens": int(max_tokens),
                            "n_gpu_layers": int(_local_llama_cache_n_gpu_layers if _local_llama_cache_n_gpu_layers is not None else -1),
                            "expert_streaming_enabled": os.environ.get("EDGE_EXPERT_STREAMING_ENABLED", ""),
                            "dense_layer_streaming_enabled": os.environ.get("EDGE_DENSE_LAYER_STREAMING_ENABLED", ""),
                            "n_batch": os.environ.get("EDGE_LOCAL_N_BATCH", ""),
                            "n_ubatch": os.environ.get("EDGE_LOCAL_N_UBATCH", ""),
                            "flash_attn": os.environ.get("EDGE_LOCAL_FLASH_ATTN", ""),
                        },
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=1,
            ).read()
        except Exception:
            pass
        # #endregion
        created = int(time.time())
        cid = f"local_{created}"
        first = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": body.get("model", "local-llama"),
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n".encode()
        first_token_logged = False
        for chunk in llm(
            prompt,
            max_tokens=max_tokens,
            temperature=float(body.get("temperature", 0.0) or 0.0),
            stream=True,
            echo=False,
        ):
            text = str((chunk.get("choices") or [{}])[0].get("text") or "")
            if sanitizer is not None:
                text = sanitizer.feed(text)
            if not text:
                continue
            if not first_token_logged:
                # #region debug-point C:first-token-seen
                try:
                    _dbg_p = os.path.join(REPO_ROOT, ".dbg", "local-full-decode.env")
                    _dbg_u, _dbg_s = "http://127.0.0.1:7777/event", "local-full-decode"
                    try:
                        with open(_dbg_p, "r", encoding="utf-8") as _dbg_f:
                            _dbg_c = _dbg_f.read()
                        _dbg_u = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SERVER_URL=")), _dbg_u)
                        _dbg_s = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SESSION_ID=")), _dbg_s)
                    except Exception:
                        pass
                    urllib.request.urlopen(
                        urllib.request.Request(
                            _dbg_u,
                            data=json.dumps({
                                "sessionId": _dbg_s,
                                "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                                "hypothesisId": "C",
                                "location": "app/servers/edge_first_proxy.py:_local_llama_stream",
                                "msg": "[DEBUG] first local token emitted",
                                "data": {
                                    "model_name": model_name,
                                    "n_gpu_layers": int(_local_llama_cache_n_gpu_layers if _local_llama_cache_n_gpu_layers is not None else -1),
                                    "text_preview": text[:80],
                                    "text_len": len(text),
                                },
                            }).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                        ),
                        timeout=1,
                    ).read()
                except Exception:
                    pass
                # #endregion
                first_token_logged = True
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": body.get("model", "local-llama"),
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        final_piece = sanitizer.finish() if sanitizer is not None else ""
        if final_piece:
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": body.get("model", "local-llama"),
                "choices": [{"index": 0, "delta": {"content": final_piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        done = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": body.get("model", "local-llama"),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except Exception as e:
        # #region debug-point D:local-llama-stream-error
        try:
            _dbg_p = os.path.join(REPO_ROOT, ".dbg", "local-full-decode.env")
            _dbg_u, _dbg_s = "http://127.0.0.1:7777/event", "local-full-decode"
            try:
                with open(_dbg_p, "r", encoding="utf-8") as _dbg_f:
                    _dbg_c = _dbg_f.read()
                _dbg_u = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SERVER_URL=")), _dbg_u)
                _dbg_s = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SESSION_ID=")), _dbg_s)
            except Exception:
                pass
            urllib.request.urlopen(
                urllib.request.Request(
                    _dbg_u,
                    data=json.dumps({
                        "sessionId": _dbg_s,
                        "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                        "hypothesisId": "D",
                        "location": "app/servers/edge_first_proxy.py:_local_llama_stream",
                        "msg": "[DEBUG] local llama stream error",
                        "data": {
                            "model_name": model_name,
                            "error": repr(e),
                            "error_text": str(e),
                            "prompt_chars": len(prompt),
                            "prompt_token_est": int(prompt_token_est),
                            "max_tokens": int(max_tokens),
                            "n_gpu_layers": int(_local_llama_cache_n_gpu_layers if _local_llama_cache_n_gpu_layers is not None else -1),
                            "expert_streaming_enabled": os.environ.get("EDGE_EXPERT_STREAMING_ENABLED", ""),
                            "dense_layer_streaming_enabled": os.environ.get("EDGE_DENSE_LAYER_STREAMING_ENABLED", ""),
                            "n_batch": os.environ.get("EDGE_LOCAL_N_BATCH", ""),
                            "n_ubatch": os.environ.get("EDGE_LOCAL_N_UBATCH", ""),
                            "flash_attn": os.environ.get("EDGE_LOCAL_FLASH_ATTN", ""),
                        },
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=1,
            ).read()
        except Exception:
            pass
        # #endregion
        err = {"error": {"message": f"local llama.cpp error: {e}", "type": "local_error"}}
        yield f"data: {json.dumps(err)}\n\n".encode()


def _local_verify_loop_stream(
    body: dict,
    *,
    draft_policy: dict[str, Any],
    metrics: dict[str, Any],
    runtime_unit_plan: Optional[dict[str, Any]] = None,
):
    """本地 verify-loop / ngram fallback speculative streaming."""
    model_name = _resolve_hermes_model_name(body)
    backend = _get_local_verify_backend(
        model_name=model_name,
        n_gpu_layers=_local_llama_cache_n_gpu_layers,
    )
    if backend is None:
        raise RuntimeError("local_verify_loop_unavailable")
    backend._ensure_loop()
    loop = backend._loop
    loop.stats = type(loop.stats)()
    prompt = _build_prompt_from_messages(body.get("messages", []), model_name=model_name)
    max_tokens = int(body.get("max_tokens", 32) or 32)
    created = int(time.time())
    cid = f"local_verify_{created}"
    decoder = codecs.getincrementaldecoder("utf-8")()
    accepted_tokens = 0
    emitted_tokens = 0
    num_draft = max(int((draft_policy.get("mtp_runtime") or {}).get("draft_n_tokens") or 1), 1)
    sanitizer = _ReasoningTagFilter() if _should_strip_reasoning_tags(model_name) else None

    try:
        runtime_request = loop.begin_request(runtime_unit_plan)
        metrics["runtime_request"] = runtime_request
        first = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": body.get("model", "local-verify-loop"),
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n".encode()

        first_token_id = loop.prefill(prompt)
        raw_piece = loop.llm.detokenize([first_token_id])
        piece = decoder.decode(raw_piece if isinstance(raw_piece, (bytes, bytearray)) else str(raw_piece).encode("utf-8"))
        if sanitizer is not None:
            piece = sanitizer.feed(piece)
        if piece:
            emitted_tokens += 1
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": body.get("model", "local-verify-loop"),
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        for token_id, from_draft in loop.generate(max_tokens=max_tokens - 1, num_draft=num_draft):
            if from_draft:
                accepted_tokens += 1
            raw_piece = loop.llm.detokenize([token_id])
            piece = decoder.decode(raw_piece if isinstance(raw_piece, (bytes, bytearray)) else str(raw_piece).encode("utf-8"))
            if sanitizer is not None:
                piece = sanitizer.feed(piece)
            if not piece:
                continue
            emitted_tokens += 1
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": body.get("model", "local-verify-loop"),
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        final_piece = decoder.decode(b"", final=True)
        if sanitizer is not None:
            final_piece = f"{sanitizer.feed(final_piece)}{sanitizer.finish()}"
        if final_piece:
            emitted_tokens += 1
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": body.get("model", "local-verify-loop"),
                "choices": [{"index": 0, "delta": {"content": final_piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        done = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": body.get("model", "local-verify-loop"),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        metrics.update(
            {
                "speculative": True,
                "hit": accepted_tokens > 0,
                "spec_elapsed_ms": float(loop.stats.draft_ms_total or 0.0),
                "local_ttft_ms": float(loop.stats.prefill_ms or 0.0),
                "cloud_ttft_ms": float(loop.stats.prefill_ms or 0.0),
                "accepted_tokens": int(accepted_tokens),
                "draft_tokens": int(loop.stats.draft_tokens or 0),
                "draft_n_tokens": int(num_draft),
                "executor": "verify_loop",
                "emitted_tokens": int(emitted_tokens),
                "runtime_request": loop.runtime_request_snapshot(),
            }
        )
    except Exception as e:
        metrics.update(
            {
                "speculative": True,
                "hit": False,
                "executor": "verify_loop",
                "error": str(e),
                "runtime_request": loop.runtime_request_snapshot(),
            }
        )
        err = {"error": {"message": f"local verify-loop error: {e}", "type": "local_verify_error"}}
        yield f"data: {json.dumps(err)}\n\n".encode()


def _local_mtp_runtime_unit_plan(
    expert_session: Optional[ExpertRequestSession],
) -> dict[str, Any]:
    if expert_session is None or getattr(expert_session, "plan", None) is None:
        return _expert_data_plane.export_runtime_unit_plan()
    return _expert_data_plane.export_runtime_unit_plan(plan=expert_session.plan)


def _runtime_unit_plan_model_format(runtime_unit_plan: Optional[dict[str, Any]]) -> str:
    plan = dict(runtime_unit_plan or {})
    model_name = str(plan.get("model") or "").strip().lower()
    if "gguf" in model_name:
        return "gguf"
    if "mlx" in model_name:
        return "mlx"
    if model_name:
        return "safetensors"
    return ""


def _local_mtp_unified_runtime_ir(
    *,
    body: dict,
    draft_policy: dict[str, Any],
    runtime_unit_plan: Optional[dict[str, Any]],
) -> dict[str, Any]:
    plan = dict(runtime_unit_plan or {})
    model_name = _resolve_hermes_model_name(body)
    model_family = str(plan.get("family") or ("gemma4" if "gemma4" in model_name.lower() else ""))
    backend_family = str(
        os.environ.get("EDGE_LOCAL_RUNTIME_BACKEND_FAMILY")
        or os.environ.get("EDGE_ACTIVE_BACKEND_FAMILY")
        or ("mlx" if model_family == "gemma4" else "auto")
    ).strip()
    runtime_backend = str(
        os.environ.get("EDGE_LOCAL_RUNTIME_BACKEND")
        or ("turbofieldfare" if model_family == "gemma4" and backend_family == "mlx" else "")
    ).strip()
    adapter_name = str(
        os.environ.get("EDGE_LOCAL_RUNTIME_ADAPTER")
        or ("gemma4_a4b" if model_family == "gemma4" and runtime_backend == "turbofieldfare" else "")
    ).strip()
    draft_n_tokens = max(int((draft_policy.get("mtp_runtime") or {}).get("draft_n_tokens") or 1), 1)
    return build_unified_runtime_ir_v0(
        request_id=str(plan.get("frontier_key") or f"local-mtp-{int(time.time())}"),
        runtime_unit_plan=plan,
        model_id=str(plan.get("model") or model_name or "local-verify-loop"),
        model_family=model_family,
        model_format=_runtime_unit_plan_model_format(plan),
        architecture="moe_decoder" if model_family == "gemma4" else "",
        runtime_mode="local_verify_loop",
        execution_intent="streaming_decode",
        backend_family=backend_family,
        runtime_backend=runtime_backend,
        adapter_name=adapter_name,
        device_class="apple_silicon",
        platform="macos",
        strategy_family="verify_loop",
        speculative_mode="mtp" if draft_n_tokens > 0 else "none",
        max_tokens=int(body.get("max_tokens", 32) or 32),
        stream=bool(body.get("stream", True)),
        residency_policy_family="tiered_streaming" if bool(plan.get("enabled")) else "bypass",
        target_tier="ram",
        prefetch_semantics="best_effort" if bool(plan.get("enabled")) else "noop",
        bootstrap_semantics="decode_preprime" if bool(plan.get("enabled")) else "none",
        required_capabilities=["streaming_expert_units"] if bool(plan.get("enabled")) else [],
        optional_capabilities=["decode_preprime", "runtime_unit_plan"],
    )


def _local_verify_loop_stream_with_runtime_plan(
    body: dict,
    *,
    draft_policy: dict[str, Any],
    metrics: dict[str, Any],
    expert_session: Optional[ExpertRequestSession],
):
    runtime_unit_plan = _local_mtp_runtime_unit_plan(expert_session)
    runtime_request_payload = _local_mtp_unified_runtime_ir(
        body=body,
        draft_policy=draft_policy,
        runtime_unit_plan=runtime_unit_plan,
    )
    metrics["runtime_unit_plan"] = runtime_unit_plan
    metrics["unified_runtime_ir"] = runtime_request_payload
    metrics["local_executor"] = "verify_loop"
    metrics["local_executor_control_plane"] = "expert_data_plane"
    metrics["local_executor_runtime"] = "local_mtp"
    metrics["local_executor_mode"] = str(runtime_unit_plan.get("mode") or "bypass")
    # #region debug-point B:stream-runtime-plan
    try:
        _dbg_p = os.path.join(REPO_ROOT, ".dbg", "dense-streaming-measure.env")
        _dbg_u, _dbg_s = "http://127.0.0.1:7777/event", "dense-streaming-measure"
        try:
            with open(_dbg_p, "r", encoding="utf-8") as _dbg_f:
                _dbg_c = _dbg_f.read()
            _dbg_u = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SERVER_URL=")), _dbg_u)
            _dbg_s = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SESSION_ID=")), _dbg_s)
        except Exception:
            pass
        urllib.request.urlopen(
            urllib.request.Request(
                _dbg_u,
                data=json.dumps({
                    "sessionId": _dbg_s,
                    "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                    "hypothesisId": "B",
                    "location": "app/servers/edge_first_proxy.py:_local_verify_loop_stream_with_runtime_plan",
                    "msg": "[DEBUG] dispatched stream request into local_mtp verify loop",
                    "data": {
                        "executor_runtime": "local_mtp",
                        "runtime_plan_mode": str(runtime_unit_plan.get("mode") or ""),
                        "runtime_plan_enabled": bool(runtime_unit_plan.get("enabled")),
                        "runtime_plan_reason": str(runtime_unit_plan.get("reason") or ""),
                        "current_units": len(runtime_unit_plan.get("current") or []),
                        "next_units": len(runtime_unit_plan.get("next") or []),
                    },
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.35,
        ).read()
    except Exception:
        pass
    # #endregion
    yield from _local_verify_loop_stream(
        body,
        draft_policy=draft_policy,
        metrics=metrics,
        runtime_unit_plan=runtime_request_payload,
    )


def _local_verify_loop_complete_with_runtime_plan(
    body: dict,
    *,
    draft_policy: dict[str, Any],
    metrics: dict[str, Any],
    expert_session: Optional[ExpertRequestSession],
) -> dict[str, Any]:
    runtime_unit_plan = _local_mtp_runtime_unit_plan(expert_session)
    runtime_request_payload = _local_mtp_unified_runtime_ir(
        body=body,
        draft_policy=draft_policy,
        runtime_unit_plan=runtime_unit_plan,
    )
    metrics["runtime_unit_plan"] = runtime_unit_plan
    metrics["unified_runtime_ir"] = runtime_request_payload
    metrics["local_executor"] = "verify_loop"
    metrics["local_executor_control_plane"] = "expert_data_plane"
    metrics["local_executor_runtime"] = "local_mtp"
    metrics["local_executor_mode"] = str(runtime_unit_plan.get("mode") or "bypass")
    # #region debug-point B:complete-runtime-plan
    try:
        _dbg_p = os.path.join(REPO_ROOT, ".dbg", "dense-streaming-measure.env")
        _dbg_u, _dbg_s = "http://127.0.0.1:7777/event", "dense-streaming-measure"
        try:
            with open(_dbg_p, "r", encoding="utf-8") as _dbg_f:
                _dbg_c = _dbg_f.read()
            _dbg_u = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SERVER_URL=")), _dbg_u)
            _dbg_s = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SESSION_ID=")), _dbg_s)
        except Exception:
            pass
        urllib.request.urlopen(
            urllib.request.Request(
                _dbg_u,
                data=json.dumps({
                    "sessionId": _dbg_s,
                    "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                    "hypothesisId": "B",
                    "location": "app/servers/edge_first_proxy.py:_local_verify_loop_complete_with_runtime_plan",
                    "msg": "[DEBUG] dispatched non-stream request into local_mtp verify loop",
                    "data": {
                        "executor_runtime": "local_mtp",
                        "runtime_plan_mode": str(runtime_unit_plan.get("mode") or ""),
                        "runtime_plan_enabled": bool(runtime_unit_plan.get("enabled")),
                        "runtime_plan_reason": str(runtime_unit_plan.get("reason") or ""),
                        "current_units": len(runtime_unit_plan.get("current") or []),
                        "next_units": len(runtime_unit_plan.get("next") or []),
                    },
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.35,
        ).read()
    except Exception:
        pass
    # #endregion

    model_name = _resolve_hermes_model_name(body)
    backend = _get_local_verify_backend(
        model_name=model_name,
        n_gpu_layers=_local_llama_cache_n_gpu_layers,
    )
    if backend is None:
        raise RuntimeError("local_verify_loop_unavailable")

    prompt = _build_prompt_from_messages(body.get("messages", []), model_name=model_name)
    max_tokens = int(body.get("max_tokens", 32) or 32)
    num_draft = max(int((draft_policy.get("mtp_runtime") or {}).get("draft_n_tokens") or 1), 1)
    result = backend.run(
        prompt,
        max_tokens=max_tokens,
        num_draft=num_draft,
        runtime_unit_plan=runtime_unit_plan,
    )
    text = _sanitize_output_text_for_model(str(result.get("text") or ""), model_name=model_name)
    metrics["runtime_request"] = result.get("runtime_request") or {}
    metrics["accepted_tokens"] = int(round(float(result.get("accept_rate") or 0.0) * float(result.get("total_tokens") or 0)))
    metrics["draft_tokens"] = int(result.get("total_tokens") or 0)
    metrics["draft_n_tokens"] = int(num_draft)
    metrics["speculative"] = True
    metrics["hit"] = bool(metrics["accepted_tokens"] > 0)
    metrics["spec_elapsed_ms"] = float(result.get("draft_ms") or 0.0)
    metrics["local_ttft_ms"] = float(result.get("prefill_ms") or 0.0)
    metrics["cloud_ttft_ms"] = float(result.get("prefill_ms") or 0.0)
    metrics["executor"] = "verify_loop"
    metrics["emitted_tokens"] = int(_estimate_output_tokens(text))

    return {
        "id": f"local_verify_{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "local-verify-loop"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "completion_tokens": int(_estimate_output_tokens(text)),
            "prompt_tokens": 0,
            "total_tokens": int(_estimate_output_tokens(text)),
        },
        "runtime_request": result.get("runtime_request") or {},
        "cgc_runtime": {
            "executor": "verify_loop",
            "tps": float(result.get("tps") or 0.0),
            "accept_rate": float(result.get("accept_rate") or 0.0),
            "avg_accept_len": float(result.get("avg_accept_len") or 0.0),
            "draft_ms": float(result.get("draft_ms") or 0.0),
            "verify_ms": float(result.get("verify_ms") or 0.0),
            "prefill_ms": float(result.get("prefill_ms") or 0.0),
            "mode": str(result.get("mode") or "edge_draft"),
        },
    }


def _schedule_local_startup_warmup() -> None:
    """在服務啟動後背景預熱 local_full 執行器，避免首個真請求承擔全部冷載入。"""
    if not _DEFAULT_ENABLE_LOCAL_STARTUP_WARMUP or not _LOCAL_FULL_MODEL_PATH:
        return

    with _local_warmup_lock:
        status = str(_local_warmup_state.get("status") or "idle")
        if status in {"running", "ready"}:
            return
        _local_warmup_state.update(
            {
                "status": "running",
                "started_at": time.time(),
                "finished_at": 0.0,
                "error": "",
                "llama_loaded": _local_llama_cache is not None,
                "verify_loop_ready": False,
            }
        )

    def _runner() -> None:
        status = "ready"
        error = ""
        llama_loaded = False
        verify_loop_ready = False
        try:
            llama_loaded = _load_local_llama() is not None
            if not llama_loaded:
                status = "failed"
                error = "llama_cpp_load_failed"
                return
            if not _DEFAULT_DISABLE_LOCAL_MTP:
                warmup_cfg = _resolve_local_runtime_model_cfg()
                backend = _get_local_verify_backend(
                    model_name=str(getattr(warmup_cfg, "name", "") or _ACTIVE_MODEL or "gemma4"),
                    n_gpu_layers=_local_llama_cache_n_gpu_layers,
                )
                if backend is not None:
                    backend._ensure_loop()
                    verify_loop_ready = backend._loop is not None
                    if verify_loop_ready:
                        try:
                            loop = backend._loop
                            loop.prefill("hi")
                            next(loop.generate(max_tokens=1, num_draft=1))
                        except Exception as loop_exc:
                            verify_loop_ready = False
                            status = "failed"
                            error = f"verify_loop_self_test_failed:{loop_exc}"
            else:
                error = "local_mtp_forced_disabled"
            print(
                f"[edge-first] local startup warmup done: "
                f"llama_loaded={llama_loaded} verify_loop_ready={verify_loop_ready}"
                f"{f' error={error}' if error else ''}",
                file=sys.stderr,
            )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            print(f"[edge-first] local startup warmup failed: {exc}", file=sys.stderr)
        finally:
            with _local_warmup_lock:
                _local_warmup_state.update(
                    {
                        "status": status,
                        "finished_at": time.time(),
                        "error": error,
                        "llama_loaded": llama_loaded,
                        "verify_loop_ready": verify_loop_ready,
                    }
                )

    threading.Thread(target=_runner, daemon=True).start()


# === MLX 層切分(layer-split 實作) ===
# Mac 跑前 P 層 forward 後 emit hidden_P, 雲 resume from layer P。
# V2(自定義部分加載): 用 safetensors safe_open 只讀前 P 層權重 + override
#   config num_hidden_layers=P 只分配 P 層, 不 load 全模型(15GB > 6.6GB 會 OOM)。
#   見 _load_local_mlx_first_p_layers + _manual_sanitize_qwen3_vl_moe。
_mlx_layer_split_cache: dict[int, tuple] = {}  # P -> (model, tokenizer)
_mlx_layer_split_lock = threading.Lock()
_mac_emit_transport = None
_mac_emit_transport_lock = threading.Lock()


def _extract_layer_id_from_key(key: str):
    """從 safetensors key 提取 layer id。支援 HF 格式(model.language_model.layers.0.*)
    與 MLX 格式(language_model.model.layers.0.*)。返回 None 若非 layer key。"""
    marker = "layers."
    idx = key.find(marker)
    if idx < 0:
        return None
    rest = key[idx + len(marker):]
    num_str = rest.split(".", 1)[0]
    try:
        return int(num_str)
    except (ValueError, IndexError):
        return None


def _normalize_mlx_key(key: str):
    """將 safetensors key 正規化為 MLX 參數樹格式(language_model.model.*)。

    HF 格式:   model.language_model.layers.0.self_attn.q_proj.weight
    MLX 格式:  language_model.model.layers.0.self_attn.q_proj.weight
    返回 None 表示跳過(visual 等不相關鍵)。
    """
    if key.startswith("model.language_model."):
        return "language_model.model." + key[len("model.language_model."):]
    if key.startswith("language_model.model."):
        return key
    if key.startswith("language_model."):
        return "language_model.model." + key[len("language_model."):]
    return None


def _manual_sanitize_qwen3_vl_moe(weights: dict, P: int):
    """手動 sanitize(當 model.sanitize 因缺 lm_head 等失敗時的 fallback)。

    做兩件事(對齊 qwen3_vl_moe.Model.sanitize + qwen3_moe.Model.sanitize):
      1. 拆 MoE packed: experts.gate_up_proj → switch_mlp.{gate_proj,up_proj}.weight
         (swapaxes(-2,-1)); experts.down_proj → switch_mlp.down_proj.weight (swapaxes)
      2. 正規化 key 前綴到 language_model.model.*
    不要求 lm_head(Mac 端不需要 lm_head)。
    """
    out = {}
    for k, v in weights.items():
        nk = _normalize_mlx_key(k)
        if nk is None:
            continue
        if nk.endswith(".mlp.experts.gate_up_proj"):
            prefix = nk[: -len(".mlp.experts.gate_up_proj")]
            mid = v.shape[-1] // 2
            out[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = v[..., :mid].swapaxes(-2, -1)
            out[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = v[..., mid:].swapaxes(-2, -1)
        elif nk.endswith(".mlp.experts.down_proj"):
            prefix = nk[: -len(".mlp.experts.down_proj")]
            out[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = v.swapaxes(-2, -1)
        else:
            out[nk] = v
    return out


def _load_local_mlx_first_p_layers(P: int):
    """加載 MLX 模型前 P 層(自定義部分加載, 不 load 全模型)。

    V2: 用 safetensors 直接讀前 P 層權重, 構建只含 P 層的 MLX 模型。
    關鍵: 全模型 15GB > 6.6GB 可用會 OOM, 必須只讀前 P 層。

    流程:
      1. 讀 config.json, 覆蓋 text_config.num_hidden_layers=P (只分配 P 層內存)
      2. 構建 qwen3_vl_moe.Model(P 層) — 內存 ~2.5GB for P=8
      3. glob *.safetensors 逐 shard 讀 key 列表 → 找前 P 層 + embed_tokens + norm 的 key
         (不依賴 model.safetensors.index.json, 避免 index 損壞導致 shard 找錯)
      4. 用 safetensors.safe_open(framework="mlx") 逐 key 從對應 shard 只讀需要的(fallback: mx.load)
      5. sanitize(拆 MoE packed / 棄 visual) + nn.quantize(4-bit) + load_weights(strict=False)
      6. 不讀 lm_head / visual / layers >= P (省內存)

    失敗返回 None(回退全雲)。
    """
    with _mlx_layer_split_lock:
        if P in _mlx_layer_split_cache:
            return _mlx_layer_split_cache[P]
        if not _LAYER_SPLIT_MODEL_PATH:
            return None
        try:
            import gc
            from pathlib import Path

            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm.utils import load_config, load_tokenizer, _get_classes

            model_path = Path(_LAYER_SPLIT_MODEL_PATH)
            config = load_config(model_path)

            # 1. 覆蓋 num_hidden_layers = P (關鍵: 只分配 P 層)
            # VL 模型: text_config 嵌套; 標準 LLM: 頂層
            text_cfg = config.get("text_config")
            if not isinstance(text_cfg, dict):
                # 標準 LLM (非 VL): num_hidden_layers 在頂層
                text_cfg = config
            orig_total = int(text_cfg.get("num_hidden_layers", 0))
            if orig_total == 0:
                print("[edge-router] MLX 層切: num_hidden_layers=0, 無法部分加載",
                      file=sys.stderr)
                return None
            text_cfg["num_hidden_layers"] = P
            # 對齊 load_model: 把 text_config.quantization_config 提升到頂層
            if "quantization_config" not in config and "quantization_config" in text_cfg:
                config["quantization_config"] = text_cfg["quantization_config"]

            # 2. 構建模型 (只 P 層, 內存 ~2.5GB for P=8)
            model_class, model_args_class = _get_classes(config=config)
            model_args = model_args_class.from_dict(config)
            model = model_class(model_args)

            # 3. 掃 shard 檔案找前 P 層需要的 key + shard (不依賴 index.json)
            #    index.json 可能損壞(如 mlx-community Qwen3-VL-30B 說 13-shard 但實際 3/4),
            #    直接 glob *.safetensors 逐檔讀 key 列表, 構建 key → shard 映射。
            shard_files = sorted(model_path.glob("*.safetensors"))
            if not shard_files:
                print(f"[edge-router] MLX 層切: {model_path} 無 *.safetensors, 無法部分加載",
                      file=sys.stderr)
                return None

            use_safe_open = False  # 强制 mx.load (原生 mlx bfloat16 + quantized 格式)
            # safe_open framework="mlx" 不支持 bfloat16, framework="pt" 破坏 quantized 格式

            # 掃每個 shard 的 key (safe_open 取 key 不 load tensor; fallback mx.load 取 keys)
            key_to_shard: dict = {}
            all_keys: list = []
            if use_safe_open:
                for shard_path in shard_files:
                    try:
                        with safe_open(str(shard_path), framework="mlx") as f:
                            shard_keys = list(f.keys())
                    except Exception as e:
                        print(f"[edge-router] MLX 層切: 讀 {shard_path.name} key 失敗: {e}",
                              file=sys.stderr)
                        return None
                    for k in shard_keys:
                        key_to_shard[k] = shard_path
                    all_keys.extend(shard_keys)
            else:
                # fallback: mx.load 整個 shard 只為取 key (內存較耗, 但不需 safetensors pkg)
                for shard_path in shard_files:
                    try:
                        all_w = mx.load(str(shard_path))
                    except Exception as e:
                        print(f"[edge-router] MLX 層切: 讀 {shard_path.name} 失敗: {e}",
                              file=sys.stderr)
                        return None
                    for k in all_w.keys():
                        key_to_shard[k] = shard_path
                        all_keys.append(k)
                    del all_w
                gc.collect()

            # needed_keys: 前 P 層所有 key + embed_tokens + norm + lm_head (sanitize 用)
            needed_keys = set()
            for k in all_keys:
                lid = _extract_layer_id_from_key(k)
                if lid is not None and lid < P:
                    needed_keys.add(k)
                elif "embed_tokens" in k and "language_model" in k:
                    needed_keys.add(k)
                elif k.endswith("language_model.norm.weight") or k == "model.language_model.norm.weight":
                    needed_keys.add(k)
                elif k == "lm_head.weight" or k == "language_model.lm_head.weight":
                    # lm_head.weight: sanitize 需要它; 若在邊界 shard 外則單獨加載
                    needed_keys.add(k)

            if not needed_keys:
                print("[edge-router] MLX 層切: shard 中無匹配 key", file=sys.stderr)
                return None

            # 4. 逐 key 從對應 shard 讀取 (per-key safe_open, 只讀 needed_keys)
            #    按 shard 分組避免重複 open; fallback mx.load 過濾
            raw_weights = {}
            shard_to_keys: dict = {}
            missing_keys = []
            for k in needed_keys:
                sp = key_to_shard.get(k)
                if sp is None:
                    missing_keys.append(k)
                    continue
                shard_to_keys.setdefault(sp, []).append(k)

            if use_safe_open:
                for shard_path, keys in shard_to_keys.items():
                    with safe_open(str(shard_path), framework="pt") as f:
                        fkeys = set(f.keys())
                        for k in keys:
                            if k in fkeys:
                                raw_weights[k] = f.get_tensor(k)
                            else:
                                missing_keys.append(k)
            else:
                for shard_path, keys in shard_to_keys.items():
                    all_w = mx.load(str(shard_path))
                    for k in keys:
                        if k in all_w:
                            raw_weights[k] = all_w[k]
                        else:
                            missing_keys.append(k)
                    del all_w
                gc.collect()

            if missing_keys:
                print(f"[edge-router] MLX 層切: key 未找到 {missing_keys[:5]}"
                      f"{'...' if len(missing_keys) > 5 else ''}, 回退全雲",
                      file=sys.stderr)
                return None

            needed_shards = sorted({sp.name for sp in shard_to_keys.keys()})

            # 5a. sanitize (嘗試 model.sanitize; 失敗用手動 _manual_sanitize_qwen3_vl_moe)
            try:
                if hasattr(model, "sanitize"):
                    weights = model.sanitize(dict(raw_weights))
                else:
                    weights = dict(raw_weights)
            except Exception as sanitize_err:
                print(f"[edge-router] MLX 層切: model.sanitize 失敗({sanitize_err!r}), "
                      f"用手動 sanitize", file=sys.stderr)
                weights = _manual_sanitize_qwen3_vl_moe(raw_weights, P)

            del raw_weights
            gc.collect()

            # 5b. quantize (複製 mlx_lm.utils.load_model 的 _quantize 邏輯, 處理 4-bit)
            quantization = config.get("quantization") or config.get("quantization_config")
            if isinstance(quantization, dict) and "group_size" in quantization:
                def _class_predicate(p, m):
                    if p in quantization:
                        return quantization[p]
                    if not hasattr(m, "to_quantized"):
                        return False
                    return f"{p}.scales" in weights

                nn.quantize(
                    model,
                    group_size=quantization["group_size"],
                    bits=quantization["bits"],
                    mode=quantization.get("mode", "affine"),
                    class_predicate=_class_predicate,
                )

            # 5c. load_weights (strict=False: lm_head/未加載的保持初始值, 不報錯)
            model.eval()
            model.load_weights(list(weights.items()), strict=False)

            del weights
            gc.collect()

            # 6. tokenizer
            tokenizer = load_tokenizer(model_path)

            _mlx_layer_split_cache[P] = (model, tokenizer)
            print(f"[edge-router] MLX 層切 P={P} 部分加載成功 "
                  f"(layers={P}/{orig_total}, shards={len(needed_shards)}, "
                  f"use_safe_open={use_safe_open})", file=sys.stderr)
            return model, tokenizer
        except Exception as e:
            print(f"[edge-router] MLX 層切 P={P} 部分加載失敗: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None


def _mlx_forward_first_p_layers(model, tokenizer, messages: list, P: int):
    """Mac MLX forward 前 P 層, 返回 (hidden_P, residual, input_ids, seq_len, kv_layers)。

    手動 forward: embed_tokens → layers[:P], 取最後一層輸出(未 norm)。
    適配 Qwen3/Llama 架構。hidden_P 轉 torch tensor 以便 transport 序列化。

    KV 捕獲 (CGC layer-split KV 注入): monkey-patch Attention.__call__,
    前 P 層 EXTEND (cache is None) 時捕獲 RoPE 後 keys + 原始 values,
    轉 torch (seq, n_kv_heads*head_dim) 對齊 sglang set_kv_buffer 期望格式。
    forward 完還原原 __call__ (try/finally 保證還原)。

    支援 qwen3_vl_moe.Model(有 .language_model.model)與 qwen3_moe.Model(有 .model)
    兩種包裝: 自動偵測 language_model / model 路徑取 embed_tokens + layers。
    """
    import mlx.core as mx
    import numpy as np
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # 对齐 sglang VL wrapper: 过滤 <|im_start|>(151644) 和 <|im_end|>(151645)
    # sglang Qwen3-VL 在 VL wrapper 层去掉这些 special tokens (10→7 tokens),
    # Mac 也需对齐, 否则 Mac seq(10) != cloud seq(7) → KV shape 不匹配 → 注入失败
    _CGC_SPECIAL_FILTER = {151644, 151645}
    _filtered = [t for t in input_ids if t not in _CGC_SPECIAL_FILTER]
    if len(_filtered) < len(input_ids):
        print(
            f"[edge-router] tokenizer 对齐: 过滤 special tokens "
            f"{len(input_ids)}→{len(_filtered)} (去掉 <|im_start|>/<|im_end|>)",
            file=sys.stderr,
        )
    input_ids = _filtered
    input_ids_mx = mx.array([input_ids])
    seq_len = len(input_ids)

    # 自動偵測模型包裝層:
    #   qwen3_vl_moe.Model  → model.language_model.model.{embed_tokens, layers}
    #   qwen3_moe.Model     → model.model.{embed_tokens, layers}
    lang = getattr(model, "language_model", None)
    if lang is not None:
        inner = getattr(lang, "model", lang)
    else:
        inner = getattr(model, "model", model)

    embed = getattr(inner, "embed_tokens", None)
    layers = getattr(inner, "layers", None)
    if embed is None or layers is None:
        raise RuntimeError(
            f"無法定位 embed_tokens/layers: model 類型={type(model).__name__}, "
            f"有 language_model={lang is not None}"
        )

    h = embed(input_ids_mx)

    # 建 causal mask (prefill)。mlx-lm 內建 create_attention_mask / create_causal_mask。
    # 優先用 create_attention_mask(h, None)(對齊模型 __call__); 失敗用 None。
    mask = None
    try:
        from mlx_lm.models.base import create_attention_mask
        mask = create_attention_mask(h, None)
    except Exception:
        try:
            from mlx_lm.models.base import create_causal_mask
            try:
                mask = create_causal_mask(seq_len)
            except TypeError:
                mask = create_causal_mask(seq_len, dtype=h.dtype)
        except Exception:
            pass

    # 逐層 forward 前 P 層。Qwen3MoeDecoderLayer.__call__(x, mask, cache)。
    # Qwen3 MLX layer 可能返回 (hidden, residual) tuple 或只 hidden。
    # 用 try 適配 (x, mask) / (x, mask, cache) / (x, mask, cache, None) 簽名。
    n_layers = min(P, len(layers))

    # === KV 捕獲: monkey-patch Attention.__call__ (前 P 層 RoPE 後 K/V) ===
    # 調原 __call__ 取正常輸出;另算 RoPE 後 K + 原始 V 捕獲 (對齊 sglang KV cache)。
    # 只在 EXTEND (cache is None) 捕獲;forward 完 try/finally 還原 __call__。
    _attn_cls = type(layers[0].self_attn)
    _orig_attn_call = _attn_cls.__call__
    _captured_kv_local: dict[int, tuple] = {}  # layer_idx -> (K_mx, V_mx)

    def _capturing_attn_call(self_attn, x, mask=None, cache=None):
        out = _orig_attn_call(self_attn, x, mask, cache)
        _idx = getattr(self_attn, "_cap_layer_idx", -1)
        if _idx >= 0 and cache is None:
            # 另算 RoPE 後 K + 原始 V (對齊 sglang KV cache 存儲)
            B, L, _D = x.shape
            keys = self_attn.k_proj(x)
            values = self_attn.v_proj(x)
            if hasattr(self_attn, "k_norm"):
                keys = self_attn.k_norm(
                    keys.reshape(B, L, self_attn.n_kv_heads, -1)
                ).transpose(0, 2, 1, 3)
            else:
                keys = keys.reshape(
                    B, L, self_attn.n_kv_heads, -1
                ).transpose(0, 2, 1, 3)
            values = values.reshape(
                B, L, self_attn.n_kv_heads, -1
            ).transpose(0, 2, 1, 3)
            keys = self_attn.rope(keys)  # cache is None, 無 offset
            _captured_kv_local[_idx] = (keys, values)
        return out

    # install: 標記每層 layer_idx + 類級 patch
    for i in range(n_layers):
        layers[i].self_attn._cap_layer_idx = i
    _attn_cls.__call__ = _capturing_attn_call

    _last_residual = None
    try:
        for i in range(n_layers):
            layer = layers[i]
            try:
                out = layer(h, mask)
            except TypeError:
                try:
                    out = layer(h, mask, None)
                except TypeError:
                    out = layer(h, mask, None, None)
            if isinstance(out, tuple):
                h, _last_residual = out
            else:
                h = out
    finally:
        # uninstall: 還原 __call__ + 清 layer_idx 標記
        _attn_cls.__call__ = _orig_attn_call
        for i in range(n_layers):
            try:
                delattr(layers[i].self_attn, "_cap_layer_idx")
            except AttributeError:
                pass

    # h: mlx tensor [1, seq_len, hidden_dim] → numpy → torch CPU
    # bfloat16 不支持 numpy buffer, 先转 float32
    h_np = np.array(h.astype(mx.float32))
    h_torch = torch.from_numpy(h_np)
    # residual 转 torch (mlx bfloat16 → float32 → numpy → torch); None 则 None
    if _last_residual is not None:
        r_np = np.array(_last_residual.astype(mx.float32))
        r_torch = torch.from_numpy(r_np)
    else:
        r_torch = None

    # === 轉換 KV: MLX (B, n_kv_heads, L, head_dim) → torch (B*L, n_kv_heads*head_dim) ===
    # 對齊 sglang set_kv_buffer 期望的 (seq, row_dim=n_kv_heads*head_dim) 格式。
    # dtype: mlx float32 → numpy → torch float32 (cloud 端 set_kv_buffer 會自動轉 self.dtype)
    kv_layers: list = []
    for i in range(n_layers):
        if i not in _captured_kv_local:
            kv_layers.append(None)
            continue
        k_mx, v_mx = _captured_kv_local[i]
        # (B, n_kv_heads, L, head_dim) → transpose(0,2,1,3) → (B, L, n_kv_heads, head_dim)
        # → reshape(B*L, n_kv_heads*head_dim)
        _B, nkv, _L, hd = k_mx.shape
        k_np = np.array(
            k_mx.astype(mx.float32).transpose(0, 2, 1, 3).reshape(-1, nkv * hd)
        )
        v_np = np.array(
            v_mx.astype(mx.float32).transpose(0, 2, 1, 3).reshape(-1, nkv * hd)
        )
        kv_layers.append({
            "k": torch.from_numpy(k_np),
            "v": torch.from_numpy(v_np),
        })

    _cap_count = sum(1 for x in kv_layers if x is not None)
    print(
        f"[edge-router] KV 捕獲 P={n_layers} 成功 {_cap_count}/{n_layers} 層 "
        f"(每層 K shape={list(kv_layers[0]['k'].shape) if kv_layers and kv_layers[0] else 'N/A'})",
        file=sys.stderr,
    )

    return h_torch, r_torch, input_ids, seq_len, kv_layers


def _get_mac_emit_transport():
    """獲取 Mac→cloud 反向 transport(MacEmitHandoff, role=emitter)。延遲構造。

    復用 CGC_Phase2.cgc_handoff_transport.HandoffTransport.make("mac_emit")。
    雲端 receiver 需另起 MacEmitHandoff(role="receiver") 並在 /v1/cgc/resume
    端點調 transport.recv(rank, step) 取 hidden_P。
    """
    global _mac_emit_transport
    with _mac_emit_transport_lock:
        if _mac_emit_transport is not None:
            return _mac_emit_transport
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from CGC_Phase2.cgc_handoff_transport import HandoffTransport
        mode = os.environ.get("CGC_LAYER_SPLIT_TRANSPORT", "mac_emit")
        _mac_emit_transport = HandoffTransport.make(mode, role="emitter")
        return _mac_emit_transport


async def _layer_split_stream(body: dict, P: int, cloud_endpoint: str, cloud_payload: dict, headers: dict):
    """LayerSplit PD: Mac prefill 前 P 層 + 雲 resume 後 L-P 層 + MTP 投機。

    實作(本會話):
      1. MLX 加載前 P 層權重(層切分, _load_local_mlx_first_p_layers)
      2. Mac prefill 前 P 層 → hidden_P(_mlx_forward_first_p_layers)
      3. emit hidden_P via MacEmitHandoff(Mac→cloud TCP PUT, 反向對稱 NixlHandoff)
      4. HTTP POST cloud /v1/cgc/resume(攜帶 request_id/rank/step/P/seq_len)
         → 雲 resume server 調 transport.recv(rank,step) 取 hidden_P
         → 注入 KV cache, 從 layer P resume forward + lm_head + MTP decode
      5. 轉發雲端 token stream 給客戶端

    失敗回退全雲(MLX 加載失敗 / forward 失敗 / transport 失敗)。

    決策樹路徑: 標準 LLM(架構支持) + 顯存夠部分層 + 時延預判通過。
    V1 僅 prefill 層切(Mac 參與 prefill 省 cloud 算力); decode 全雲(DSV4+MTP 37tok/s)。
    V2(per-step decode emit)待 Mac decode 性能驗證後啟用。
    """
    # 1. MLX 加載前 P 層
    model_tuple = _load_local_mlx_first_p_layers(P)
    if model_tuple is None:
        print(f"[edge-router] layer-split P={P} MLX 加載失敗, 回退全雲",
              file=sys.stderr)
        async for chunk in _cloud_stream(cloud_endpoint, cloud_payload, headers):
            yield chunk
        return

    model, tokenizer = model_tuple
    messages = body.get("messages", [])

    try:
        # 2. Mac prefill 前 P 層 → hidden_P + residual_P + kv_layers (CGC KV 注入)
        hidden_P, residual_P, input_ids, seq_len, kv_layers = _mlx_forward_first_p_layers(
            model, tokenizer, messages, P
        )
        # 3. emit hidden_P + kv_layers via reverse transport (Mac→cloud)
        transport = _get_mac_emit_transport()
        rank = 0
        step = 0  # prefill step
        request_id = hashlib.sha1(
            f"{time.time()}.{seq_len}.{P}".encode()
        ).hexdigest()[:12]
        payload = {
            "finished_layer": P,
            "hidden_states": hidden_P,
            "residual": residual_P,
            "step": step,
            "request_id": request_id,
            "input_ids": input_ids,
            "seq_len": seq_len,
            "model": body.get("model", ""),
            # === CGC KV 注入: 前 P 層 RoPE 後 K/V,cloud 端 set_kv_buffer 注入 ===
            # kv_layers: list[dict] 長度 P, 每項 {"k": torch(seq,nkv*hd), "v": torch(...)}
            # None 項表該層捕獲失敗 (cloud 端會跳過, 該層 KV 維持空)
            "kv_layers": kv_layers,
        }
        transport.send(rank, step, payload)
        _kv_ok = sum(1 for x in kv_layers if x is not None)
        print(
            f"[edge-router] layer-split emit hidden_P P={P} "
            f"shape={list(hidden_P.shape)} req={request_id} seq={seq_len} "
            f"kv_layers={_kv_ok}/{len(kv_layers)}",
            file=sys.stderr,
        )

        # 4. HTTP POST cloud resume endpoint
        #    cloud /v1/cgc/resume 契約(待雲端實作):
        #      - 收到 HTTP 請求 → 調 MacEmitHandoff(role="receiver").recv(rank, step)
        #      - 取 hidden_P + input_ids → 重建 layers P..L-1 的 KV cache
        #      - 從 layer P resume forward + norm + lm_head + MTP decode
        #      - streaming 返回 token (OpenAI chunk 格式)
        resume_url = cloud_endpoint.replace(
            "/v1/chat/completions", "/v1/cgc/resume"
        )
        resume_payload = {
            **cloud_payload,
            "stream": True,
            "extra_body": {
                **dict(cloud_payload.get("extra_body") or {}),
                "cgc_resume_from_layer": P,
                "cgc_resume_request_id": request_id,
                "cgc_resume_rank": rank,
                "cgc_resume_step": step,
                "cgc_resume_seq_len": seq_len,
                "cgc_resume_finished_layer": P,
            },
        }

        # 5. 轉發雲端 stream
        async for chunk in _cloud_stream(resume_url, resume_payload, headers):
            yield chunk

    except Exception as e:
        print(f"[edge-router] layer-split P={P} 失敗: {e!r}, 回退全雲",
              file=sys.stderr)
        import traceback
        traceback.print_exc()
        err = {
            "error": {
                "message": f"layer_split error: {e}",
                "type": "layer_split_error",
                "fallback": "cloud_full",
            }
        }
        yield f"data: {json.dumps(err)}\n\n".encode()
        async for chunk in _cloud_stream(cloud_endpoint, cloud_payload, headers):
            yield chunk


app = FastAPI(title="Edge-First Proxy")


@app.on_event("startup")
async def _startup_warmup():
    """背景預熱本地執行器，不阻塞 HTTP 服務啟動。"""
    _schedule_local_startup_warmup()


@app.on_event("shutdown")
async def _shutdown_cleanup():
    """关闭 aiohttp 连接池。"""
    global _aiohttp_session
    if _aiohttp_session is not None and not _aiohttp_session.closed:
        await _aiohttp_session.close()
        _aiohttp_session = None


@app.get("/health")
def health():
    _tracker = get_acceptance_tracker()
    edge_model_loaded = bool(_edge_llm is not None or _local_llama_cache is not None)
    local_verify_loop_ready = any(
        getattr(backend, "_loop", None) is not None
        for backend in _local_verify_backends.values()
    )
    return {
        "status": "ok",
        "edge_first_enabled": os.environ.get("EDGE_FIRST_ENABLED", "1") == "1",
        "edge_model_loaded": edge_model_loaded,
        "local_verify_loop_ready": local_verify_loop_ready,
        "cloud_url": os.environ.get("CLOUD_URL", ""),
        "edge_speculation_min_confidence": _DEFAULT_EDGE_SPECULATION_MIN_CONFIDENCE,
        "warmup_enabled": _DEFAULT_ENABLE_WARMUP,
        "warmup_cache_size": len(_warmup_state),
        "local_startup_warmup_enabled": _DEFAULT_ENABLE_LOCAL_STARTUP_WARMUP,
        "local_startup_warmup": dict(_local_warmup_state),
        "cache_sizes": {
            "l1_exact": len(_first_token_cache),
            "l2_prefix": len(_prefix_token_cache),
            "l3_tail": len(_tail_token_cache),
            "l4_semantic": len(_semantic_cache),
        },
        # temperature>0 适配配置
        "temperature_adaptation": {
            "topk_size": _TEMP_TOPK_SIZE,
            "disable_threshold": _TEMP_SPECULATION_DISABLE_THRESHOLD,
            "bands": [_TEMP_BAND_DETERMINISTIC, _TEMP_BAND_LOW, _TEMP_BAND_MEDIUM, _TEMP_BAND_HIGH],
        },
        # AcceptanceTracker 状态
        "acceptance_tracker": _tracker.get_status(),
        # Draft Registry 状态
        "draft_registry": get_draft_registry().get_status(),
        # 四態路由配置
        "router": _transport_runtime_snapshot(),
        "expert_data_plane": _expert_data_plane.runtime_snapshot(),
    }


@app.get("/stats")
def stats():
    """缓存命中率 + 投机准确率 + TTFT 分布统计 + AcceptanceTracker 状态。"""
    result = _get_stats()
    result["acceptance_tracker"] = get_acceptance_tracker().get_status()
    return result


@app.post("/stats/reset")
def reset_stats():
    """重置统计 (测试用)。"""
    global _stats, _draft_mode_stats
    with _stats_lock:
        _stats = _new_stats_state()
    with _draft_mode_stats_lock:
        _draft_mode_stats = {}
    get_acceptance_tracker().reset()
    _persist_live_report_snapshots()
    return {"status": "reset"}


@app.get("/acceptance")
def acceptance_status():
    """AcceptanceTracker 完整状态 — 三态状态机 + per-family accept rate。"""
    return get_acceptance_tracker().get_status()


@app.post("/acceptance/reset")
def acceptance_reset():
    """重置 AcceptanceTracker (测试用)。"""
    get_acceptance_tracker().reset()
    _persist_live_report_snapshots()
    return {"status": "reset", "tracker": get_acceptance_tracker().get_status()}


@app.get("/report-contracts")
def report_contracts_endpoint():
    """四份 JSON 报告契约."""
    return route_report_contracts()


@app.get("/report-snapshots/route_policy_v2")
def report_snapshot_route_policy_v2():
    with _report_snapshot_lock:
        return dict(_last_route_policy_snapshot)


@app.get("/report-snapshots/route_heat_snapshot")
def report_snapshot_route_heat():
    with _report_snapshot_lock:
        return dict(_last_heat_snapshot)


@app.get("/report-snapshots/draft_mode_acceptance")
def report_snapshot_draft_mode_acceptance():
    return _draft_mode_acceptance_snapshot()


@app.get("/report-snapshots/single_node_candidate_matrix")
def report_snapshot_single_node_candidate_matrix():
    with _report_snapshot_lock:
        return dict(_last_single_node_candidate_matrix)


@app.get("/expert-data-plane")
def expert_data_plane_snapshot():
    return _expert_data_plane.runtime_snapshot()


@app.post("/expert-data-plane/reload")
def expert_data_plane_reload():
    snapshot = _expert_data_plane.reload_catalog()
    _persist_live_report_snapshots()
    return {"status": "reloaded", "expert_data_plane": snapshot}


@app.post("/expert-data-plane/reset")
async def expert_data_plane_reset(request: Request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    snapshot = _expert_data_plane.reset_runtime(
        drop_resident=bool(body.get("drop_resident", True)),
        drop_affinity=bool(body.get("drop_affinity", False)),
    )
    _persist_live_report_snapshots()
    return {"status": "reset", "expert_data_plane": snapshot}


@app.post("/expert-data-plane/pin")
async def expert_data_plane_pin(request: Request):
    body = await request.json()
    keys = [str(key) for key in (body.get("keys") or []) if str(key).strip()]
    result = _expert_data_plane.set_pin_state(keys, pinned=bool(body.get("pinned", True)))
    _persist_live_report_snapshots()
    return result


@app.post("/route-test")
async def route_test(request: Request):
    """調試端點: 對請求做路由決策, 返回 mode/P/reason(不實際推理)。"""
    body = await request.json()
    transport_route = _align_transport_route_with_local_runtime(_build_transport_route_context(body))
    transport_runtime = _augment_transport_runtime_snapshot()
    family_info = _attach_frontier_context(_classify_prompt_family(body.get("messages", [])), body, request)
    expert_plan = _preview_request_for_route(
        body=body,
        family_info=family_info,
        route_mode=str(transport_route.get("mode") or ROUTE_CLOUD_PD),
    )
    transport_route = _align_transport_route_with_runtime_unit_plan(transport_route, expert_plan)
    final_preview_mode = str(transport_route.get("mode") or ROUTE_CLOUD_PD)
    if final_preview_mode != str((expert_plan.get("runtime_unit_plan") or {}).get("route_mode") or ""):
        expert_plan = _preview_request_for_route(
            body=body,
            family_info=family_info,
            route_mode=final_preview_mode,
        )
        transport_route = _align_transport_route_with_runtime_unit_plan(transport_route, expert_plan)
    transport_debug = _transport_debug_snapshot(body, transport_route)
    draft_policy, hermes_feature_schema, hermes_policy = _build_draft_policy_from_hermes(body, family_info, transport_route)
    return {
        "transport_runtime": transport_runtime,
        **transport_debug,
        "final_route_mode": str((hermes_policy or {}).get("mode") or ROUTE_CLOUD_PD),
        "expert_data_plane": expert_plan,
        "draft_policy": draft_policy,
        "hermes_feature_schema": hermes_feature_schema or {},
        "hermes_policy": hermes_policy or {},
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Edge-first streaming：本地首 token + 雲端接續。"""
    body = await request.json()
    stream = body.get("stream", False)
    extra_body = dict(body.get("extra_body") or {})
    route_override = str(
        body.get("cgc_route_override")
        or extra_body.get("cgc_route_override")
        or ""
    ).strip()
    route_override_reason = str(
        body.get("cgc_route_override_reason")
        or extra_body.get("cgc_route_override_reason")
        or ""
    ).strip()
    handoff_source = str(
        body.get("cgc_handoff_source")
        or extra_body.get("cgc_handoff_source")
        or ""
    ).strip()
    route_override_p = int(
        body.get("cgc_route_override_pivot_layer")
        or extra_body.get("cgc_route_override_pivot_layer")
        or 0
    )
    cloud_url = os.environ.get("CLOUD_URL", "http://127.0.0.1:30000")
    cloud_endpoint = f"{cloud_url.rstrip('/')}/v1/chat/completions"
    messages = body.get("messages", [])
    family_info = _attach_frontier_context(_classify_prompt_family(messages), body, request)
    cloud_payload = _prepare_cloud_payload(body, family_info)
    request_headers = _cloud_request_headers(request.headers, family_info)
    edge_enabled = os.environ.get("EDGE_FIRST_ENABLED", "1") == "1"

    # 四態路由(按架構圖決策樹): local_full / layer_split_pd / cloud_pd / cloud_fallback
    transport_route = _align_transport_route_with_local_runtime(_build_transport_route_context(body))
    transport_runtime = _augment_transport_runtime_snapshot()
    preview_plan = _preview_request_for_route(
        body=body,
        family_info=family_info,
        route_mode=str(transport_route.get("mode") or ROUTE_CLOUD_PD),
    )
    transport_route = _align_transport_route_with_runtime_unit_plan(transport_route, preview_plan)
    final_preview_mode = str(transport_route.get("mode") or ROUTE_CLOUD_PD)
    if final_preview_mode != str((preview_plan.get("runtime_unit_plan") or {}).get("route_mode") or ""):
        preview_plan = _preview_request_for_route(
            body=body,
            family_info=family_info,
            route_mode=final_preview_mode,
        )
        transport_route = _align_transport_route_with_runtime_unit_plan(transport_route, preview_plan)
    draft_policy, hermes_feature_schema, hermes_policy = _build_draft_policy_from_hermes(body, family_info, transport_route)
    route_mode = str((hermes_policy or {}).get("mode") or ROUTE_CLOUD_PD)
    route_P = int((hermes_policy or {}).get("pivot_layer") or 0)
    route_reason = str((hermes_policy or {}).get("reason") or "")
    if route_override in {ROUTE_LOCAL_FULL, ROUTE_LAYER_SPLIT_PD, ROUTE_CLOUD_PD, ROUTE_CLOUD_FALLBACK}:
        route_mode = route_override
        if route_override != ROUTE_LAYER_SPLIT_PD:
            route_P = 0
        elif route_override_p > 0:
            route_P = route_override_p
        suffix = route_override_reason or "request_override"
        route_reason = f"handoff_override:{suffix}"
    local_verify_loop_available = bool(
        route_mode == ROUTE_LOCAL_FULL
        and draft_policy.get("enabled")
        and str(draft_policy.get("draft_mode") or "") == "mtp"
        and bool((draft_policy.get("mtp_runtime") or {}).get("available"))
        and _get_local_verify_backend(model_name=_resolve_hermes_model_name(body), n_gpu_layers=_local_llama_cache_n_gpu_layers) is not None
    )
    local_verify_loop_enabled = bool(stream and local_verify_loop_available)
    if stream and route_mode == ROUTE_LOCAL_FULL and not local_verify_loop_enabled and _load_local_llama() is None:
        print("[edge-router] local_full 模式但 llama.cpp 主模型加載失敗, 回退雲", file=sys.stderr)
        route_mode = ROUTE_CLOUD_FALLBACK
        route_reason = "llama_cpp_load_failed"
    expert_session = _expert_data_plane.begin_request(
        model_name=str(body.get("model") or _ACTIVE_MODEL),
        family_info=family_info,
        draft_policy=draft_policy,
        route_mode=route_mode,
    )
    expert_session_for_stream = expert_session
    if expert_session is not None:
        plan = getattr(expert_session, "plan", None)
        if plan is None or not bool(getattr(plan, "enabled", False)):
            expert_session_for_stream = None
    # #region debug-point A:request-path-plan
    try:
        _dbg_p = os.path.join(REPO_ROOT, ".dbg", "dense-streaming-measure.env")
        _dbg_u, _dbg_s = "http://127.0.0.1:7777/event", "dense-streaming-measure"
        try:
            with open(_dbg_p, "r", encoding="utf-8") as _dbg_f:
                _dbg_c = _dbg_f.read()
            _dbg_u = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SERVER_URL=")), _dbg_u)
            _dbg_s = next((ln.split("=", 1)[1] for ln in _dbg_c.splitlines() if ln.startswith("DEBUG_SESSION_ID=")), _dbg_s)
        except Exception:
            pass
        _runtime_plan = _local_mtp_runtime_unit_plan(expert_session_for_stream)
        urllib.request.urlopen(
            urllib.request.Request(
                _dbg_u,
                data=json.dumps({
                    "sessionId": _dbg_s,
                    "runId": os.environ.get("EDGE_DEBUG_RUN_ID", "pre-fix"),
                    "hypothesisId": "A",
                    "location": "app/servers/edge_first_proxy.py:chat_completions",
                    "msg": "[DEBUG] resolved local request plan and executor path",
                    "data": {
                        "stream": bool(stream),
                        "route_mode": str(route_mode),
                        "route_reason": str(route_reason or ""),
                        "draft_mode": str(draft_policy.get("draft_mode") or ""),
                        "draft_enabled": bool(draft_policy.get("enabled")),
                        "local_verify_loop_available": bool(local_verify_loop_available),
                        "local_verify_loop_enabled": bool(local_verify_loop_enabled),
                        "runtime_plan_mode": str(_runtime_plan.get("mode") or ""),
                        "runtime_plan_enabled": bool(_runtime_plan.get("enabled")),
                        "runtime_plan_reason": str(_runtime_plan.get("reason") or ""),
                        "runtime_plan_current_units": len(_runtime_plan.get("current") or []),
                        "runtime_plan_next_units": len(_runtime_plan.get("next") or []),
                        "runtime_plan_next_next_units": len(_runtime_plan.get("next_next") or []),
                    },
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.35,
        ).read()
    except Exception:
        pass
    # #endregion
    _record_draft_mode_request(draft_policy)
    guard_overrides: list[dict[str, Any]] = []
    if not draft_policy["enabled"]:
        guard_overrides.append({
            "type": "draft_mode_auto_disable",
            "draft_mode": draft_policy["draft_mode"],
            "reason": draft_policy["disable_reason"],
        })
    if draft_policy.get("source") == "local_fallback":
        guard_overrides.append({
            "type": "hermes_fallback",
            "reason": str(draft_policy.get("reason") or "local_fallback"),
        })
    _update_route_policy_snapshot(
        feature_schema=hermes_feature_schema or {
            "route_family": str(family_info.get("family") or "generic"),
            "route_mode": route_mode,
            "prompt_hash": str(family_info.get("prompt_hash") or ""),
            "response_contract": draft_policy["response_contract"],
            "history_accept_rate": round(get_acceptance_tracker().get_accept_rate(), 3),
            "cache_hit_rate": round(_get_stats().get("cache_hit_rate", 0.0), 3),
            "grammar_mode_roi": draft_policy["roi"],
            "recent_json_success_rate": draft_policy["json_success_rate"],
        },
        final_policy={
            "route_mode": route_mode,
            "draft_mode": draft_policy["draft_mode"],
            "response_contract": draft_policy["response_contract"],
            "grammar_mode": draft_policy["grammar_mode"],
            "draft_enabled": draft_policy["enabled"],
            "route_reason": route_reason,
            "policy_source": str(draft_policy.get("source") or "unknown"),
        },
        guard_overrides=guard_overrides,
        hermes_policy=hermes_policy,
        transport_runtime=transport_runtime,
        transport_route=transport_route,
    )
    _update_heat_snapshot(
        family_info,
        route_mode,
        transport_runtime=transport_runtime,
        transport_route=transport_route,
    )
    request_started_at = time.monotonic()
    _record_stats(total_requests=1, family=str(family_info.get("family") or "generic"))

    if stream and route_mode == ROUTE_LOCAL_FULL:
        local_stream_metrics: dict[str, Any] = {}
        local_stream = (
            _local_verify_loop_stream_with_runtime_plan(
                body,
                draft_policy=draft_policy,
                metrics=local_stream_metrics,
                expert_session=expert_session_for_stream,
            )
            if local_verify_loop_enabled
            else _local_llama_stream(body)
        )
        return StreamingResponse(
            _instrument_passthrough_stream(
                local_stream,
                request_started_at=request_started_at,
                draft_policy=draft_policy,
                family_info=family_info,
                route_mode=ROUTE_LOCAL_FULL,
                transport_runtime=transport_runtime,
                transport_route=transport_route,
                expert_session=expert_session_for_stream,
                speculation_info=local_stream_metrics,
            ),
            media_type="text/event-stream",
            headers={
                "x-edge-router": ROUTE_LOCAL_FULL,
                "x-edge-router-reason": _header_safe(route_reason),
                "x-cgc-draft-mode": _header_safe(draft_policy["draft_mode"]),
                "x-cgc-local-executor": "verify_loop" if local_verify_loop_enabled else "llama_cpp",
                "Cache-Control": "no-cache",
            },
        )
    if stream and route_mode == ROUTE_LAYER_SPLIT_PD:
        return StreamingResponse(
            _instrument_passthrough_stream(
                _layer_split_stream(body, route_P, cloud_endpoint, cloud_payload, request_headers),
                request_started_at=request_started_at,
                draft_policy=draft_policy,
                family_info=family_info,
                route_mode=ROUTE_LAYER_SPLIT_PD,
                transport_runtime=transport_runtime,
                transport_route=transport_route,
                expert_session=expert_session_for_stream,
            ),
            media_type="text/event-stream",
            headers={
                "x-edge-router": ROUTE_LAYER_SPLIT_PD,
                "x-edge-router-P": str(route_P),
                "x-edge-router-reason": _header_safe(route_reason),
                    "x-edge-router-mac-time-est": str(transport_route.get("mac_time_est", "")),
                "Cache-Control": "no-cache",
            },
        )
    if stream and handoff_source and route_mode in {ROUTE_CLOUD_PD, ROUTE_CLOUD_FALLBACK}:
        session = await _get_cloud_session()
        resp = await session.post(
            cloud_endpoint,
            json={**cloud_payload, "stream": True},
            headers={**request_headers, "Content-Type": "application/json"},
        )
        passthrough_headers = {
            "x-edge-router": route_mode,
            "x-edge-router-reason": _header_safe(route_reason),
            "Cache-Control": "no-cache",
        }
        content_type = resp.headers.get("content-type", "text/event-stream")
        if resp.status >= 400:
            data = await resp.read()
            resp.release()
            return Response(
                content=data,
                status_code=resp.status,
                media_type=content_type,
                headers=passthrough_headers,
            )

        async def _handoff_passthrough(resp=resp):
            try:
                async for chunk in resp.content.iter_any():
                    if chunk:
                        yield chunk
            finally:
                resp.release()

        return StreamingResponse(
            _handoff_passthrough(),
            media_type=content_type,
            status_code=resp.status,
            headers=passthrough_headers,
        )

    # 純轉發模式（非 streaming 或 route=cloud_pd/cloud_fallback 或 edge-first 禁用）
    if not stream or not edge_enabled:
        if route_mode == ROUTE_LOCAL_FULL and local_verify_loop_available:
            local_complete_metrics: dict[str, Any] = {}
            try:
                payload_json = _local_verify_loop_complete_with_runtime_plan(
                    body,
                    draft_policy=draft_policy,
                    metrics=local_complete_metrics,
                    expert_session=expert_session_for_stream,
                )
                response_text = _extract_response_text_from_payload(payload_json)
                json_success = None
                if draft_policy["response_contract"] == "json":
                    json_success = _is_valid_json_text(response_text)
                _record_draft_mode_outcome(
                    draft_policy,
                    speculative=True,
                    hit=bool(local_complete_metrics.get("hit")),
                    spec_elapsed_ms=float(local_complete_metrics.get("spec_elapsed_ms") or 0.0),
                    local_ttft_ms=float(local_complete_metrics.get("local_ttft_ms") or 0.0),
                    cloud_ttft_ms=float(local_complete_metrics.get("cloud_ttft_ms") or 0.0),
                    json_success=json_success,
                )
                _record_request_completion(
                    response_text=response_text,
                    request_started_at=request_started_at,
                    first_token_ttft_ms=float(local_complete_metrics.get("local_ttft_ms") or 0.0),
                    success=True,
                    content_success=bool(str(response_text or "").strip()),
                    path_kind=_classify_request_path_kind(expert_session),
                )
                _expert_data_plane.complete_request(
                    expert_session,
                    success=True,
                    response_text=response_text,
                )
                _refresh_acceptance_live_reports(
                    family_info=family_info,
                    route_mode=route_mode,
                    transport_runtime=transport_runtime,
                    transport_route=transport_route,
                )
                return JSONResponse(
                    content=payload_json,
                    media_type="application/json",
                    headers={
                        "x-edge-first": "local_complete",
                        "x-edge-first-route-family": _header_safe(family_info.get("family") or "generic"),
                        "x-cgc-draft-mode": _header_safe(draft_policy["draft_mode"]),
                        "x-cgc-draft-enabled": "1" if draft_policy["enabled"] else "0",
                        "x-cgc-draft-disable-reason": _header_safe(draft_policy["disable_reason"] or ""),
                        "x-cgc-policy-source": _header_safe(draft_policy.get("source") or "unknown"),
                        "x-cgc-hermes-route-mode": _header_safe((hermes_policy or {}).get("mode") or ""),
                        "x-cgc-local-executor": "verify_loop",
                    },
                )
            except Exception as e:
                _record_request_completion(
                    response_text="",
                    request_started_at=request_started_at,
                    first_token_ttft_ms=float(local_complete_metrics.get("local_ttft_ms") or 0.0),
                    success=False,
                    content_success=False,
                    path_kind=_classify_request_path_kind(expert_session),
                )
                _expert_data_plane.complete_request(
                    expert_session,
                    success=False,
                    response_text="",
                )
                _refresh_acceptance_live_reports(
                    family_info=family_info,
                    route_mode=route_mode,
                    transport_runtime=transport_runtime,
                    transport_route=transport_route,
                )
                return JSONResponse(content={"error": str(e)}, status_code=502)
        session = await _get_cloud_session()
        try:
            resp = await session.post(
                cloud_endpoint,
                json=cloud_payload,
                headers={**request_headers, "Content-Type": "application/json"},
            )
            content_type = resp.headers.get("content-type", "application/json")
            if "text/event-stream" in content_type:
                async def _passthrough(resp=resp):
                    try:
                        async for chunk in resp.content.iter_any():
                            if chunk:
                                yield chunk
                    finally:
                        resp.release()
                return StreamingResponse(
                    _instrument_passthrough_stream(
                        _passthrough(),
                        request_started_at=request_started_at,
                        draft_policy=draft_policy,
                        family_info=family_info,
                        route_mode=route_mode,
                        transport_runtime=transport_runtime,
                        transport_route=transport_route,
                        expert_session=expert_session_for_stream,
                    ),
                    media_type=content_type,
                )
            data = await resp.read()
            resp.release()
            payload_json = json.loads(data)
            payload_json = _sanitize_payload_content_for_model(
                payload_json,
                model_name=str(body.get("model") or _ACTIVE_MODEL or ""),
            )
            json_success = None
            response_text = _extract_response_text_from_payload(payload_json)
            if draft_policy["response_contract"] == "json":
                json_success = _is_valid_json_text(response_text)
            _record_draft_mode_outcome(
                draft_policy,
                speculative=False,
                hit=False,
                json_success=json_success,
            )
            _record_request_completion(
                response_text=response_text,
                request_started_at=request_started_at,
                success=True,
                content_success=bool(str(response_text or "").strip()),
                path_kind=_classify_request_path_kind(expert_session),
            )
            _expert_data_plane.complete_request(
                expert_session,
                success=True,
                response_text=response_text,
            )
            _refresh_acceptance_live_reports(
                family_info=family_info,
                route_mode=route_mode,
                transport_runtime=transport_runtime,
                transport_route=transport_route,
            )
            return JSONResponse(
                content=payload_json,
                media_type="application/json",
                headers={
                    "x-edge-first": "passthrough",
                    "x-edge-first-route-family": _header_safe(family_info.get("family") or "generic"),
                    "x-cgc-draft-mode": _header_safe(draft_policy["draft_mode"]),
                    "x-cgc-draft-enabled": "1" if draft_policy["enabled"] else "0",
                    "x-cgc-draft-disable-reason": _header_safe(draft_policy["disable_reason"] or ""),
                    "x-cgc-policy-source": _header_safe(draft_policy.get("source") or "unknown"),
                    "x-cgc-hermes-route-mode": _header_safe((hermes_policy or {}).get("mode") or ""),
                },
            )
        except Exception as e:
            _record_request_completion(
                response_text="",
                request_started_at=request_started_at,
                success=False,
                content_success=False,
                path_kind=_classify_request_path_kind(expert_session),
            )
            _expert_data_plane.complete_request(
                expert_session,
                success=False,
                response_text="",
            )
            _refresh_acceptance_live_reports(
                family_info=family_info,
                route_mode=route_mode,
                transport_runtime=transport_runtime,
                transport_route=transport_route,
            )
            return JSONResponse(content={"error": str(e)}, status_code=502)

    # Edge-first streaming 模式 (parallel preflight: miss不痛)
    async def _edge_first_stream():
        t0 = time.monotonic()
        path_kind = _classify_request_path_kind(expert_session)
        first_token_sent = False
        local_ttft_ms = 0.0
        cloud_first_ttft_ms = 0.0
        spec_hit = False
        cloud_text_parts: list[str] = []
        request_failed = False
        frontier_advancer = _StreamFrontierAdvancer(expert_session=expert_session)
        # === temperature>0 适配 ===
        # 检测客户端 temperature, 决定投机策略
        client_temperature = float(body.get("temperature", 0.0) or 0.0)
        temp_band = _get_temperature_band(client_temperature)
        _record_stats(temp_band=temp_band)
        # temperature>1.0 → 禁用投机 (太随机, 依赖 parallel preflight)
        # temperature>0 → 使用 top-k fuzzy match
        temp_allows_spec = _should_speculate_for_temperature(client_temperature, family_info)
        speculate = temp_allows_spec and draft_policy["enabled"] and _should_speculate(family_info)
        _maybe_schedule_warmup(cloud_endpoint, cloud_payload, request_headers, family_info)

        if not temp_allows_spec and temp_band == _TEMP_BAND_HIGH:
            _record_stats(temp_disabled=1)
            print(f"[edge-first] temperature={client_temperature} > 1.0, speculation disabled (parallel preflight active)", file=sys.stderr)

        # === parallel preflight ===
        # 1. 立即启动云端请求 (async task, 不阻塞)
        cloud_stream_payload = {**cloud_payload, "stream": True}
        # 不减 max_tokens: hit 时跳过云端首个 content token, miss 时也跳过 (保持现有行为)
        # 这样云端始终生成完整 max_tokens, 避免并行时无法预知是否投机的困境
        loop = asyncio.get_running_loop()
        session = await _get_cloud_session()
        cloud_resp_future = loop.create_task(
            session.post(
                cloud_endpoint,
                json=cloud_stream_payload,
                headers={**request_headers, "Content-Type": "application/json"},
            )
        )
        cloud_preflight_t = time.monotonic()

        # 2. 运行投机 (直接调用, 避免 tokenizer 线程安全 bus error)
        predicted_text = None
        predicted_candidates: list[str] = []  # top-k 候选 (temperature>0 fuzzy match)
        if speculate:
            try:
                if temp_band == _TEMP_BAND_DETERMINISTIC:
                    # temperature=0: 精确单 token 预测 (现有行为)
                    predicted_text = _edge_generate_first_token(
                        messages,
                        1,
                        model_name=str(body.get("model") or _ACTIVE_MODEL or ""),
                    )
                else:
                    # temperature>0: top-k 预测, 首选用于发送, 全列表用于 fuzzy hit
                    predicted_text, predicted_candidates = _predict_first_token_topk(
                        messages, client_temperature, k=_TEMP_TOPK_SIZE
                    )
                    if predicted_candidates:
                        print(f"[edge-first] top-k prediction (temp={client_temperature}): "
                              f"first='{predicted_text}' candidates={predicted_candidates}",
                              file=sys.stderr)
            except Exception as e:
                print(f"[edge-first] speculation error: {e}", file=sys.stderr)
            _record_stats(speculated=1)

        spec_elapsed = (time.monotonic() - cloud_preflight_t) * 1000
        first_text = predicted_text

        # 3. 如果投机有结果，立即发送首 token (TTFT = spec_time, 通常 1-2ms)
        if first_text:
            ttft_ms = (time.monotonic() - t0) * 1000
            local_ttft_ms = ttft_ms
            _record_ttft_sample(ttft_ms, path_kind=path_kind)
            first_chunk = {
                "id": f"edge_{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "edge-first"),
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": first_text}, "finish_reason": None}],
            }
            frontier_advancer.observe_text(str(first_text))
            yield f"data: {json.dumps(first_chunk)}\n\n".encode()
            first_token_sent = True
            print(f"[edge-first] TTFT: {ttft_ms:.0f}ms (local) predicted='{first_text}' spec_elapsed={spec_elapsed:.0f}ms", file=sys.stderr)

        # 4. 等待云端响应 (可能已在 spec 期间完成大部分计算)
        try:
            resp = await cloud_resp_future
            content_type = resp.headers.get("content-type", "text/event-stream")

            cloud_first = True
            cloud_first_token_recorded = False
            _user_msg = _extract_user_message(messages)
            _prompt_hash = str(family_info.get("prompt_hash") or "")

            def _try_record_cloud_first(chunk_bytes):
                """从 cloud chunk 中提取并记录首 token，比较投机准确率。
                返回 (has_content, content_text) — has_content=True 表示有内容需要处理。
                """
                nonlocal cloud_first_token_recorded, spec_hit
                try:
                    line = chunk_bytes.decode().strip()
                    if not line.startswith("data:"):
                        return False, None
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        return False, None
                    obj = json.loads(data_str)
                    choices = obj.get("choices", [])
                    if not choices:
                        return False, None
                    delta = choices[0].get("delta", {})
                    cloud_content = delta.get("content", "")
                    if not cloud_content or not cloud_content.strip():
                        return False, None
                    if not cloud_first_token_recorded:
                        cloud_first_token_recorded = True
                        nonlocal cloud_first_ttft_ms
                        cloud_first_ttft_ms = (time.monotonic() - t0) * 1000
                        cloud_first_text = cloud_content.strip().split()[0] if cloud_content.strip() else cloud_content
                        _record_first_token(_prompt_hash, cloud_first_text, _user_msg)
                        # 統計投機準確率
                        if predicted_text and cloud_first_text:
                            _spec_family = str(family_info.get("family") or "generic")
                            # temperature=0: 精确匹配
                            # temperature>0: fuzzy match (cloud 首 token 在 top-k 候选中)
                            is_exact = predicted_text.strip() == cloud_first_text.strip()
                            is_fuzzy = (not is_exact) and bool(predicted_candidates) and any(
                                c.strip() == cloud_first_text.strip() for c in predicted_candidates
                            )
                            if is_exact:
                                _record_stats(speculation_correct=1)
                                spec_hit = True
                                get_acceptance_tracker().record(hit=True, family=_spec_family)
                                print(f"[edge-first] spec HIT (exact): predicted='{predicted_text}' == cloud='{cloud_first_text}'", file=sys.stderr)
                            elif is_fuzzy:
                                # fuzzy hit: cloud 首 token 在 top-k 候选中 (temperature>0)
                                _record_stats(speculation_fuzzy_hit=1)
                                spec_hit = True  # fuzzy hit 也算 hit, blank 云端首 token
                                get_acceptance_tracker().record(hit=True, family=_spec_family)
                                print(f"[edge-first] spec FUZZY HIT: cloud='{cloud_first_text}' in top-k={predicted_candidates} (temp={client_temperature})", file=sys.stderr)
                            else:
                                if temp_band == _TEMP_BAND_DETERMINISTIC:
                                    _record_stats(speculation_wrong=1)
                                else:
                                    _record_stats(speculation_fuzzy_miss=1)
                                spec_hit = False
                                get_acceptance_tracker().record(hit=False, family=_spec_family)
                                cloud_ttft = (time.monotonic() - t0) * 1000
                                _tracker = get_acceptance_tracker()
                                _tracker_status = _tracker.get_status()
                                print(f"[edge-first] spec MISS: predicted='{predicted_text}' != cloud='{cloud_first_text}' cloud_ttft={cloud_ttft:.0f}ms (miss penalty=0, preflight started at t=0) | tracker: state={_tracker_status['state']} rate={_tracker_status['global_accept_rate']} samples={_tracker_status['global_samples']}", file=sys.stderr)
                        if not first_token_sent:
                            ttft_ms = (time.monotonic() - t0) * 1000
                            cloud_first_ttft_ms = ttft_ms
                            _record_ttft_sample(ttft_ms, path_kind=path_kind)
                    return True, cloud_content
                except Exception:
                    return False, None

            # 5. 从已启动的云端响应 stream (with miss correction)
            try:
                async for chunk in resp.content.iter_any():
                    if not chunk:
                        continue
                    has_content, cloud_content = _try_record_cloud_first(chunk)
                    if has_content and cloud_content:
                        cloud_text_parts.append(str(cloud_content))

                    # 如果本地已發送首 token，处理云端首个 content chunk
                    if cloud_first and first_token_sent:
                        if not has_content:
                            # 非 content chunk (如 role-only delta): 跳过 (role 已在预测 chunk 中发送)
                            continue
                        # 找到首个 content chunk
                        cloud_first = False
                        if spec_hit:
                            # HIT: 空白云端首 token (已被本地预测替代)
                            try:
                                line = chunk.decode().strip()
                                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                                    obj = json.loads(line[5:].strip())
                                    choices = obj.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        if delta.get("content"):
                                            delta["content"] = ""
                                            yield f"data: {json.dumps(obj)}\n\n".encode()
                                            continue
                            except Exception:
                                pass
                        else:
                            # MISS: 发送 correction marker, 让云端正确 token 通过
                            correction_chunk = {
                                "id": f"edge_{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": body.get("model", "edge-first"),
                                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
                                "x-cgc-speculation": "miss",
                                "x-cgc-predicted": predicted_text or "",
                            }
                            yield f"data: {json.dumps(correction_chunk)}\n\n".encode()
                            # 让云端正确首 token 通过 (不 blank)
                            frontier_advancer.observe_text(str(cloud_content or ""))
                            yield chunk
                            print(f"[edge-first] CORRECTION: cloud correct='{cloud_content}' replaces wrong='{predicted_text}'", file=sys.stderr)
                        continue
                    cloud_first = False
                    if has_content and cloud_content:
                        frontier_advancer.observe_text(str(cloud_content))
                    yield chunk
            finally:
                resp.release()

        except Exception as e:
            request_failed = True
            error_chunk = {"error": {"message": f"cloud error: {e}", "type": "cloud_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n".encode()
        finally:
            json_success = None
            final_response_text = "".join(cloud_text_parts)
            if first_token_sent and predicted_text and spec_hit:
                final_response_text = f"{predicted_text}{final_response_text}"
            elif not final_response_text and predicted_text:
                final_response_text = str(predicted_text)
            if draft_policy["response_contract"] == "json":
                json_success = _is_valid_json_text(final_response_text)
            _record_draft_mode_outcome(
                draft_policy,
                speculative=bool(speculate and predicted_text),
                hit=spec_hit,
                spec_elapsed_ms=spec_elapsed,
                local_ttft_ms=local_ttft_ms,
                cloud_ttft_ms=cloud_first_ttft_ms,
                json_success=json_success,
            )
            _record_request_completion(
                response_text=final_response_text,
                request_started_at=t0,
                first_token_ttft_ms=local_ttft_ms or cloud_first_ttft_ms,
                success=(not request_failed),
                content_success=bool(final_response_text.strip()),
                path_kind=path_kind,
            )
            _expert_data_plane.complete_request(
                expert_session,
                success=(not request_failed),
                response_text=final_response_text,
            )
            _refresh_acceptance_live_reports(
                family_info=family_info,
                route_mode=route_mode,
                transport_runtime=transport_runtime,
                transport_route=transport_route,
            )

    _tracker = get_acceptance_tracker()
    return StreamingResponse(
        _edge_first_stream(),
        media_type="text/event-stream",
        headers={
            "x-edge-first": "enabled" if draft_policy["enabled"] else "gated",
            "x-edge-first-route-family": _header_safe(family_info.get("family") or "generic"),
            "x-edge-first-confidence": _header_safe(family_info.get("confidence") or 0.0),
            "x-cgc-draft-mode": _header_safe(draft_policy["draft_mode"]),
            "x-cgc-draft-enabled": "1" if draft_policy["enabled"] else "0",
            "x-cgc-draft-disable-reason": _header_safe(draft_policy["disable_reason"] or ""),
            "x-cgc-policy-source": _header_safe(draft_policy.get("source") or "unknown"),
            "x-cgc-hermes-route-mode": _header_safe((hermes_policy or {}).get("mode") or ""),
            "x-cgc-acceptance-state": _header_safe(_tracker.get_state()),
            "x-cgc-acceptance-rate": _header_safe(round(_tracker.get_accept_rate(), 3)),
            "x-cgc-temp-band": _header_safe(_get_temperature_band(float(body.get("temperature", 0.0) or 0.0))),
            "Cache-Control": "no-cache",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-First Proxy (Cross-Model)")
    parser.add_argument("--port", type=int, default=30001, help="Proxy 監聽端口")
    parser.add_argument("--cloud-url", type=str, default="http://127.0.0.1:30000",
                        help="雲端 sglang URL")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--active-model", type=str, default="gemma4",
                        help="激活的模型: gemma4 / dsv4 / qwen3vl / huihui_moe (及别名 g4/ds/qwen/huihui)")
    parser.add_argument("--tokenizer-path", type=str, default="",
                        help="Tokenizer 路徑 (空=從 registry 讀取)")
    args = parser.parse_args()

    os.environ.setdefault("CLOUD_URL", args.cloud_url)

    # 初始化模型配置 (从 registry 加载校准规则)
    cfg = _init_model_config(args.active_model)
    if args.tokenizer_path:
        os.environ["DSV4_TOKENIZER_PATH"] = args.tokenizer_path
    elif cfg and cfg.tokenizer_path:
        os.environ["DSV4_TOKENIZER_PATH"] = cfg.tokenizer_path

    print(f"[edge-first] Proxy 啟動: http://{args.host}:{args.port}")
    print(f"[edge-first] 雲端: {args.cloud_url}")
    print(f"[edge-first] Active model: {_ACTIVE_MODEL}")
    if cfg:
        print(f"[edge-first] Model: {cfg.display_name}, hidden={cfg.hidden_size}, "
              f"vocab={cfg.vocab_size}, EOS={sorted(cfg.eos_tokens)}")
        print(f"[edge-first] Calibration: {len(_PROMPT_FAMILY_RULES)} rules from registry")
    print(f"[edge-first] Tokenizer: {os.environ.get('DSV4_TOKENIZER_PATH', 'auto')}")
    print(f"[edge-first] Edge model: {os.environ.get('EDGE_MODEL_PATH', 'auto-download Qwen2.5-0.5B')}")

    # 初始化 Draft Registry (多 Draft 动态加载)
    draft_reg = get_draft_registry()
    draft_available = draft_reg.set_active(args.active_model)
    print(f"[edge-first] Draft registry: active={args.active_model}, available={draft_available}")
    print(f"[edge-first] Draft status: {draft_reg.get_status()}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
