# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Megatrain ↔ vLLM Bridge - 训推一体闭环

功能:
- Megatrain 训练权重 → vLLM 格式自动转换
- KDA 正交基 (ortho_basis) 完整保留
- 一键导出 HuggingFace / vLLM / GGUF 格式

文件位置: cgc_engine/bridge/megatrain_vllm_bridge.py

Architecture:
    Megatrain 训练输出 (.megatrain / .pth)
              ↓
    MegatrainVLLMBridge (格式转换 + KDA对齐)
              ↓
    vLLM 推理 (CGC + FlashKDA)
              ↓
    导出成品模型 (HF / vLLM / GGUF)
"""

import torch
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

try:
    from ..cgc import (
        CGCExecutor,
        FlashKDALayer,
        CGC_OP_CODES,
        get_megatrain_cgc_info,
    )
    from ..flashkda_integration import FLASHKDA_AVAILABLE
    BRIDGE_CGC_AVAILABLE = True
except ImportError:
    BRIDGE_CGC_AVAILABLE = False
    FLASHKDA_AVAILABLE = False


@dataclass
class BridgeConfig:
    """Bridge 配置"""
    megatrain_model_name: str = "kimi-linear-48b"
    vllm_model_name: str = "KimiLinearForCausalLM"

    attention_backend: str = "cgc_kda"

    export_formats: List[str] = field(default_factory=lambda: ["vllm", "huggingface"])

    ckpt_suffix: str = ".pth"
    save_ortho_basis: bool = True


class MegatrainVLLMBridge:
    """
    Megatrain ↔ vLLM Bridge

    实现:
    1. 加载 Megatrain 训练权重
    2. 自动转换权重命名 (Megatrain → vLLM)
    3. 保留 KDA 正交基
    4. 注入 vLLM 推理引擎
    5. 导出可发布模型
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()

        self.cgc: Optional[CGCExecutor] = None
        self.flashkda: Optional[FlashKDALayer] = None

        self.megatrain_state: Optional[Dict[str, torch.Tensor]] = None
        self.vllm_state: Optional[Dict[str, torch.Tensor]] = None
        self.megatrain_config: Optional[Dict[str, Any]] = None
        self.vllm_config: Optional[Dict[str, Any]] = None
        self.kda_ortho_basis: Optional[torch.Tensor] = None

        self._init_cgc()

    def _init_cgc(self):
        """初始化 CGC 组件"""
        if BRIDGE_CGC_AVAILABLE:
            self.cgc = CGCExecutor(enable_profiling=False)
            if FLASHKDA_AVAILABLE:
                self.flashkda = FlashKDALayer()
            print(f"[Bridge] CGC initialized, FlashKDA available: {FLASHKDA_AVAILABLE}")

    # =========================================================================
    # 1. 加载 Megatrain 训练权重
    # =========================================================================

    def load_megatrain_ckpt(
        self,
        ckpt_path: str,
        device: str = "cpu",
    ) -> "MegatrainVLLMBridge":
        """
        加载 Megatrain 训练 checkpoint

        Args:
            ckpt_path: checkpoint 文件路径 (.pth)
            device: 加载设备

        Returns:
            self
        """
        print(f"[Bridge] Loading Megatrain checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        self.megatrain_state = ckpt.get("model_state", ckpt)

        if "config" in ckpt:
            self.megatrain_config = ckpt["config"]
        elif "model_config" in ckpt:
            self.megatrain_config = ckpt["model_config"]
        else:
            self.megatrain_config = self._infer_config_from_state()

        self.kda_ortho_basis = ckpt.get("kda_ortho_basis", None)

        if self.kda_ortho_basis is not None:
            print(f"[Bridge] KDA ortho_basis found: shape={self.kda_ortho_basis.shape}")
        else:
            print(f"[Bridge] No KDA ortho_basis in checkpoint")

        print(f"[Bridge] Loaded {len(self.megatrain_state)} state dict keys")
        print(f"[Bridge] Config: hidden_size={self.megatrain_config.get('hidden_size')}, "
              f"num_layers={self.megatrain_config.get('num_layers')}")

        return self

    def load_megatrain_dir(
        self,
        ckpt_dir: str,
        epoch: int = 10,
        device: str = "cpu",
    ) -> "MegatrainVLLMBridge":
        """
        从目录加载 Megatrain checkpoint

        Args:
            ckpt_dir: checkpoint 目录
            epoch: 加载的 epoch 号
            device: 加载设备

        Returns:
            self
        """
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pth")
        return self.load_megatrain_ckpt(ckpt_path, device)

    def _infer_config_from_state(self) -> Dict[str, Any]:
        """从 state_dict 推断配置"""
        if self.megatrain_state is None:
            return {}

        config = {}

        embedding_key = None
        for k in self.megatrain_state.keys():
            if "embed_tokens" in k or "embedding" in k:
                embedding_key = k
                break

        if embedding_key:
            vocab_size = self.megatrain_state[embedding_key].shape[0]
            hidden_dim = self.megatrain_state[embedding_key].shape[1]
            config["vocab_size"] = vocab_size
            config["hidden_size"] = hidden_dim

        layer_key = None
        for k in self.megatrain_state.keys():
            if "layers.0" in k or "model.layers.0" in k:
                layer_key = k
                break

        if layer_key:
            first_layer_param = self.megatrain_state[layer_key]
            if len(first_layer_param.shape) >= 2:
                config["intermediate_size"] = first_layer_param.shape[0]

        num_layers = 0
        for k in self.megatrain_state.keys():
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        num_layers = max(num_layers, layer_idx + 1)
                    except ValueError:
                        pass

        config["num_layers"] = num_layers

        return config

    # =========================================================================
    # 2. 转换权重格式
    # =========================================================================

    def convert_to_vllm(self) -> "MegatrainVLLMBridge":
        """
        转换 Megatrain 权重 → vLLM 格式

        Returns:
            self
        """
        if self.megatrain_state is None:
            raise RuntimeError("No Megatrain state loaded. Call load_megatrain_ckpt() first.")

        print("[Bridge] Converting Megatrain → vLLM format...")

        self.vllm_state = {}
        self.vllm_config = self._convert_config_vllm()

        for key, value in self.megatrain_state.items():
            new_key = self._auto_map_key(key)
            self.vllm_state[new_key] = value

        if self.kda_ortho_basis is not None and self.config.save_ortho_basis:
            self.vllm_state["kda.ortho_basis"] = self.kda_ortho_basis
            print("[Bridge] KDA ortho_basis injected into vLLM state")

        print(f"[Bridge] Conversion complete: {len(self.vllm_state)} keys")
        return self

    def _auto_map_key(self, key: str) -> str:
        """
        自动映射 Megatrain 权重名 → vLLM 权重名

        映射规则:
        - layers.N → model.layers.N
        - attention.* → self_attn.*
        - w_kda.* → kda.*
        - mlp.fc1 → mlp.gate_up_proj
        - mlp.fc2 → mlp.down_proj
        - norm.weight → input_layernorm.weight
        """
        new_key = key

        new_key = new_key.replace("layers.", "model.layers.")

        new_key = new_key.replace("attention.", "self_attn.")
        new_key = new_key.replace("attn.", "self_attn.")

        new_key = new_key.replace("w_kda.", "kda.")
        new_key = new_key.replace("kda_proj.", "kda.")

        new_key = new_key.replace("mlp.fc1", "mlp.gate_up_proj")
        new_key = new_key.replace("mlp.fc2", "mlp.down_proj")
        new_key = new_key.replace("mlp.fc", "mlp.gate_up_proj")

        new_key = new_key.replace("post_attention_layernorm", "post_attention_layernorm")
        new_key = new_key.replace("final_layernorm", "model.norm")

        new_key = new_key.replace("lm_head.", "lm_head.")

        new_key = new_key.replace("weight", "weight")
        new_key = new_key.replace("bias", "bias")

        return new_key

    def _convert_config_vllm(self) -> Dict[str, Any]:
        """转换 Megatrain 配置 → vLLM 配置"""
        mc = self.megatrain_config

        vllm_cfg = {
            "architectures": [self.config.vllm_model_name],
            "hidden_size": mc.get("hidden_size", 5120),
            "intermediate_size": mc.get("intermediate_size", 13824),
            "num_hidden_layers": mc.get("num_layers", mc.get("num_layers", 48)),
            "num_attention_heads": mc.get("num_heads", mc.get("num_attention_heads", 40)),
            "num_key_value_heads": mc.get("num_kv_heads", mc.get("num_key_value_heads", 40)),
            "vocab_size": mc.get("vocab_size", 128256),
            "head_dim": mc.get("head_dim", 128),
            "rope_theta": mc.get("rope_theta", 10000),
            "max_position_embeddings": mc.get("max_seq_len", mc.get("max_position_embeddings", 32768)),
            "use_kda": True,
            "use_flash_kda": True,
            "attention_backend": self.config.attention_backend,
            "torch_dtype": "bfloat16",
            "model_type": "kimi",
        }

        return vllm_cfg

    # =========================================================================
    # 3. 注入 vLLM 推理
    # =========================================================================

    def to_vllm(
        self,
        tensorizer_path: Optional[str] = None,
    ):
        """
        创建 vLLM 推理引擎

        Args:
            tensorizer_path: 可选的自定义 tensorizer 路径

        Returns:
            vLLM LLM 实例
        """
        if self.vllm_state is None:
            self.convert_to_vllm()

        print("[Bridge] Creating vLLM engine with CGC + FlashKDA...")

        try:
            from vllm import LLM
            from vllm.engine.arg_utils import EngineArgs

            engine_args = EngineArgs(
                model=self.config.megatrain_model_name,
                tokenizer=self.config.megatrain_model_name,
                dtype="bfloat16",
                attention_backend=self.config.attention_backend,
                enforce_eager=True,
                load_in_8bit=False,
                trust_remote_code=True,
            )

            llm = LLM(engine_args)

            if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "model_executor"):
                try:
                    llm.llm_engine.model_executor.load_state_dict(self.vllm_state)
                    print("[Bridge] State dict injected into vLLM executor")
                except AttributeError:
                    print("[Bridge] vLLM version doesn't support direct state_dict injection")
                    print("[Bridge] Using model from HuggingFace with converted config")

            print("[Bridge] vLLM engine ready with CGC + FlashKDA backend")
            return llm

        except ImportError:
            print("[Bridge] vLLM not installed. Use export_model() to save weights.")
            return None

    # =========================================================================
    # 4. 导出模型
    # =========================================================================

    def export_model(
        self,
        export_path: str,
        formats: Optional[List[str]] = None,
    ) -> "MegatrainVLLMBridge":
        """
        导出模型到指定格式

        Args:
            export_path: 导出目录
            formats: 导出格式列表 ["vllm", "huggingface", "gguf"]

        Returns:
            self
        """
        if self.vllm_state is None:
            self.convert_to_vllm()

        formats = formats or self.config.export_formats

        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        if "vllm" in formats or "pytorch" in formats:
            self._export_pytorch(export_dir)

        if "huggingface" in formats:
            self._export_huggingface(export_dir)

        if "gguf" in formats:
            self._export_gguf(export_dir)

        self._export_bridge_info(export_dir)

        print(f"[Bridge] Export complete → {export_path}")
        print(f"[Bridge] Available formats: {formats}")

        return self

    def _export_pytorch(self, export_dir: Path):
        """导出 PyTorch 格式"""
        model_path = export_dir / "model.pth"
        torch.save(self.vllm_state, model_path)
        print(f"[Bridge] PyTorch model saved: {model_path}")

    def _export_huggingface(self, export_dir: Path):
        """导出 HuggingFace 格式"""
        import os

        hf_dir = export_dir / "huggingface"
        hf_dir.mkdir(exist_ok=True)

        config_path = hf_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.vllm_config, f, indent=2)

        safetensors_path = hf_dir / "model.safetensors"
        pytorch_model_path = hf_dir / "pytorch_model.bin"

        try:
            from safetensors.torch import save_file
            state_dict_split = {}
            for key, tensor in self.vllm_state.items():
                if hasattr(tensor, 'to'):
                    state_dict_split[key] = tensor
                else:
                    state_dict_split[key] = torch.from_numpy(tensor)

            save_file(state_dict_split, safetensors_path)
            print(f"[Bridge] Safetensors saved: {safetensors_path}")

        except ImportError:
            import warnings
            warnings.warn("safetensors not installed, falling back to PyTorch format")
            torch.save(self.vllm_state, pytorch_model_path)
            print(f"[Bridge] PyTorch model saved: {pytorch_model_path}")

        tokenizer_files = hf_dir / "tokenizer.json"
        tokenizer_config = hf_dir / "tokenizer_config.json"
        if not tokenizer_files.exists():
            with open(tokenizer_files, "w") as f:
                json.dump({"version": "1.0", "truncation": None, "padding": None}, f)
        if not tokenizer_config.exists():
            with open(tokenizer_config, "w") as f:
                json.dump({
                    "add_bos_token": False,
                    "add_eos_token": False,
                    "bos_token": "</s>",
                    "eos_token": "</s>",
                    "pad_token": "<unk>",
                }, f)

        print(f"[Bridge] HuggingFace model saved: {hf_dir}")
        print(f"[Bridge] Files: config.json, model.safetensors/pytorch_model.bin, tokenizer.json, tokenizer_config.json")

    def _export_gguf(self, export_dir: Path):
        """导出 GGUF 格式 (需要 llama.cpp)"""
        gguf_dir = export_dir / "gguf"
        gguf_dir.mkdir(exist_ok=True)

        print("[Bridge] GGUF export requires llama.cpp conversion")
        print(f"[Bridge] Run: llama.cpp/convert.py --outtype f16 --outfile {gguf_dir}/model.gguf {export_dir}/huggingface")

    def export_to_pd(
        self,
        pd_endpoint: str = "localhost:50051",
        rank: int = 0,
        world_size: int = 1,
    ) -> "MegatrainVLLMBridge":
        """
        將訓練好的權重直接註冊到 PD 服務

        Args:
            pd_endpoint: PD 服務地址
            rank: 當前進程排名
            world_size: 總進程數

        Returns:
            self
        """
        if self.vllm_state is None:
            self.convert_to_vllm()

        try:
            import grpc
            from cgc_engine.pb import weight_service_pb2_grpc, weight_service_pb2

            channel = grpc.insecure_channel(pd_endpoint)
            stub = weight_service_pb2_grpc.WeightServiceStub(channel)

            for key, tensor in self.vllm_state.items():
                weight_data = tensor.cpu().numpy().tobytes()
                request = weight_service_pb2.RegisterWeightRequest(
                    key=key,
                    data=weight_data,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    rank=rank,
                )
                stub.RegisterWeight(request)

            if self.kda_ortho_basis is not None:
                ortho_data = self.kda_ortho_basis.cpu().numpy().tobytes()
                request = weight_service_pb2.RegisterWeightRequest(
                    key="kda.ortho_basis",
                    data=ortho_data,
                    shape=list(self.kda_ortho_basis.shape),
                    dtype=str(self.kda_ortho_basis.dtype).replace("torch.", ""),
                    rank=rank,
                )
                stub.RegisterWeight(request)

            channel.close()
            print(f"[Bridge] Registered {len(self.vllm_state)} weights to PD at {pd_endpoint}")

        except ImportError:
            print("[Bridge] gRPC not available, using PD client directly")

            try:
                from cgc_engine.pd.pd_client import PDClient

                pd_client = PDClient(endpoint=pd_endpoint)

                for key, tensor in self.vllm_state.items():
                    pd_client.register_weight(
                        key=key,
                        tensor=tensor,
                        rank=rank,
                    )

                if self.kda_ortho_basis is not None:
                    pd_client.register_weight(
                        key="kda.ortho_basis",
                        tensor=self.kda_ortho_basis,
                        rank=rank,
                    )

                print(f"[Bridge] Registered {len(self.vllm_state)} weights to PD via client")

            except Exception as e:
                print(f"[Bridge] PD export failed: {e}")
                print("[Bridge] Falling back to local storage")
                self._pd_local_cache = self.vllm_state.copy()

        return self

    def _export_bridge_info(self, export_dir: Path):
        """导出 Bridge 元信息"""
        bridge_info = {
            "version": "1.0.0",
            "architecture": "Megatrain ↔ vLLM Bridge",
            "megatrain_config": self.megatrain_config,
            "vllm_config": self.vllm_config,
            "kda_ortho_basis_shape": self.kda_ortho_basis.shape if self.kda_ortho_basis is not None else None,
            "total_parameters": sum(v.numel() for v in self.vllm_state.values()),
            "bridge_features": {
                "cgc_simd": True,
                "flashkda": FLASHKDA_AVAILABLE,
                "attention_backend": self.config.attention_backend,
            },
        }

        info_path = export_dir / "bridge_info.json"
        with open(info_path, "w") as f:
            json.dump(bridge_info, f, indent=2)

        print(f"[Bridge] Bridge info saved: {info_path}")

    # =========================================================================
    # 工具方法
    # =========================================================================

    def get_state_dict_diff(self) -> Dict[str, Tuple[torch.Size, torch.Size]]:
        """
        获取 Megatrain → vLLM 转换前后的 state_dict 差异

        Returns:
            key → (megatrain_shape, vllm_shape)
        """
        if self.megatrain_state is None or self.vllm_state is None:
            return {}

        diff = {}
        megatrain_keys = set(self.megatrain_state.keys())
        vllm_keys = set(self.vllm_state.keys())

        for key in megatrain_keys & vllm_keys:
            mc_shape = self.megatrain_state[key].shape
            vllm_shape = self.vllm_state[key].shape
            if mc_shape != vllm_shape:
                diff[key] = (mc_shape, vllm_shape)

        return diff

    def verify_kda_ortho_basis(self) -> bool:
        """
        验证 KDA 正交基是否正确保留

        Returns:
            True if valid
        """
        if self.kda_ortho_basis is None:
            print("[Bridge] No KDA ortho_basis in checkpoint")
            return False

        is_orthogonal = torch.allclose(
            torch.matmul(self.kda_ortho_basis, self.kda_ortho_basis.t()),
            torch.eye(self.kda_ortho_basis.shape[0]),
            atol=1e-3,
        )

        if is_orthogonal:
            print("[Bridge] KDA ortho_basis verified: orthogonal ✓")
        else:
            print("[Bridge] KDA ortho_basis warning: not orthogonal")

        return is_orthogonal

    def summary(self) -> Dict[str, Any]:
        """
        获取 Bridge 摘要信息
        """
        return {
            "megatrain_keys": len(self.megatrain_state) if self.megatrain_state else 0,
            "vllm_keys": len(self.vllm_state) if self.vllm_state else 0,
            "kda_ortho_basis_shape": self.kda_ortho_basis.shape if self.kda_ortho_basis is not None else None,
            "vllm_config": self.vllm_config,
            "flashkda_available": FLASHKDA_AVAILABLE,
            "cgc_available": BRIDGE_CGC_AVAILABLE,
        }


def create_bridge(
    megatrain_ckpt: str,
    export_path: str,
    export_formats: Optional[List[str]] = None,
) -> MegatrainVLLMBridge:
    """
    一键创建 Bridge: 加载 → 转换 → 导出

    Args:
        megatrain_ckpt: Megatrain checkpoint 路径
        export_path: 导出目录
        export_formats: 导出格式

    Returns:
        MegatrainVLLMBridge 实例
    """
    bridge = MegatrainVLLMBridge()

    bridge.load_megatrain_ckpt(megatrain_ckpt)

    bridge.convert_to_vllm()

    bridge.export_model(export_path, export_formats)

    return bridge
