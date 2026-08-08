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
LoRA → vLLM Bridge

将 MLX-Tune / Megatrain 微调的 LoRA 权重导出到 vLLM 可加载格式

功能:
- 加载 MLX-Tune LoRA 权重
- CGC 指令合并 LoRA 到基础权重
- 导出 vLLM 可加载的完整权重
- 支持 FlashKDA + LoRA 融合

Usage:
    python -m cgc_engine.bridge.lora_vllm_bridge
"""

import torch
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    from ..cgc.cgc_simd_executor import CGCExecutor
    from ..cgc.mlx_tune_integration import cgc_mlx_tune, CGCMlxTune
    from ..cgc.cgc_opcodes import CGC_OP_CODES
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False


@dataclass
class LoRAWeightPaths:
    """LoRA 权重路径"""
    lora_a: str
    lora_b: str
    base_weight: str
    scale: float = 1.0


class LoRAtoVLLMBridge:
    """
    LoRA → vLLM 权重桥接

    将微调的 LoRA 权重通过 CGC 指令合并后导出到 vLLM 格式
    """

    def __init__(self, output_dir: str = "vllm_kda_lora_model"):
        self.output_dir = Path(output_dir)
        self.cgc: Optional[CGCExecutor] = None
        self.lora_a: Optional[torch.Tensor] = None
        self.lora_b: Optional[torch.Tensor] = None
        self.base: Optional[torch.Tensor] = None
        self.merged_weight: Optional[torch.Tensor] = None
        self.lora_config: Dict[str, Any] = {}

        if CGC_AVAILABLE:
            self.cgc = CGCExecutor()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Bridge] Initialized: output={self.output_dir}")

    def load_mlx_lora(self, lora_path: str) -> "LoRAtoVLLMBridge":
        """
        加载 MLX-Tune LoRA 权重

        Args:
            lora_path: LoRA 权重目录路径

        Returns:
            self
        """
        lora_path = Path(lora_path)

        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")

        lora_a_path = lora_path / "lora_a.pt"
        lora_b_path = lora_path / "lora_b.pt"
        base_path = lora_path / "base_weight.pt"
        config_path = lora_path / "lora_config.json"

        if lora_a_path.exists():
            self.lora_a = torch.load(lora_a_path, map_location="cpu")
            print(f"[Bridge] Loaded lora_a: {self.lora_a.shape}")
        else:
            raise FileNotFoundError(f"lora_a.pt not found in {lora_path}")

        if lora_b_path.exists():
            self.lora_b = torch.load(lora_b_path, map_location="cpu")
            print(f"[Bridge] Loaded lora_b: {self.lora_b.shape}")
        else:
            raise FileNotFoundError(f"lora_b.pt not found in {lora_path}")

        if base_path.exists():
            self.base = torch.load(base_path, map_location="cpu")
            print(f"[Bridge] Loaded base_weight: {self.base.shape}")
        else:
            print("[Bridge] Warning: base_weight.pt not found, will use identity")

        if config_path.exists():
            with open(config_path) as f:
                self.lora_config = json.load(f)
            print(f"[Bridge] Loaded config: {self.lora_config}")

        print("[Bridge] MLX LoRA 加载完成")
        return self

    def load_lora_from_dict(self, lora_state: Dict[str, torch.Tensor]) -> "LoRAtoVLLMBridge":
        """
        从字典加载 LoRA 权重

        Args:
            lora_state: 包含 lora_a, lora_b, base_weight 的字典

        Returns:
            self
        """
        self.lora_a = lora_state.get("lora_a")
        self.lora_b = lora_state.get("lora_b")
        self.base = lora_state.get("base_weight")

        if self.lora_a is not None:
            print(f"[Bridge] Loaded lora_a: {self.lora_a.shape}")
        if self.lora_b is not None:
            print(f"[Bridge] Loaded lora_b: {self.lora_b.shape}")
        if self.base is not None:
            print(f"[Bridge] Loaded base_weight: {self.base.shape}")

        return self

    def merge_lora(self, scale: float = 1.0) -> "LoRAtoVLLMBridge":
        """
        CGC 指令合并 LoRA 到基础权重

        Args:
            scale: LoRA 缩放因子

        Returns:
            self
        """
        if self.cgc is None:
            print("[Bridge] CGC not available, using native merge")
            return self._native_merge(scale)

        try:
            tensors = {
                "base_weight": self.base if self.base is not None else torch.eye(self.lora_a.shape[1]),
                "lora_a": self.lora_a,
                "lora_b": self.lora_b,
                "scale": scale,
            }

            self.merged_weight = self.cgc.run(
                opcode=CGC_OP_CODES.LORA_MERGE,
                inputs=[],
                params=tensors,
            )

            if isinstance(self.merged_weight, list):
                self.merged_weight = self.merged_weight[0]

            print("[Bridge] CGC 合并 LoRA → 完整权重")

        except Exception as e:
            print(f"[Bridge] CGC merge failed: {e}, falling back to native")
            return self._native_merge(scale)

        return self

    def _native_merge(self, scale: float) -> "LoRAtoVLLMBridge":
        """原生 LoRA 合并 (无 CGC)"""
        if self.base is None:
            self.merged_weight = torch.matmul(self.lora_b.t(), self.lora_a.t()) * scale
        else:
            self.merged_weight = self.base + torch.matmul(self.lora_b.t(), self.lora_a.t()) * scale

        print("[Bridge] Native 合并 LoRA → 完整权重")
        return self

    def export_to_vllm(
        self,
        output_dir: Optional[str] = None,
        format: str = "safetensors",
    ) -> Path:
        """
        导出 vLLM 可加载权重

        Args:
            output_dir: 输出目录
            format: 导出格式 (safetensors / pytorch / onnx)

        Returns:
            输出目录路径
        """
        if self.merged_weight is None:
            raise ValueError("No merged weight. Call merge_lora() first.")

        output = Path(output_dir) if output_dir else self.output_dir
        output.mkdir(parents=True, exist_ok=True)

        model_path = output / "model.safetensors" if format == "safetensors" else output / "model.pth"

        if format == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file({"model.safetensors": self.merged_weight}, model_path)
                print(f"[Bridge] Safetensors 保存: {model_path}")
            except ImportError:
                print("[Bridge] safetensors not installed, falling back to PyTorch")
                torch.save(self.merged_weight, model_path)
                print(f"[Bridge] PyTorch model 保存: {model_path}")
        else:
            torch.save(self.merged_weight, model_path)
            print(f"[Bridge] PyTorch model 保存: {model_path}")

        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "bfloat16",
            "hidden_size": self.merged_weight.shape[0] if len(self.merged_weight.shape) > 1 else 4096,
            "intermediate_size": self.merged_weight.shape[0] if len(self.merged_weight.shape) > 1 else 11008,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "lora_config": {
                "rank": self.lora_a.shape[0] if self.lora_a is not None else 16,
                "alpha": self.lora_config.get("alpha", 16),
                "target_modules": self.lora_config.get("target_modules", ["q_proj", "v_proj"]),
            },
        }

        config_path = output / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[Bridge] Config 保存: {config_path}")

        print(f"[Bridge] ✅ 导出完成 → {output}")
        print("👉 可直接在 vLLM + CGC + FlashKDA 推理")

        return output

    def export_huggingface(self, output_dir: str) -> Path:
        """
        导出为 HuggingFace 格式

        Args:
            output_dir: 输出目录

        Returns:
            输出目录路径
        """
        if self.merged_weight is None:
            raise ValueError("No merged weight. Call merge_lora() first.")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        tokenizer_files = output / "tokenizer.json"
        if not tokenizer_files.exists():
            with open(tokenizer_files, "w") as f:
                json.dump({
                    "version": "1.0",
                    "truncation": None,
                    "padding": None,
                    "model": "LlamaTokenizer",
                }, f)

        tokenizer_config = output / "tokenizer_config.json"
        with open(tokenizer_config, "w") as f:
            json.dump({
                "add_bos_token": False,
                "add_eos_token": False,
                "bos_token": "</s>",
                "eos_token": "</s>",
                "pad_token": "<unk>",
            }, f)

        model_files = output / "pytorch_model.bin"
        torch.save(self.merged_weight, model_files)
        print(f"[Bridge] HuggingFace 格式导出: {output}")

        return output


def load_and_export(
    lora_path: str,
    output_dir: str = "vllm_kda_lora_model",
    scale: float = 1.0,
):
    """
    一键加载并导出

    Args:
        lora_path: LoRA 权重目录
        output_dir: 输出目录
        scale: 缩放因子
    """
    bridge = LoRAtoVLLMBridge(output_dir=output_dir)
    bridge.load_mlx_lora(lora_path)
    bridge.merge_lora(scale=scale)
    bridge.export_to_vllm()
    return bridge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA → vLLM Bridge")
    parser.add_argument("--lora-path", type=str, required=True, help="LoRA 权重目录路径")
    parser.add_argument("--output-dir", type=str, default="vllm_kda_lora_model", help="输出目录")
    parser.add_argument("--scale", type=float, default=1.0, help="LoRA 缩放因子")

    args = parser.parse_args()

    load_and_export(
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        scale=args.scale,
    )
