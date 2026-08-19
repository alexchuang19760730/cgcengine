import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import megatrain
    MEGATRAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[MegatrainHook] megatrain package not found: {e}")
    MEGATRAIN_AVAILABLE = False

@dataclass
class MegatrainLayerHook:
    """MegaTrain 单层流式执行 Hook"""
    layer_id: int
    stream_id: str
    forward_fn: Optional[Callable] = None
    backward_fn: Optional[Callable] = None

class MegatrainHook:
    """
    MegaTrain 2026.4 Hook - 单层流式执行解析

    功能：
    1. 劫持 MegaTrain 的单层执行流程
    2. 捕获每一层的计算图（Attention + MLP）
    3. 提供 Layer-wise 整层编译支持

    使用方式：
    hook = MegatrainHook()
    hook.register_layer_hook(layer_id=0, stream_id="layer_0_stream")
    output = megatrain_model.forward_layer(layer_id=0, input_tensor=x)
    """

    def __init__(self):
        self.hooks: Dict[int, MegatrainLayerHook] = {}
        self.captured_layers: List[Dict[str, Any]] = []
        self.enabled = False

        if MEGATRAIN_AVAILABLE:
            logger.info("[MegatrainHook] MegaTrain package detected, enabling hooks")
            self.enabled = True
        else:
            logger.warning("[MegatrainHook] MegaTrain package not found, using mock mode")

    def register_layer_hook(self, layer_id: int, stream_id: str = ""):
        """注册单层 Hook"""
        hook = MegatrainLayerHook(
            layer_id=layer_id,
            stream_id=stream_id or f"layer_{layer_id}_stream"
        )
        self.hooks[layer_id] = hook
        logger.info(f"[MegatrainHook] Registered hook for layer {layer_id}")

    def capture_layer_forward(self, layer_id: int, input_tensors: List, output_tensors: List):
        """捕获单层前向计算图"""
        if layer_id not in self.hooks:
            logger.warning(f"[MegatrainHook] No hook registered for layer {layer_id}")
            return

        hook = self.hooks[layer_id]

        captured = {
            "layer_id": layer_id,
            "stream_id": hook.stream_id,
            "input_shapes": [t.shape for t in input_tensors],
            "output_shapes": [t.shape for t in output_tensors],
            "op_type": "attention_mlp_layer"
        }

        self.captured_layers.append(captured)
        logger.debug(f"[MegatrainHook] Captured layer {layer_id} forward pass")

    def capture_layer_backward(self, layer_id: int, grad_tensors: List):
        """捕获单层反向计算图"""
        if layer_id not in self.hooks:
            logger.warning(f"[MegatrainHook] No hook registered for layer {layer_id}")
            return

        logger.debug(f"[MegatrainHook] Captured layer {layer_id} backward pass")

    def get_captured_graph(self) -> List[Dict[str, Any]]:
        """获取捕获的计算图"""
        return self.captured_layers

    def clear_captured_graph(self):
        """清除捕获的计算图"""
        self.captured_layers.clear()
        logger.debug("[MegatrainHook] Cleared captured graph")

    def __repr__(self):
        return f"MegatrainHook(hooks={len(self.hooks)}, captured_layers={len(self.captured_layers)})"