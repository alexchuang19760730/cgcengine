from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Type
from ..types import CGCModule, CGCFunction, CGCNode, CGCTensor, DType
from .base import Backend

class ComputeKernel:
    def __init__(self, name: str, backend: str, func: Callable):
        self.name = name
        self.backend = backend
        self.func = func
        self.signature = None
        self.priority = 0

class ComputeInjector:
    _kernels: Dict[str, List[ComputeKernel]] = {}
    
    @classmethod
    def register_kernel(cls, name: str, backend: str, priority: int = 0):
        def decorator(func: Callable) -> Callable:
            kernel = ComputeKernel(name, backend, func)
            kernel.priority = priority
            if name not in cls._kernels:
                cls._kernels[name] = []
            cls._kernels[name].append(kernel)
            cls._kernels[name].sort(key=lambda k: k.priority, reverse=True)
            return func
        return decorator
    
    @classmethod
    def get_kernel(cls, name: str, backend: str) -> Optional[ComputeKernel]:
        if name not in cls._kernels:
            return None
        for kernel in cls._kernels[name]:
            if kernel.backend == backend or kernel.backend == "universal":
                return kernel
        return None
    
    @classmethod
    def list_kernels(cls) -> List[str]:
        return list(cls._kernels.keys())

@ComputeInjector.register_kernel("GDSCopy", "cuda", priority=100)
def cuda_gds_copy_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("GDSCopy", "universal", priority=50)
def universal_gds_copy_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("SPDKRead", "universal", priority=100)
def spdk_read_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("SPDKWrite", "universal", priority=100)
def spdk_write_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("FlashMoE", "cuda", priority=100)
def cuda_flash_moe_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("FlashMoE", "ascend", priority=95)
def ascend_flash_moe_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("R-SWA", "universal", priority=100)
def rswa_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    """
    R-SWA 真實計算算子 (基於 Host 2 移植)
    
    Inputs: [query, key, value, (optional) reference_key, reference_value]
    Outputs: [out_tensor]
    Attributes: {"window_size": int, "num_heads": int, "head_dim": int}
    """
    import torch
    import torch.nn.functional as F
    
    q, k, v = inputs[0], inputs[1], inputs[2]
    B, T, C = q.shape
    num_heads = attributes.get("num_heads", 32)
    head_dim = attributes.get("head_dim", C // num_heads)
    window_size = attributes.get("window_size", 128)
    
    # 支持 GQA: 從 k 的形狀推斷 kv_heads
    kv_heads = k.shape[-1] // head_dim
    
    # 解析 Reference KV
    ref_k, ref_v = None, None
    if len(inputs) > 3:
        ref_k = inputs[3]
    if len(inputs) > 4:
        ref_v = inputs[4]
        
    # 重塑形狀為 (B, H, T, D)
    q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, kv_heads, head_dim).transpose(1, 2)
    
    # 拼接 Reference KV
    full_k = k
    full_v = v
    if ref_k is not None and ref_v is not None:
        full_k = torch.cat([ref_k, k], dim=2)
        full_v = torch.cat([ref_v, v], dim=2)
        
    # Repeat K, V for GQA
    if num_heads != kv_heads:
        num_kv_groups = num_heads // kv_heads
        full_k = torch.repeat_interleave(full_k, num_kv_groups, dim=1)
        full_v = torch.repeat_interleave(full_v, num_kv_groups, dim=1)
        
    # 構建滑動窗口 Mask (Reference 部分永遠可見)
    full_len = full_k.size(2)
    output_start = full_len - T
    ref_len = ref_k.size(2) if ref_k is not None else 0
    
    attn_mask = torch.ones((T, full_len), device=q.device, dtype=torch.bool)
    for i in range(T):
        pos = output_start + i
        l_bound = max(ref_len, pos - window_size)
        attn_mask[i, :l_bound] = False
        
    # 計算注意力
    attn = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=attn_mask)
    attn = attn.transpose(1, 2).reshape(B, T, C)
    
    # 寫入 Output KV
    if not outputs:
        outputs.append(attn)
        outputs.append(full_k[:, :, -window_size:].detach())  # new_k
        outputs.append(full_v[:, :, -window_size:].detach())  # new_v
    else:
        outputs[0] = attn

@ComputeInjector.register_kernel("NFSoRDMA", "universal", priority=90)
def nfs_rdma_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("OMLX", "universal", priority=80)
def omlx_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("RoPE", "universal", priority=100)
def rope_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("FlashAttention", "cuda", priority=100)
def cuda_flash_attn_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

@ComputeInjector.register_kernel("FlashAttention", "ascend", priority=95)
def ascend_flash_attn_kernel(inputs: List, outputs: List, attributes: Dict) -> None:
    pass

class ComputeBackend(Backend):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.supported_dtypes = []
        self.supported_ops = list(ComputeInjector.list_kernels())
        self.priority = 80
    
    def compile(self, module: CGCModule) -> Dict[str, Any]:
        compiled = {}
        for func_name, func in module.functions.items():
            compiled[func_name] = self.compile_function(func)
        return compiled
    
    def compile_function(self, func: CGCFunction) -> Dict[str, Any]:
        compiled_nodes = []
        for node in func.topological_sort():
            kernel = ComputeInjector.get_kernel(node.op_type, self.name)
            if kernel:
                compiled_nodes.append({
                    "op_type": node.op_type,
                    "kernel": kernel.name,
                    "kernel_func": kernel.func,
                    "inputs": [inp.name for inp in node.inputs],
                    "outputs": [out.name for out in node.outputs],
                    "attributes": node.attributes,
                    "priority": kernel.priority
                })
            else:
                compiled_nodes.append({
                    "op_type": node.op_type,
                    "kernel": None,
                    "fallback": True,
                    "inputs": [inp.name for inp in node.inputs],
                    "outputs": [out.name for out in node.outputs]
                })
        
        return {
            "name": func.name,
            "parameters": [p.name for p in func.parameters],
            "results": [r.name for r in func.results],
            "compiled_nodes": compiled_nodes,
            "target": self.name
        }
    
    def run(self, compiled_module: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for func_name, func in compiled_module.items():
            results[func_name] = self._run_function(func, inputs)
        return results
    
    def _run_function(self, func: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {}
        for node in func["compiled_nodes"]:
            if node["kernel"]:
                kernel_func = node["kernel_func"]
                kernel_inputs = [inputs.get(name) for name in node["inputs"]]
                kernel_outputs = []
                kernel_func(kernel_inputs, kernel_outputs, node["attributes"])
                for out_name in node["outputs"]:
                    outputs[out_name] = kernel_outputs
        return outputs