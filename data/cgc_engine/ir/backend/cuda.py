from __future__ import annotations
from typing import List, Dict, Any, Optional
from ..types import CGCModule, CGCFunction, CGCNode, CGCTensor, DType, CGCType, Shape
from .base import Backend, register_backend

@register_backend("cuda")
class CUDABackend(Backend):
    def __init__(self):
        self.name = "cuda"
        self.priority = 100
        self.supported_dtypes = [
            DType.FLOAT32,
            DType.FLOAT16,
            DType.BFLOAT16,
            DType.INT8,
            DType.UINT8,
            DType.INT32,
            DType.INT64,
            DType.BOOL,
        ]
        self.supported_ops = [
            "Parameter",
            "Constant",
            "Identity",
            "Add",
            "Mul",
            "MatMul",
            "Conv2D",
            "Attention",
            "LayerNorm",
            "Linear",
            "Reshape",
            "Transpose",
            "GELU",
            "Softmax",
            "MoE",
            "Mean",
            "Sum",
            "Max",
            "Cat",
            "Slice",
            "Dropout",
            "ReLU",
            "Sigmoid",
            "Tanh",
            "BatchNorm",
            "Pooling",
            "GDSCopy",
            "SPDKRead",
            "SPDKWrite",
            "FlashMoE",
            "R-SWA",
            "NFSoRDMA",
            "OMLX",
            "RoPE",
            "FlashAttention",
        ]
    
    def compile(self, module: CGCModule) -> Dict[str, Any]:
        compiled = {}
        for func_name, func in module.functions.items():
            compiled[func_name] = self.compile_function(func)
        return compiled
    
    def compile_function(self, func: CGCFunction) -> Dict[str, Any]:
        nodes = []
        for node in func.topological_sort():
            nodes.append(self._compile_node(node))
        
        return {
            "name": func.name,
            "parameters": [p.name for p in func.parameters],
            "results": [r.name for r in func.results],
            "nodes": nodes,
            "attributes": func.attributes,
        }
    
    def _compile_node(self, node: CGCNode) -> Dict[str, Any]:
        return {
            "op_type": node.op_type,
            "name": node.name,
            "inputs": [inp.name for inp in node.inputs],
            "outputs": [out.name for out in node.outputs],
            "attributes": node.attributes,
            "target": "cuda",
        }
    
    def run(self, compiled_module: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for func_name, func in compiled_module.items():
            results[func_name] = self._run_function(func, inputs)
        return results
    
    def _run_function(self, func: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "message": f"Function {func['name']} would run on CUDA"}
    
    def optimize(self, module: CGCModule, level: int = 3) -> CGCModule:
        if level >= 1:
            module = self._fuse_matmul_add(module)
        if level >= 2:
            module = self._fuse_layernorm_add(module)
        if level >= 3:
            module = self._optimize_memory_layout(module)
        return module
    
    def _fuse_matmul_add(self, module: CGCModule) -> CGCModule:
        for func in module.functions.values():
            new_body = []
            i = 0
            while i < len(func.body):
                node = func.body[i]
                if node.op_type == "MatMul" and i + 1 < len(func.body):
                    next_node = func.body[i + 1]
                    if next_node.op_type == "Add" and next_node.inputs[0] in node.outputs:
                        fused_node = CGCNode(
                            op_type="FusedMatMulAdd",
                            name=f"{node.name}_fused",
                            inputs=[node.inputs[0], node.inputs[1], next_node.inputs[1]],
                            outputs=next_node.outputs,
                            attributes={**node.attributes}
                        )
                        new_body.append(fused_node)
                        i += 2
                        continue
                new_body.append(node)
                i += 1
            func.body = new_body
        return module
    
    def _fuse_layernorm_add(self, module: CGCModule) -> CGCModule:
        for func in module.functions.values():
            new_body = []
            i = 0
            while i < len(func.body):
                node = func.body[i]
                if node.op_type == "LayerNorm" and i + 1 < len(func.body):
                    next_node = func.body[i + 1]
                    if next_node.op_type == "Add" and next_node.inputs[0] in node.outputs:
                        fused_node = CGCNode(
                            op_type="FusedLayerNormAdd",
                            name=f"{node.name}_fused",
                            inputs=[node.inputs[0]] + node.inputs[1:] + [next_node.inputs[1]],
                            outputs=next_node.outputs,
                            attributes={**node.attributes}
                        )
                        new_body.append(fused_node)
                        i += 2
                        continue
                new_body.append(node)
                i += 1
            func.body = new_body
        return module
    
    def _optimize_memory_layout(self, module: CGCModule) -> CGCModule:
        return module