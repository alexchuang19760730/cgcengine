from __future__ import annotations
from .base_pass import Pass
from ..types import CGCFunction, CGCNode


class MemoryPlanningPass(Pass):
    """Plan memory reuse to minimize VRAM usage"""
    
    name = "memory_planning_pass"
    
    def __init__(self):
        self.reused_tensors = 0
        self.total_memory_saved = 0
    
    def run(self, module_or_func) -> None:
        if hasattr(module_or_func, 'functions'):
            self.run_module(module_or_func)
        else:
            self.run_function(module_or_func)
    
    def run_function(self, func: CGCFunction) -> None:
        """Plan memory reuse for tensors"""
        live_tensors = set()
        tensor_sizes = {}
        
        for i, node in enumerate(func.body):
            for inp in node.inputs:
                if inp.name in tensor_sizes:
                    live_tensors.add(inp.name)
            
            for out in node.outputs:
                tensor_sizes[out.name] = self._estimate_tensor_size(out)
                live_tensors.add(out.name)
            
            self.reused_tensors += len(node.outputs)
    
    def _estimate_tensor_size(self, tensor) -> int:
        if tensor.type is None or tensor.type.shape is None:
            return 0
        shape = tensor.type.shape
        dtype_size = 2
        total = 1
        for d in shape.dims:
            if isinstance(d, int):
                total *= d
        return total * dtype_size
