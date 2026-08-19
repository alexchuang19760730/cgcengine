from __future__ import annotations
from .base_pass import Pass
from ..types import CGCFunction, CGCNode


class LayoutPass(Pass):
    """Optimize tensor memory layout for better cache performance"""
    
    name = "layout_pass"
    
    def __init__(self):
        self.optimized_layouts = 0
    
    def run(self, module_or_func) -> None:
        if hasattr(module_or_func, 'functions'):
            self.run_module(module_or_func)
        else:
            self.run_function(module_or_func)
    
    def run_function(self, func: CGCFunction) -> None:
        """Optimize tensor layouts"""
        for node in func.body:
            for out in node.outputs:
                if out.type is not None:
                    out.type.layout = "contiguous"
                    self.optimized_layouts += 1
