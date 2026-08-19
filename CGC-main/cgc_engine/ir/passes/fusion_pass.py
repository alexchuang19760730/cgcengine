from __future__ import annotations
from typing import List
from .base_pass import Pass
from ..types import CGCFunction, CGCNode


class FusionPass(Pass):
    """Fuse common patterns like MatMul+Add, LayerNorm+Add"""
    
    name = "fusion_pass"
    
    def __init__(self):
        self.fused_patterns = 0
    
    def run(self, module_or_func) -> None:
        if hasattr(module_or_func, 'functions'):
            self.run_module(module_or_func)
        else:
            self.run_function(module_or_func)
    
    def run_function(self, func: CGCFunction) -> None:
        """Fuse MatMul+Add and LayerNorm+Add patterns"""
        new_body = []
        i = 0
        nodes = func.body
        
        while i < len(nodes):
            node = nodes[i]
            
            if i + 1 < len(nodes):
                next_node = nodes[i + 1]
                
                if node.op_type == "MatMul" and next_node.op_type == "Add":
                    if next_node.inputs[0] == node.outputs[0] or next_node.inputs[1] == node.outputs[0]:
                        fused_node = CGCNode(
                            op_type="FusedMatMulAdd",
                            inputs=[node.inputs[0], node.inputs[1], next_node.inputs[1] if next_node.inputs[0] == node.outputs[0] else next_node.inputs[0]],
                            outputs=next_node.outputs,
                            attributes={"fused": True, "pattern": "MatMulAdd"}
                        )
                        new_body.append(fused_node)
                        self.fused_patterns += 1
                        i += 2
                        continue
                
                if node.op_type == "LayerNorm" and next_node.op_type == "Add":
                    if next_node.inputs[0] == node.outputs[0] or next_node.inputs[1] == node.outputs[0]:
                        fused_node = CGCNode(
                            op_type="FusedLayerNormAdd",
                            inputs=[node.inputs[0], node.inputs[1], next_node.inputs[1] if next_node.inputs[0] == node.outputs[0] else next_node.inputs[0]],
                            outputs=next_node.outputs,
                            attributes={"fused": True, "pattern": "LayerNormAdd"}
                        )
                        new_body.append(fused_node)
                        self.fused_patterns += 1
                        i += 2
                        continue
            
            new_body.append(node)
            i += 1
        
        func.body = new_body
