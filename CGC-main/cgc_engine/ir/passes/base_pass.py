from __future__ import annotations
from abc import ABC, abstractmethod
from ..types import CGCModule, CGCFunction, CGCNode


class Pass(ABC):
    """Base class for all optimization passes"""
    
    name: str = "base_pass"
    
    @abstractmethod
    def run(self, module_or_func) -> None:
        """Run the pass on a module or function"""
        pass
    
    def run_module(self, module: CGCModule) -> None:
        """Run pass on all functions in a module"""
        for func_name, func in module.functions.items():
            self.run_function(func)
    
    def run_function(self, func: CGCFunction) -> None:
        """Run pass on a single function"""
        pass
