from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type, TypeVar
from ..types import CGCModule, CGCFunction, CGCNode, CGCTensor, DType

T = TypeVar('T')

class Backend(ABC):
    name: str
    supported_dtypes: List[DType]
    supported_ops: List[str]
    priority: int = 0
    
    @abstractmethod
    def __init__(self):
        self.name = "base"
        self.supported_dtypes = []
        self.supported_ops = []
    
    @abstractmethod
    def compile(self, module: CGCModule) -> Any:
        """Compile a CGC module to target backend representation"""
        pass
    
    @abstractmethod
    def compile_function(self, func: CGCFunction) -> Any:
        """Compile a single function to target backend representation"""
        pass
    
    @abstractmethod
    def run(self, compiled_module: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run a compiled module with given inputs"""
        pass
    
    def can_handle_op(self, op_type: str) -> bool:
        """Check if the backend supports a specific operation"""
        return op_type in self.supported_ops
    
    def can_handle_dtype(self, dtype: DType) -> bool:
        """Check if the backend supports a specific dtype"""
        return dtype in self.supported_dtypes
    
    def validate_module(self, module: CGCModule) -> bool:
        """Validate if the module can be compiled by this backend"""
        for func_name, func in module.functions.items():
            if not self.validate_function(func):
                return False
        return True
    
    def validate_function(self, func: CGCFunction) -> bool:
        """Validate if a function can be compiled by this backend"""
        for node in func.body:
            if not self.can_handle_op(node.op_type):
                return False
            for inp in node.inputs:
                if not self.can_handle_dtype(inp.type.dtype):
                    return False
            for out in node.outputs:
                if not self.can_handle_dtype(out.type.dtype):
                    return False
        return True
    
    def get_preferred_dtype(self) -> DType:
        """Get the preferred dtype for this backend"""
        if DType.BFLOAT16 in self.supported_dtypes:
            return DType.BFLOAT16
        elif DType.FLOAT16 in self.supported_dtypes:
            return DType.FLOAT16
        elif DType.FLOAT32 in self.supported_dtypes:
            return DType.FLOAT32
        else:
            return DType.FLOAT32
    
    def optimize(self, module: CGCModule, level: int = 3) -> CGCModule:
        """Optimize the module for this backend"""
        return module
    
    def __str__(self) -> str:
        return f"{self.name} backend"
    
    def __repr__(self) -> str:
        return f"<Backend: {self.name}>"

class BackendRegistry:
    _backends: Dict[str, Backend] = {}
    
    @classmethod
    def register(cls, backend: Backend) -> None:
        """Register a backend instance"""
        cls._backends[backend.name] = backend
    
    @classmethod
    def register_class(cls, name: str, backend_class: Type[Backend]) -> None:
        """Register a backend class"""
        cls._backends[name] = backend_class()
    
    @classmethod
    def get_backend(cls, name: str) -> Optional[Backend]:
        """Get a registered backend instance"""
        return cls._backends.get(name)
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backends"""
        return list(cls._backends.keys())
    
    @classmethod
    def get_supported_backends(cls, module: Optional[CGCModule] = None) -> List[Backend]:
        """Get all backends that can handle the given module"""
        backends = []
        for name in cls._backends:
            backend = cls.get_backend(name)
            if backend is None:
                continue
            if module is None:
                backends.append(backend)
            elif backend.validate_module(module):
                backends.append(backend)
        return sorted(backends, key=lambda b: b.priority, reverse=True)
    
    @classmethod
    def auto_select(cls, module: CGCModule) -> Optional[Backend]:
        """Auto-select the best backend for the given module"""
        supported = cls.get_supported_backends(module)
        return supported[0] if supported else None

def register_backend(name: str):
    """Decorator to register a backend class"""
    def decorator(cls: Type[Backend]) -> Type[Backend]:
        BackendRegistry.register_class(name, cls)
        return cls
    return decorator