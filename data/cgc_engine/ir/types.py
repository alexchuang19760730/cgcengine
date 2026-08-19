from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set, Union
from enum import Enum

class DType(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"

class MemorySpace(str, Enum):
    GLOBAL = "global"
    SHARED = "shared"
    REGISTER = "register"
    CONSTANT = "constant"

class TensorLayout(str, Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    BLOCKED = "blocked"

@dataclass(frozen=True)
class Shape:
    dims: Tuple[int, ...]
    
    def __post_init__(self):
        for dim in self.dims:
            if dim < 0 and dim != -1:
                raise ValueError(f"Invalid dimension: {dim}")
    
    @property
    def rank(self) -> int:
        return len(self.dims)
    
    def __len__(self) -> int:
        return self.rank
    
    def __getitem__(self, idx: int) -> int:
        return self.dims[idx]
    
    def is_static(self) -> bool:
        return all(dim > 0 for dim in self.dims)

@dataclass(frozen=True)
class CGCType:
    dtype: DType
    shape: Shape
    layout: TensorLayout = TensorLayout.ROW_MAJOR
    memory_space: MemorySpace = MemorySpace.GLOBAL
    strides: Optional[Tuple[int, ...]] = None
    
    @property
    def numel(self) -> int:
        result = 1
        for dim in self.shape.dims:
            if dim > 0:
                result *= dim
        return result
    
    @property
    def size_bytes(self) -> int:
        dtype_sizes = {
            DType.FLOAT32: 4,
            DType.FLOAT16: 2,
            DType.BFLOAT16: 2,
            DType.INT8: 1,
            DType.UINT8: 1,
            DType.INT32: 4,
            DType.INT64: 8,
            DType.BOOL: 1,
        }
        return self.numel * dtype_sizes.get(self.dtype, 4)

@dataclass(eq=False)
class CGCTensor:
    name: str
    type: CGCType
    is_parameter: bool = False
    is_constant: bool = False
    producers: List["CGCNode"] = field(default_factory=list)
    consumers: List["CGCNode"] = field(default_factory=list)
    version: int = 0
    
    def __hash__(self):
        return hash((self.name, self.version))
    
    def __eq__(self, other):
        if isinstance(other, CGCTensor):
            return self.name == other.name and self.version == other.version
        return False

@dataclass(eq=False)
class CGCNode:
    op_type: str
    name: Optional[str] = None
    inputs: List[CGCTensor] = field(default_factory=list)
    outputs: List[CGCTensor] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        for inp in self.inputs:
            inp.consumers.append(self)
        for out in self.outputs:
            out.producers.append(self)
    
    @property
    def is_identity(self) -> bool:
        return self.op_type == "Identity"
    
    @property
    def is_parameter(self) -> bool:
        return self.op_type == "Parameter"
    
    @property
    def is_constant(self) -> bool:
        return self.op_type == "Constant"

@dataclass
class CGCFunction:
    name: str
    parameters: List[CGCTensor] = field(default_factory=list)
    constants: List[CGCTensor] = field(default_factory=list)
    results: List[CGCTensor] = field(default_factory=list)
    body: List[CGCNode] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: CGCNode) -> None:
        self.body.append(node)
        node.metadata.setdefault("function", self)
    
    def remove_node(self, node: CGCNode) -> None:
        self.body.remove(node)
    
    def topological_sort(self) -> List[CGCNode]:
        in_degree = {node: len(node.inputs) for node in self.body}
        queue = [n for n in self.body if in_degree[n] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            for out in node.outputs:
                for consumer in out.consumers:
                    if consumer in in_degree:
                        in_degree[consumer] -= 1
                        if in_degree[consumer] == 0:
                            queue.append(consumer)
        
        return result
    
    def get_node_by_name(self, name: str) -> Optional[CGCNode]:
        for node in self.body:
            if node.name == name:
                return node
        return None
    
    def __deepcopy__(self, memo):
        from copy import deepcopy
        new_func = CGCFunction(name=self.name)
        memo[id(self)] = new_func
        
        new_func.parameters = deepcopy(self.parameters, memo)
        new_func.constants = deepcopy(self.constants, memo)
        new_func.results = deepcopy(self.results, memo)
        new_func.body = deepcopy(self.body, memo)
        new_func.attributes = deepcopy(self.attributes, memo)
        new_func.metadata = deepcopy(self.metadata, memo)
        
        return new_func

@dataclass
class CGCModule:
    name: str
    functions: Dict[str, CGCFunction] = field(default_factory=dict)
    global_vars: Dict[str, CGCTensor] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_function(self, func: CGCFunction) -> None:
        self.functions[func.name] = func
    
    def get_function(self, name: str) -> Optional[CGCFunction]:
        return self.functions.get(name)
    
    def remove_function(self, name: str) -> None:
        self.functions.pop(name, None)