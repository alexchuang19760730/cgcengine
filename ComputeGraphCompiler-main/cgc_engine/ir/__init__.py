from .types import (
    DType,
    MemorySpace,
    TensorLayout,
    Shape,
    CGCType,
    CGCTensor,
    CGCNode,
    CGCFunction,
    CGCModule,
)

from .ops import (
    create_parameter,
    create_constant,
    create_identity,
    create_add,
    create_mul,
    create_matmul,
    create_conv2d,
    create_attention,
    create_layer_norm,
    create_linear,
    create_reshape,
    create_transpose,
    create_gelu,
    create_softmax,
    create_moe,
    create_mean,
    create_sum,
    create_max,
    create_cat,
    create_slice,
    create_gds_copy,
    create_spdk_read,
    create_spdk_write,
    create_flash_moe,
    create_rswa,
    create_nfs_rdma,
    create_omlx,
)

from .backend.base import (
    Backend,
    BackendRegistry,
    register_backend,
)

from .backend.cuda import CUDABackend
from .backend.metal import MetalBackend
from .backend.ascend import AscendBackend

from .passes import (
    FusionPass,
    LayoutPass,
    MemoryPlanningPass,
)

__all__ = [
    "DType",
    "MemorySpace",
    "TensorLayout",
    "Shape",
    "CGCType",
    "CGCTensor",
    "CGCNode",
    "CGCFunction",
    "CGCModule",
    "create_parameter",
    "create_constant",
    "create_identity",
    "create_add",
    "create_mul",
    "create_matmul",
    "create_conv2d",
    "create_attention",
    "create_layer_norm",
    "create_linear",
    "create_reshape",
    "create_transpose",
    "create_gelu",
    "create_softmax",
    "create_moe",
    "create_mean",
    "create_sum",
    "create_max",
    "create_cat",
    "create_slice",
    "create_gds_copy",
    "create_spdk_read",
    "create_spdk_write",
    "create_flash_moe",
    "create_rswa",
    "create_nfs_rdma",
    "create_omlx",
    "Backend",
    "BackendRegistry",
    "register_backend",
    "CUDABackend",
    "MetalBackend",
    "AscendBackend",
    "FusionPass",
    "LayoutPass",
    "MemoryPlanningPass",
]

def get_backend(name: str):
    return BackendRegistry.get_backend(name)

def list_backends():
    return BackendRegistry.list_backends()

def auto_select_backend(module):
    return BackendRegistry.auto_select(module)