from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Set
from .types import CGCNode, CGCTensor, CGCFunction, CGCType, Shape, DType

def create_parameter(func: CGCFunction, name: str, cgc_type: CGCType) -> CGCTensor:
    param_tensor = CGCTensor(name=name, type=cgc_type, is_parameter=True)
    func.parameters.append(param_tensor)
    node = CGCNode(op_type="Parameter", name=f"{name}_param", outputs=[param_tensor])
    func.add_node(node)
    return param_tensor

def create_constant(func: CGCFunction, name: str, cgc_type: CGCType, value: Any = None) -> CGCTensor:
    const_tensor = CGCTensor(name=name, type=cgc_type, is_constant=True)
    func.constants.append(const_tensor)
    node = CGCNode(
        op_type="Constant", 
        name=f"{name}_const", 
        outputs=[const_tensor],
        attributes={"value": value}
    )
    func.add_node(node)
    return const_tensor

def create_identity(func: CGCFunction, input: CGCTensor, name: Optional[str] = None) -> CGCTensor:
    output = CGCTensor(
        name=name or f"{input.name}_identity",
        type=CGCType(dtype=input.type.dtype, shape=input.type.shape)
    )
    node = CGCNode(
        op_type="Identity",
        name=name,
        inputs=[input],
        outputs=[output]
    )
    func.add_node(node)
    return output

def create_add(func: CGCFunction, a: CGCTensor, b: CGCTensor, name: Optional[str] = None) -> CGCTensor:
    if a.type.shape != b.type.shape:
        raise ValueError(f"Shape mismatch in Add: {a.type.shape} vs {b.type.shape}")
    
    output = CGCTensor(
        name=name or f"{a.name}_plus_{b.name}",
        type=CGCType(dtype=a.type.dtype, shape=a.type.shape)
    )
    node = CGCNode(
        op_type="Add",
        name=name,
        inputs=[a, b],
        outputs=[output]
    )
    func.add_node(node)
    return output

def create_mul(func: CGCFunction, a: CGCTensor, b: CGCTensor, name: Optional[str] = None) -> CGCTensor:
    if a.type.shape != b.type.shape:
        raise ValueError(f"Shape mismatch in Mul: {a.type.shape} vs {b.type.shape}")
    
    output = CGCTensor(
        name=name or f"{a.name}_mul_{b.name}",
        type=CGCType(dtype=a.type.dtype, shape=a.type.shape)
    )
    node = CGCNode(
        op_type="Mul",
        name=name,
        inputs=[a, b],
        outputs=[output]
    )
    func.add_node(node)
    return output

def create_matmul(
    func: CGCFunction,
    a: CGCTensor,
    b: CGCTensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    name: Optional[str] = None
) -> CGCTensor:
    a_shape = a.type.shape.dims
    b_shape = b.type.shape.dims
    
    if transpose_a:
        a_m, a_k = a_shape
    else:
        if len(a_shape) == 2:
            a_m, a_k = a_shape
        else:
            a_m, a_k = a_shape[-2], a_shape[-1]
    
    if transpose_b:
        b_k, b_n = b_shape
    else:
        if len(b_shape) == 2:
            b_k, b_n = b_shape
        else:
            b_k, b_n = b_shape[-2], b_shape[-1]
    
    if a_k != b_k:
        raise ValueError(f"MatMul dimension mismatch: a_k={a_k}, b_k={b_k}")
    
    output_shape = Shape(dims=(a_m, b_n))
    output = CGCTensor(
        name=name or f"{a.name}_matmul_{b.name}",
        type=CGCType(dtype=a.type.dtype, shape=output_shape)
    )
    node = CGCNode(
        op_type="MatMul",
        name=name,
        inputs=[a, b],
        outputs=[output],
        attributes={
            "transpose_a": transpose_a,
            "transpose_b": transpose_b
        }
    )
    func.add_node(node)
    return output

def create_conv2d(
    func: CGCFunction,
    input: CGCTensor,
    weight: CGCTensor,
    bias: Optional[CGCTensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    name: Optional[str] = None
) -> CGCTensor:
    input_shape = input.type.shape.dims
    weight_shape = weight.type.shape.dims
    
    if len(input_shape) != 4:
        raise ValueError(f"Conv2D input must be 4D, got {len(input_shape)}D")
    if len(weight_shape) != 4:
        raise ValueError(f"Conv2D weight must be 4D, got {len(weight_shape)}D")
    
    n, c_in, h_in, w_in = input_shape
    out_channels, _, k_h, k_w = weight_shape
    
    if c_in % groups != 0:
        raise ValueError(f"Input channels {c_in} must be divisible by groups {groups}")
    
    h_out = (h_in + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    w_out = (w_in + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    
    output_shape = Shape(dims=(n, out_channels, h_out, w_out))
    output = CGCTensor(
        name=name or f"{input.name}_conv2d",
        type=CGCType(dtype=input.type.dtype, shape=output_shape)
    )
    
    inputs = [input, weight]
    if bias is not None:
        inputs.append(bias)
    
    node = CGCNode(
        op_type="Conv2D",
        name=name,
        inputs=inputs,
        outputs=[output],
        attributes={
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups
        }
    )
    func.add_node(node)
    return output

def create_attention(
    func: CGCFunction,
    query: CGCTensor,
    key: CGCTensor,
    value: CGCTensor,
    causal: bool = False,
    mask: Optional[CGCTensor] = None,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    name: Optional[str] = None
) -> CGCTensor:
    q_shape = query.type.shape.dims
    k_shape = key.type.shape.dims
    v_shape = value.type.shape.dims
    
    if len(q_shape) != 4:
        raise ValueError(f"Query must be 4D (bs, heads, seq, head_dim), got {len(q_shape)}D")
    
    bs, heads, q_seq, head_dim = q_shape
    _, _, k_seq, _ = k_shape
    _, _, v_seq, _ = v_shape
    
    if k_seq != v_seq:
        raise ValueError(f"Key and Value sequence lengths must match: {k_seq} vs {v_seq}")
    
    if scale is None:
        scale = head_dim ** -0.5
    
    output_shape = Shape(dims=(bs, heads, q_seq, head_dim))
    output = CGCTensor(
        name=name or f"{query.name}_attn",
        type=CGCType(dtype=query.type.dtype, shape=output_shape)
    )
    
    inputs = [query, key, value]
    if mask is not None:
        inputs.append(mask)
    
    node = CGCNode(
        op_type="Attention",
        name=name,
        inputs=inputs,
        outputs=[output],
        attributes={
            "causal": causal,
            "dropout": dropout,
            "scale": scale
        }
    )
    func.add_node(node)
    return output

def create_layer_norm(
    func: CGCFunction,
    input: CGCTensor,
    weight: Optional[CGCTensor] = None,
    bias: Optional[CGCTensor] = None,
    normalized_shape: Optional[Tuple[int, ...]] = None,
    eps: float = 1e-5,
    name: Optional[str] = None
) -> CGCTensor:
    input_shape = input.type.shape.dims
    
    if normalized_shape is None:
        normalized_shape = input_shape[-1:] if len(input_shape) >= 2 else input_shape
    
    output = CGCTensor(
        name=name or f"{input.name}_ln",
        type=CGCType(dtype=input.type.dtype, shape=input.type.shape)
    )
    
    inputs = [input]
    if weight is not None:
        inputs.append(weight)
    if bias is not None:
        inputs.append(bias)
    
    node = CGCNode(
        op_type="LayerNorm",
        name=name,
        inputs=inputs,
        outputs=[output],
        attributes={
            "normalized_shape": normalized_shape,
            "eps": eps
        }
    )
    func.add_node(node)
    return output

def create_linear(
    func: CGCFunction,
    input: CGCTensor,
    weight: CGCTensor,
    bias: Optional[CGCTensor] = None,
    name: Optional[str] = None
) -> CGCTensor:
    input_shape = input.type.shape.dims
    weight_shape = weight.type.shape.dims
    
    if len(input_shape) < 2:
        raise ValueError(f"Linear input must have at least 2D, got {len(input_shape)}D")
    if len(weight_shape) != 2:
        raise ValueError(f"Linear weight must be 2D, got {len(weight_shape)}D")
    
    in_features = input_shape[-1]
    out_features, weight_in = weight_shape
    
    if in_features != weight_in:
        raise ValueError(f"Linear dimension mismatch: in_features={in_features}, weight_in={weight_in}")
    
    output_shape_dims = list(input_shape[:-1]) + [out_features]
    output = CGCTensor(
        name=name or f"{input.name}_linear",
        type=CGCType(dtype=input.type.dtype, shape=Shape(dims=tuple(output_shape_dims)))
    )
    
    inputs = [input, weight]
    if bias is not None:
        inputs.append(bias)
    
    node = CGCNode(
        op_type="Linear",
        name=name,
        inputs=inputs,
        outputs=[output]
    )
    func.add_node(node)
    return output

def create_reshape(
    func: CGCFunction,
    input: CGCTensor,
    shape: Tuple[int, ...],
    name: Optional[str] = None
) -> CGCTensor:
    input_shape = input.type.shape.dims
    input_numel = 1
    for dim in input_shape:
        if dim > 0:
            input_numel *= dim
    
    output_numel = 1
    dynamic_dims = []
    for i, dim in enumerate(shape):
        if dim == 0:
            dynamic_dims.append((i, shape.index(0)))
        elif dim > 0:
            output_numel *= dim
    
    if -1 in shape:
        num_ones = shape.count(1)
        output_numel = input_numel // (output_numel // (1 if num_ones == 0 else 1))
    
    output = CGCTensor(
        name=name or f"{input.name}_reshape",
        type=CGCType(dtype=input.type.dtype, shape=Shape(dims=shape))
    )
    node = CGCNode(
        op_type="Reshape",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"shape": shape}
    )
    func.add_node(node)
    return output

def create_transpose(
    func: CGCFunction,
    input: CGCTensor,
    dim0: int,
    dim1: int,
    name: Optional[str] = None
) -> CGCTensor:
    input_shape = list(input.type.shape.dims)
    input_shape[dim0], input_shape[dim1] = input_shape[dim1], input_shape[dim0]
    
    output = CGCTensor(
        name=name or f"{input.name}_transpose",
        type=CGCType(dtype=input.type.dtype, shape=Shape(dims=tuple(input_shape)))
    )
    node = CGCNode(
        op_type="Transpose",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"dim0": dim0, "dim1": dim1}
    )
    func.add_node(node)
    return output

def create_gelu(
    func: CGCFunction,
    input: CGCTensor,
    approximate: str = "none",
    name: Optional[str] = None
) -> CGCTensor:
    output = CGCTensor(
        name=name or f"{input.name}_gelu",
        type=CGCType(dtype=input.type.dtype, shape=input.type.shape)
    )
    node = CGCNode(
        op_type="GELU",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"approximate": approximate}
    )
    func.add_node(node)
    return output

def create_softmax(
    func: CGCFunction,
    input: CGCTensor,
    dim: int = -1,
    name: Optional[str] = None
) -> CGCTensor:
    output = CGCTensor(
        name=name or f"{input.name}_softmax",
        type=CGCType(dtype=input.type.dtype, shape=input.type.shape)
    )
    node = CGCNode(
        op_type="Softmax",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"dim": dim}
    )
    func.add_node(node)
    return output

def create_moe(
    func: CGCFunction,
    input: CGCTensor,
    weights: CGCTensor,
    expert_weights: List[CGCTensor],
    expert_biases: Optional[List[CGCTensor]] = None,
    num_experts: int = 8,
    top_k: int = 2,
    name: Optional[str] = None
) -> CGCTensor:
    input_shape = input.type.shape.dims
    batch_size, seq_len, hidden_dim = input_shape
    
    output = CGCTensor(
        name=name or f"{input.name}_moe",
        type=CGCType(dtype=input.type.dtype, shape=Shape(dims=input_shape))
    )
    
    inputs = [input, weights] + expert_weights
    if expert_biases is not None:
        inputs.extend(expert_biases)
    
    node = CGCNode(
        op_type="MoE",
        name=name,
        inputs=inputs,
        outputs=[output],
        attributes={
            "num_experts": num_experts,
            "top_k": top_k
        }
    )
    func.add_node(node)
    return output

def create_reduction(
    func: CGCFunction,
    input: CGCTensor,
    reduction_type: str,
    dims: Optional[Tuple[int, ...]] = None,
    keepdim: bool = False,
    name: Optional[str] = None
) -> CGCTensor:
    input_shape = input.type.shape.dims
    
    if dims is None:
        dims = tuple(range(len(input_shape)))
    
    output_shape_dims = []
    for i, dim in enumerate(input_shape):
        if i in dims and not keepdim:
            continue
        output_shape_dims.append(dim)
    
    output = CGCTensor(
        name=name or f"{input.name}_{reduction_type}",
        type=CGCType(dtype=input.type.dtype, shape=Shape(dims=tuple(output_shape_dims)))
    )
    node = CGCNode(
        op_type=reduction_type.capitalize(),
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"dims": dims, "keepdim": keepdim}
    )
    func.add_node(node)
    return output

def create_mean(
    func: CGCFunction,
    input: CGCTensor,
    dims: Optional[Tuple[int, ...]] = None,
    keepdim: bool = False,
    name: Optional[str] = None
) -> CGCTensor:
    return create_reduction(func, input, "Mean", dims, keepdim, name)

def create_sum(
    func: CGCFunction,
    input: CGCTensor,
    dims: Optional[Tuple[int, ...]] = None,
    keepdim: bool = False,
    name: Optional[str] = None
) -> CGCTensor:
    return create_reduction(func, input, "Sum", dims, keepdim, name)

def create_max(
    func: CGCFunction,
    input: CGCTensor,
    dims: Optional[Tuple[int, ...]] = None,
    keepdim: bool = False,
    name: Optional[str] = None
) -> CGCTensor:
    return create_reduction(func, input, "Max", dims, keepdim, name)

def create_cat(
    func: CGCFunction,
    tensors: List[CGCTensor],
    dim: int = 0,
    name: Optional[str] = None
) -> CGCTensor:
    if not tensors:
        raise ValueError("Cat requires at least one tensor")
    
    first_shape = tensors[0].type.shape.dims
    dtype = tensors[0].type.dtype
    
    for tensor in tensors[1:]:
        if tensor.type.dtype != dtype:
            raise ValueError(f"Cat tensors must have same dtype")
        
        tensor_shape = tensor.type.shape.dims
        for i, (d1, d2) in enumerate(zip(first_shape, tensor_shape)):
            if i != dim and d1 != d2:
                raise ValueError(f"Cat shape mismatch at dim {i}")
    
    output_shape = list(first_shape)
    for tensor in tensors:
        output_shape[dim] += tensor.type.shape.dims[dim]
    
    output = CGCTensor(
        name=name or "cat",
        type=CGCType(dtype=dtype, shape=Shape(dims=tuple(output_shape)))
    )
    node = CGCNode(
        op_type="Cat",
        name=name,
        inputs=tensors,
        outputs=[output],
        attributes={"dim": dim}
    )
    func.add_node(node)
    return output

def create_slice(
    func: CGCFunction,
    input: CGCTensor,
    dim: int,
    start: int,
    end: int,
    step: int = 1,
    name: Optional[str] = None
) -> CGCTensor:
    input_shape = list(input.type.shape.dims)
    output_shape = input_shape.copy()
    output_shape[dim] = (end - start + step - 1) // step
    
    output = CGCTensor(
        name=name or f"{input.name}_slice",
        type=CGCType(dtype=input.type.dtype, shape=Shape(dims=tuple(output_shape)))
    )
    node = CGCNode(
        op_type="Slice",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"dim": dim, "start": start, "end": end, "step": step}
    )
    func.add_node(node)
    return output

def create_gds_copy(
    func: CGCFunction,
    input: CGCTensor,
    name: Optional[str] = None
) -> CGCTensor:
    output = CGCTensor(
        name=name or f"{input.name}_gds_copy",
        type=CGCType(dtype=input.type.dtype, shape=input.type.shape)
    )
    node = CGCNode(
        op_type="GDSCopy",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"zero_copy": True}
    )
    func.add_node(node)
    return output

def create_spdk_read(
    func: CGCFunction,
    input: CGCTensor,
    offset: int = 0,
    size: Optional[int] = None,
    name: Optional[str] = None
) -> CGCTensor:
    input_shape = input.type.shape.dims
    output_shape = input_shape if size is None else Shape(dims=(size,))
    
    output = CGCTensor(
        name=name or f"{input.name}_spdk_read",
        type=CGCType(dtype=input.type.dtype, shape=output_shape)
    )
    node = CGCNode(
        op_type="SPDKRead",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"offset": offset, "size": size}
    )
    func.add_node(node)
    return output

def create_spdk_write(
    func: CGCFunction,
    input: CGCTensor,
    offset: int = 0,
    name: Optional[str] = None
) -> CGCTensor:
    output = CGCTensor(
        name=name or f"{input.name}_spdk_write",
        type=CGCType(dtype=DType.BOOL, shape=Shape(dims=(1,)))
    )
    node = CGCNode(
        op_type="SPDKWrite",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"offset": offset}
    )
    func.add_node(node)
    return output

def create_flash_moe(
    func: CGCFunction,
    input: CGCTensor,
    expert_weights: List[CGCTensor],
    gate_weights: CGCTensor,
    num_experts: int = 8,
    top_k: int = 2,
    name: Optional[str] = None
) -> CGCTensor:
    output = CGCTensor(
        name=name or f"{input.name}_flash_moe",
        type=CGCType(dtype=input.type.dtype, shape=input.type.shape)
    )
    node = CGCNode(
        op_type="FlashMoE",
        name=name,
        inputs=[input, gate_weights] + expert_weights,
        outputs=[output],
        attributes={
            "num_experts": num_experts,
            "top_k": top_k,
            "optimized": True
        }
    )
    func.add_node(node)
    return output

def create_rswa(
    func: CGCFunction,
    query: CGCTensor,
    key: CGCTensor,
    value: CGCTensor,
    reference_key: Optional[CGCTensor] = None,
    reference_value: Optional[CGCTensor] = None,
    causal: bool = False,
    name: Optional[str] = None
) -> CGCTensor:
    q_shape = query.type.shape.dims
    output_shape = Shape(dims=q_shape)
    
    output = CGCTensor(
        name=name or f"{query.name}_rswa",
        type=CGCType(dtype=query.type.dtype, shape=output_shape)
    )
    
    inputs = [query, key, value]
    if reference_key is not None:
        inputs.append(reference_key)
    if reference_value is not None:
        inputs.append(reference_value)
    
    node = CGCNode(
        op_type="R-SWA",
        name=name,
        inputs=inputs,
        outputs=[output],
        attributes={
            "causal": causal,
            "has_reference": reference_key is not None
        }
    )
    func.add_node(node)
    return output

def create_nfs_rdma(
    func: CGCFunction,
    input: CGCTensor,
    remote_addr: str = "",
    name: Optional[str] = None
) -> CGCTensor:
    output = CGCTensor(
        name=name or f"{input.name}_nfs_rdma",
        type=CGCType(dtype=input.type.dtype, shape=input.type.shape)
    )
    node = CGCNode(
        op_type="NFSoRDMA",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"remote_addr": remote_addr, "direct_io": True}
    )
    func.add_node(node)
    return output

def create_omlx(
    func: CGCFunction,
    input: CGCTensor,
    operation: str = "default",
    name: Optional[str] = None
) -> CGCTensor:
    output = CGCTensor(
        name=name or f"{input.name}_omlx_{operation}",
        type=CGCType(dtype=input.type.dtype, shape=input.type.shape)
    )
    node = CGCNode(
        op_type="OMLX",
        name=name,
        inputs=[input],
        outputs=[output],
        attributes={"operation": operation}
    )
    func.add_node(node)
    return output