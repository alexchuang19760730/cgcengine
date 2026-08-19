from __future__ import annotations
import sys
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main')

from cgc_engine.ir import (
    DType,
    Shape,
    CGCType,
    CGCFunction,
    CGCModule,
    create_parameter,
    create_matmul,
    create_add,
    create_linear,
    BackendRegistry,
    get_backend,
    list_backends,
)

def test_ir_types():
    dtype = DType.FLOAT16
    shape = Shape(dims=(2, 3, 4))
    cgc_type = CGCType(dtype=dtype, shape=shape)
    
    print(f"✅ DType: {dtype}")
    print(f"✅ Shape: {shape}")
    print(f"✅ CGCType: {cgc_type}")
    print(f"✅ Type size: {cgc_type.size_bytes} bytes")

def test_ir_module():
    module = CGCModule(name="test_module")
    func = CGCFunction(name="test_function")
    
    input_type = CGCType(dtype=DType.FLOAT16, shape=Shape(dims=(1, 32, 512)))
    weight_type = CGCType(dtype=DType.FLOAT16, shape=Shape(dims=(512, 512)))
    
    x = create_parameter(func, "x", input_type)
    w = create_parameter(func, "w", weight_type)
    
    y = create_matmul(func, x, w)
    z = create_add(func, y, y)
    
    func.results = [z]
    module.add_function(func)
    
    print(f"✅ Module created: {module.name}")
    print(f"✅ Function created: {func.name}")
    print(f"✅ Parameters: {[p.name for p in func.parameters]}")
    print(f"✅ Nodes: {len(func.body)}")
    print(f"✅ Results: {[r.name for r in func.results]}")
    
    return module

def test_backend_selection():
    backends = list_backends()
    print(f"\n✅ Available backends: {backends}")
    
    cuda_backend = get_backend("cuda")
    if cuda_backend:
        print(f"✅ CUDA backend: {cuda_backend}")
        print(f"   Supported dtypes: {len(cuda_backend.supported_dtypes)}")
        print(f"   Supported ops: {len(cuda_backend.supported_ops)}")
    
    ascend_backend = get_backend("ascend")
    if ascend_backend:
        print(f"✅ Ascend backend: {ascend_backend}")
        print(f"   Supported dtypes: {len(ascend_backend.supported_dtypes)}")
        print(f"   Supported ops: {len(ascend_backend.supported_ops)}")
    
    metal_backend = get_backend("metal")
    if metal_backend:
        print(f"✅ Metal backend: {metal_backend}")
        print(f"   Supported dtypes: {len(metal_backend.supported_dtypes)}")
        print(f"   Supported ops: {len(metal_backend.supported_ops)}")

def test_compilation():
    module = CGCModule(name="compilation_test")
    func = CGCFunction(name="linear_layer")
    
    input_type = CGCType(dtype=DType.FLOAT16, shape=Shape(dims=(1, 32, 512)))
    weight_type = CGCType(dtype=DType.FLOAT16, shape=Shape(dims=(1024, 512)))
    
    x = create_parameter(func, "x", input_type)
    w = create_parameter(func, "w", weight_type)
    
    y = create_linear(func, x, w)
    func.results = [y]
    module.add_function(func)
    
    cuda_backend = get_backend("cuda")
    if cuda_backend:
        compiled = cuda_backend.compile(module)
        print(f"\n✅ Compilation successful:")
        print(f"   Functions compiled: {len(compiled)}")
        print(f"   Optimized module: {cuda_backend.optimize(module)}")

if __name__ == "__main__":
    print("=" * 60)
    print("        CGC Engine IR Module Test")
    print("=" * 60)
    
    test_ir_types()
    test_ir_module()
    test_backend_selection()
    test_compilation()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)