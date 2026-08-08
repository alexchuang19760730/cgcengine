import re

with open("/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/cgc/cgc_cpp/src/kernels/ortho_kda_v4_binding.cpp", "r") as f:
    content = f.read()

# Remove #ifdef __CUDACC__ around function bodies in the class
content = re.sub(r'#ifdef __CUDACC__\n(.*?)\n#endif', r'\1', content, flags=re.DOTALL)

with open("/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/cgc/cgc_cpp/src/kernels/ortho_kda_v4_binding.cpp", "w") as f:
    f.write(content)
