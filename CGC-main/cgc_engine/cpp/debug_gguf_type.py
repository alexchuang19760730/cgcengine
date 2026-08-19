#!/usr/bin/env python3
"""Debug GGUF tensor parsing for IQ4_XS type."""

import os
import sys
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_gguf():
    filepath = r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf"
    
    import gguf as _gguf
    reader = _gguf.GGUFReader(filepath)
    
    print(f"Total tensors: {len(reader.tensors)}")
    print(f"Data offset: {reader.data_offset}")
    print()
    
    # Find tensors with issues
    for t in reader.tensors:
        name = t.name
        dims = [int(d) for d in t.shape]
        ggml_type = t.tensor_type.value
        type_name = str(t.tensor_type)
        nbytes = getattr(t, 'nbytes', 0)
        offset = int(t.data_offset)
        
        # Calculate elements
        n_elements = 1
        for d in dims:
            n_elements *= d
        
        if nbytes > 0:
            expected_elements_from_bytes = nbytes  # for non-quantized
            if ggml_type in (0, 1, 30):  # F32, F16, BF16
                bpe = {0: 4, 1: 2, 30: 2}[ggml_type]
                expected_elements = nbytes // bpe
                if expected_elements != n_elements:
                    print(f"MISMATCH (type {ggml_type}): {name}")
                    print(f"  dims={dims}, n_elements={n_elements}")
                    print(f"  nbytes={nbytes}, expected_elements={expected_elements}")
                    print(f"  offset={offset}")
                    print()
        
        if "ffn_down_exps" in name or "ffn_gate_up_exps" in name:
            print(f"{name}: dims={dims}, type={ggml_type} ({type_name}), nbytes={nbytes}, offset={offset}")
            print(f"  n_elements={n_elements}, size_from_elements={n_elements * 4}")

if __name__ == "__main__":
    debug_gguf()
