#!/usr/bin/env python3
import sys

content = open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py').read()

add_code = '''

# ==============================================
# CGC KDA Auto-Registration
# ==============================================
import os

def _try_register_cgc_kda():
    """Auto-register CGC KDA backend if environment variable is set."""
    if os.environ.get("VLLM_USE_CGC_KDA", "0") == "1":
        try:
            import sys
            sys.path.insert(0, "/home/gs01")
            from vllm_attention_backend.cgc_kda_backend import CGCKDAAttentionBackend
            register_backend(AttentionBackendEnum.FLASH_ATTN, 
                           "vllm_attention_backend.cgc_kda_backend.CGCKDAAttentionBackend")
            print("[CGC-KDA] Auto-registered KDA backend (VLLM_USE_CGC_KDA=1)")
        except Exception as e:
            print(f"[CGC-KDA] Failed to auto-register: {e}")

_try_register_cgc_kda()
'''

open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py', 'w').write(content + add_code)
print("Done")
