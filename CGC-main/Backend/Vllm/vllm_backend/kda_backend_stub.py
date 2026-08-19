"""
CGC KDA Attention Backend - 独立文件以避免循环导入
"""
import sys
import os
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[4])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# 延迟导入 vLLM 以避免循环导入
def _get_flash_backend():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    return FlashAttentionBackend

class CGCKDAAttentionBackend(_get_flash_backend()):
    @staticmethod
    def get_impl_cls():
        # 延迟导入以避免循环导入
        from Backend.Vllm.vllm_backend.cgc_kda_backend import CGCKDAImpl
        return CGCKDAImpl

    @staticmethod
    def get_name():
        return "FLASH_ATTN"
