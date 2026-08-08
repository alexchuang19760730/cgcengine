import torch
import numpy as np
# Assuming kv_cache_pb2 is generated in cgc_engine/proto
import proto.kv_cache_pb2 as kv_cache

class KVConverter:
    """
    CGC端云统一KV转换工具
    适配：vLLM(B, H, S, D) <--> llama.cpp(H, S, D)
    兼容：KDA零扩增KV + δ-mem记忆增量特征
    """
    @staticmethod
    def from_vllm(vllm_kv, delta_mem_incr, layer_id, model_info):
        key, value = vllm_kv
        B, H, S, D = key.shape
        return kv_cache.KVLayer(
            layer_id=layer_id,
            num_heads=H,
            head_dim=D,
            seq_len=S,
            dtype="bf16",
            key_data=key.cpu().to(torch.float16).ravel().numpy().tobytes(),
            value_data=value.cpu().to(torch.float16).ravel().numpy().tobytes(),
            delta_mem_incr=delta_mem_incr.tobytes(),
            mem_decay_factor=0.95
        )

    @staticmethod
    def to_vllm(layer: kv_cache.KVLayer):
        H = layer.num_heads
        S = layer.seq_len
        D = layer.head_dim
        key = torch.from_numpy(np.frombuffer(layer.key_data, dtype=np.float16)).view(1, H, S, D).bfloat16()
        value = torch.from_numpy(np.frombuffer(layer.value_data, dtype=np.float16)).view(1, H, S, D).bfloat16()
        # 加载δ-mem记忆增量修正
        delta_incr = np.frombuffer(layer.delta_mem_incr, dtype=np.float16)
        return (key, value, delta_incr)

    @staticmethod
    def from_llama_cpp(llama_kv, delta_mem_incr, layer_id, model_info):
        key, value = llama_kv
        H, S, D = key.shape
        return kv_cache.KVLayer(
            layer_id=layer_id,
            num_heads=H,
            head_dim=D,
            seq_len=S,
            dtype="f16",
            key_data=key.tobytes(),
            value_data=value.tobytes(),
            delta_mem_incr=delta_mem_incr.tobytes(),
            mem_decay_factor=0.95
        )

    @staticmethod
    def to_llama_cpp(layer: kv_cache.KVLayer):
        H = layer.num_heads
        S = layer.seq_len
        D = layer.head_dim
        key = np.frombuffer(layer.key_data, dtype=np.float16).reshape(H, S, D)
        value = np.frombuffer(layer.value_data, dtype=np.float16).reshape(H, S, D)
        delta_incr = np.frombuffer(layer.delta_mem_incr, dtype=np.float16)
        return (key, value, delta_incr)

    @staticmethod
    def vllm_to_llama_cpp(vllm_kv):
        key, value = vllm_kv
        B, H, S, D = key.shape
        key = key.squeeze(0).permute(0, 1, 2).cpu().numpy().astype(np.float16)
        value = value.squeeze(0).permute(0, 1, 2).cpu().numpy().astype(np.float16)
        return (key, value)

    @staticmethod
    def llama_cpp_to_vllm(llama_kv):
        key, value = llama_kv
        H, S, D = key.shape
        key = torch.from_numpy(key).bfloat16().unsqueeze(0)
        value = torch.from_numpy(value).bfloat16().unsqueeze(0)
        return (key, value)
