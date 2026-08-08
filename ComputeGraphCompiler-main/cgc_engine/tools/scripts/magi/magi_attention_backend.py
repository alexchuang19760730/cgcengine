"""
MagiCompiler Phase 3: KDA/FA2 注意力後端深度集成
Deep Integration with KDA/FA2 Attention Backends
"""

import os
import torch
from typing import Optional, Dict, Any, Tuple, Union, List
from dataclasses import dataclass

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import flash_kda
    HAS_FLASH_KDA = True
except ImportError:
    HAS_FLASH_KDA = False


@dataclass
class AttentionConfig:
    """注意力配置"""
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    max_seq_len: int = 8192
    dtype: torch.dtype = torch.bfloat16
    causal: bool = True
    use_flash_attn: bool = True
    use_kda: bool = False


class MagiAttentionBackend:
    """
    MagiCompiler 注意力後端
    支持 FA2 和 KDA 兩種實現，集成 CUDA Graph 優化
    """

    def __init__(self, config: Optional[AttentionConfig] = None):
        """
        初始化注意力後端

        Args:
            config: 注意力配置
        """
        self.config = config or AttentionConfig()

        # 編譯後的注意力函數
        self.compiled_forward = None

        # CUDA Graph 緩存
        self.graph_cache: Dict[int, torch.cuda.CUDAGraph] = {}
        self.input_placeholders: Dict[int, torch.Tensor] = {}
        self.output_placeholders: Dict[int, torch.Tensor] = {}

        # 統計信息
        self.stats = {
            "graph_hits": 0,
            "graph_misses": 0,
            "compile_time_ms": 0,
        }

        # 初始化注意力函數
        self._initialize_attention()

    def _initialize_attention(self):
        """初始化注意力函數"""
        if self.config.use_kda and HAS_FLASH_KDA:
            self._setup_kda()
        elif self.config.use_flash_attn and HAS_FLASH_ATTN:
            self._setup_flash_attn()
        else:
            self._setup_fallback()

    def _setup_kda(self):
        """設置 KDA 注意力"""
        print("[MagiAttention] ✅ 使用 KDA (Kimi Delta Attention)")

        def kda_forward(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            **kwargs
        ) -> torch.Tensor:
            # KDA 需要 bfloat16
            orig_dtype = query.dtype
            q, k, v = query.to(torch.bfloat16), key.to(torch.bfloat16), value.to(torch.bfloat16)

            output = flash_kda.kda_attention(
                q, k, v,
                causal=self.config.causal,
                head_dim=self.config.head_dim,
            )

            return output.to(orig_dtype)

        self.forward_func = kda_forward
        self.name = "kda"

    def _setup_flash_attn(self):
        """設置 Flash Attention 2"""
        print("[MagiAttention] ✅ 使用 Flash Attention 2")

        def fa2_forward(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            **kwargs
        ) -> torch.Tensor:
            causal = kwargs.get('causal', self.config.causal)
            softmax_scale = kwargs.get('softmax_scale', None)

            return flash_attn_func(
                query, key, value,
                causal=causal,
                softmax_scale=softmax_scale
            )

        self.forward_func = fa2_forward
        self.name = "flash_attn_2"

    def _setup_fallback(self):
        """設置後備注意力實現"""
        print("[MagiAttention] ⚠️ 使用 PyTorch 原生注意力 (後備)")

        def fallback_forward(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            **kwargs
        ) -> torch.Tensor:
            causal = kwargs.get('causal', self.config.causal)
            scale = kwargs.get('scale', 1.0 / (self.config.head_dim ** 0.5))

            # 標準注意力計算
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale

            if causal:
                seq_len = query.size(-2)
                mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1)
                scores = scores.masked_fill(mask == 1, float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, value)

            return output

        self.forward_func = fallback_forward
        self.name = "pytorch"

    def compile(self, mode: str = "reduce-overhead", **compile_kwargs):
        """
        編譯注意力函數

        Args:
            mode: torch.compile 模式
            compile_kwargs: 額外編譯參數
        """
        if self.compiled_forward is not None:
            print("[MagiAttention] 注意力函數已編譯")
            return

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        self.compiled_forward = torch.compile(
            self.forward_func,
            mode=mode,
            fullgraph=True,
            dynamic=True,
            **compile_kwargs
        )
        end_time.record()
        torch.cuda.synchronize()

        self.stats["compile_time_ms"] = start_time.elapsed_time(end_time)
        print(f"[MagiAttention] ✅ 注意力函數編譯完成 ({self.stats['compile_time_ms']:.2f} ms)")

    def capture_graph(
        self,
        seq_len: int,
        sample_query: torch.Tensor,
        sample_key: torch.Tensor,
        sample_value: torch.Tensor,
        **kwargs
    ) -> torch.cuda.CUDAGraph:
        """
        捕獲指定序列長度的 CUDA Graph

        Args:
            seq_len: 序列長度
            sample_query: 樣本 query
            sample_key: 樣本 key
            sample_value: 樣本 value
            kwargs: 額外參數

        Returns:
            捕獲的 CUDA Graph
        """
        if seq_len in self.graph_cache:
            return self.graph_cache[seq_len]

        # 創建佔位符
        input_placeholder_q = sample_query.clone().detach().requires_grad_(False).cuda()
        input_placeholder_k = sample_key.clone().detach().requires_grad_(False).cuda()
        input_placeholder_v = sample_value.clone().detach().requires_grad_(False).cuda()

        # 預熱
        with torch.no_grad():
            warmup_output = self.forward_func(
                input_placeholder_q,
                input_placeholder_k,
                input_placeholder_v,
                **kwargs
            )

        # 創建輸出佔位符
        output_placeholder = warmup_output.clone().detach().requires_grad_(False).cuda()

        # 捕獲 Graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.forward_func(
                input_placeholder_q,
                input_placeholder_k,
                input_placeholder_v,
                **kwargs
            )
            output_placeholder.copy_(output)

        # 存儲
        self.graph_cache[seq_len] = graph
        self.input_placeholders[seq_len] = (input_placeholder_q, input_placeholder_k, input_placeholder_v)
        self.output_placeholders[seq_len] = output_placeholder

        print(f"[MagiAttention] ✅ 捕獲 CUDA Graph: seq_len={seq_len}")

        return graph

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        use_graph: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        執行注意力計算

        Args:
            query: query 張量
            key: key 張量
            value: value 張量
            use_graph: 是否使用 CUDA Graph
            kwargs: 額外參數

        Returns:
            注意力輸出
        """
        seq_len = query.size(-2)

        # 嘗試使用 CUDA Graph
        if use_graph and seq_len in self.graph_cache:
            self.stats["graph_hits"] += 1

            # 更新輸入
            q_ph, k_ph, v_ph = self.input_placeholders[seq_len]
            q_ph.copy_(query)
            k_ph.copy_(key)
            v_ph.copy_(value)

            # 重放
            self.graph_cache[seq_len].replay()

            return self.output_placeholders[seq_len].clone()

        self.stats["graph_misses"] += 1

        # 使用編譯或原始函數
        if self.compiled_forward is not None:
            return self.compiled_forward(query, key, value, **kwargs)
        else:
            return self.forward_func(query, key, value, **kwargs)


class MagiKVAttentionBackend:
    """
    MagiCompiler KV Cache 注意力後端
    專為 vLLM KV Cache 設計的注意力實現
    """

    def __init__(self, config: Optional[AttentionConfig] = None):
        """
        初始化 KV 注意力後端

        Args:
            config: 注意力配置
        """
        self.config = config or AttentionConfig()
        self.attention_backend = MagiAttentionBackend(config)

        # KV Cache 狀態
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None

        # Prefill/Decode 狀態
        self._is_prefilled = False
        self._current_seq_len = 0

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        use_graph: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Prefill 階段：處理完整的提示序列

        Args:
            query: query 張量 [batch, seq_len, num_heads, head_dim]
            key: key 張量 [batch, seq_len, num_kv_heads, head_dim]
            value: value 張量 [batch, seq_len, num_kv_heads, head_dim]
            use_graph: 是否使用 CUDA Graph

        Returns:
            注意力輸出
        """
        # 捕獲 Prefill Graph（如果需要）
        if use_graph and not self._is_prefilled:
            seq_len = query.size(1)
            self.attention_backend.capture_graph(seq_len, query, key, value, **kwargs)

        # 執行注意力計算
        output = self.attention_backend.forward(query, key, value, use_graph=use_graph, **kwargs)

        # 保存 KV Cache
        self.key_cache = key.clone().detach()
        self.value_cache = value.clone().detach()
        self._current_seq_len = query.size(1)
        self._is_prefilled = True

        return output

    def decode(
        self,
        query: torch.Tensor,
        use_graph: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode 階段：逐 token 生成

        Args:
            query: 當前 token 的 query 張量 [batch, 1, num_heads, head_dim]
            use_graph: 是否使用 CUDA Graph

        Returns:
            注意力輸出
        """
        if not self._is_prefilled:
            raise RuntimeError("需要先調用 prefill()")

        # 更新序列長度
        self._current_seq_len += 1

        # 捕獲 Decode Graph（如果需要）
        if use_graph:
            self.attention_backend.capture_graph(self._current_seq_len, query, self.key_cache, self.value_cache, **kwargs)

        # 執行注意力計算
        output = self.attention_backend.forward(query, self.key_cache, self.value_cache, use_graph=use_graph, **kwargs)

        # 更新 KV Cache
        # 注意：KV Cache 更新在 vLLM 內部處理，這裡只返回注意力輸出
        return output

    def update_kv_cache(self, new_key: torch.Tensor, new_value: torch.Tensor):
        """
        更新 KV Cache

        Args:
            new_key: 新 key [batch, 1, num_kv_heads, head_dim]
            new_value: 新 value [batch, 1, num_kv_heads, head_dim]
        """
        if self.key_cache is None or self.value_cache is None:
            self.key_cache = new_key
            self.value_cache = new_value
        else:
            self.key_cache = torch.cat([self.key_cache, new_key], dim=1)
            self.value_cache = torch.cat([self.value_cache, new_value], dim=1)

    def reset(self):
        """重置狀態"""
        self.key_cache = None
        self.value_cache = None
        self._is_prefilled = False
        self._current_seq_len = 0


def create_attention_backend(
    use_kda: bool = False,
    use_flash_attn: bool = True,
    compile: bool = True,
    **config_kwargs
) -> MagiKVAttentionBackend:
    """
    工廠函數：創建 MagiCompiler 注意力後端

    Args:
        use_kda: 是否使用 KDA
        use_flash_attn: 是否使用 Flash Attention
        compile: 是否編譯
        config_kwargs: 配置參數

    Returns:
        MagiKVAttentionBackend 實例
    """
    config = AttentionConfig(
        use_kda=use_kda,
        use_flash_attn=use_flash_attn,
        **config_kwargs
    )

    backend = MagiKVAttentionBackend(config)

    if compile:
        backend.attention_backend.compile()

    return backend
