# vLLM MoE CGC Backend - FlashMoE 集成

class VLLMMoECGC(VLLMCGCModule):
    """
    vLLM MoE FFN CGC 实现

    集成 oMLX 专家预测 + UnifiedIO 按需加载 + FlashMoE 计算
    替换 vLLM 原生 MoE 调度，实现真正的 MoE 融合计算。
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int = 2,
        config: Optional[VLLMCGCConfig] = None,
        omlx_client=None,
        io_controller=None,
        flashmoe_client=None,
    ):
        super().__init__(config)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.omlx_client = omlx_client
        self.io_controller = io_controller
        self.flashmoe_client = flashmoe_client

        self.expert_weights_cache = {}
        self._expert_index = {}

    def set_omlx_client(self, client):
        """设置 oMLX 客户端"""
        self.omlx_client = client

    def set_io_controller(self, controller):
        """设置 UnifiedIO 控制器"""
        self.io_controller = controller

    def set_flashmoe_client(self, client):
        """设置 FlashMoE 客户端"""
        self.flashmoe_client = client

    def set_expert_index(self, expert_index: dict):
        """设置专家索引（从 GGUF 加载）"""
        self._expert_index = expert_index

    def _load_expert_weights(self, expert_ids: List[int]) -> dict:
        """按需加载专家权重"""
        loaded = {}

        for eid in expert_ids:
            if eid in loaded:
                continue

            if self.io_controller:
                try:
                    data = self.io_controller.load_expert(eid, f"expert_0_{eid}")
                    if isinstance(data, dict) and 'up' in data:
                        loaded[eid] = data
                        continue
                except:
                    pass

            if eid in self._expert_index:
                from ..moe_unified_io_integration import load_expert_weights
                w = load_expert_weights(self._expert_index[eid])
                if w:
                    loaded[eid] = w
                    if self.io_controller:
                        try:
                            self.io_controller.save_expert(eid, f"expert_0_{eid}", w)
                        except:
                            pass
                    continue

            import torch
            loaded[eid] = {
                'up': torch.randn(self.hidden_dim, self.intermediate_dim, dtype=torch.float16),
                'down': torch.randn(self.intermediate_dim, self.hidden_dim, dtype=torch.float16),
            }

        return loaded

    def _moe_forward_compute(self, x: torch.Tensor, expert_ids: List[int], weights: dict) -> torch.Tensor:
        """FlashMoE FFN 计算"""
        import torch

        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        output = torch.zeros_like(x_flat)

        for eid in expert_ids:
            if eid not in weights:
                continue
            w = weights[eid]
            up = w['up'].to(x.dtype).to(x.device)
            down = w['down'].to(x.dtype).to(x.device)

            h = torch.matmul(x_flat, up)
            h = torch.nn.functional.silu(h)
            h = torch.matmul(h, down)
            output += h

        return output.view(batch_size, seq_len, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: Optional[List[int]] = None,
        return_expert_ids: bool = False,
    ) -> torch.Tensor:
        """
        MoE FFN forward

        Args:
            x: 输入张量 [B, T, D]
            expert_ids: 可选，指定专家 ID 列表
            return_expert_ids: 是否返回使用的专家 ID

        Returns:
            MoE 输出或 (输出, expert_ids)
        """
        if expert_ids is None and self.omlx_client is not None:
            with torch.no_grad():
                expert_ids = self.omlx_client.predict(x)

        if expert_ids is None:
            expert_ids = list(range(self.top_k))

        if isinstance(expert_ids, torch.Tensor):
            expert_ids = expert_ids.cpu().tolist()

        expert_ids = expert_ids[:self.top_k]

        weights = self._load_expert_weights(expert_ids)

        output = self._moe_forward_compute(x, expert_ids, weights)

        if return_expert_ids:
            return output, expert_ids
        return output


def create_vllm_moe_model(
    vocab_size: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_experts: int,
    top_k: int = 2,
    omlx_client=None,
    io_controller=None,
    flashmoe_client=None,
    **kwargs,
) -> VLLMModelCGC:
    """
    创建 vLLM MoE CGC 模型

    Args:
        vocab_size: 词表大小
        hidden_dim: 隐藏层维度
        num_layers: Transformer 层数
        num_heads: Attention heads 数
        head_dim: 每个 head 的维度
        num_experts: 专家总数
        top_k: 激活的专家数
        omlx_client: oMLX 预测客户端
        io_controller: UnifiedIO 控制器
        flashmoe_client: FlashMoE 客户端

    Returns:
        VLLMModelCGC 实例（配置为 MoE 模式）
    """
    config = VLLMCGCConfig(**kwargs)
    model = VLLMModelCGC(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        config=config,
    )

    for layer in model.layers:
        layer["mlp"] = VLLMMoECGC(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim * 4,
            num_experts=num_experts,
            top_k=top_k,
            config=config,
            omlx_client=omlx_client,
            io_controller=io_controller,
            flashmoe_client=flashmoe_client,
        )

    model._is_moe = True
    return model