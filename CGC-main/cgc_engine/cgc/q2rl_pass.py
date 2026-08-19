import torch
import torch.fx as fx

class InsertQ2RLPass:
    """
    Q2RL Strategy Vector Injection Pass
    在端侧 Decode 的最后阶段 (Logits 采样前)，插入云端传来的 Q2RL 价值向量，
    以此来引导模型生成更高价值的 Token 或 Agent 动作。
    这实现了在不训练主干网络的情况下，赋予端侧主动强化学习决策能力。
    """
    def __init__(self, enable_q2rl=True, strategy_vector_dim=None):
        self.enable_q2rl = enable_q2rl
        self.strategy_vector_dim = strategy_vector_dim

    def __call__(self, graph_module: fx.GraphModule) -> fx.GraphModule:
        if not self.enable_q2rl:
            return graph_module

        # 遍历计算图，找到输出节点前的 Logits 张量
        graph = graph_module.graph
        
        # 寻找 output 节点
        output_node = None
        for node in graph.nodes:
            if node.op == 'output':
                output_node = node
                break
                
        if output_node is None:
            return graph_module

        # 假设 output_node 的参数 (args) 中的第一个就是 logits
        # 实际情况中可能是个 tuple，需要针对特定模型的输出格式做适配
        logits_node = output_node.args[0]
        if isinstance(logits_node, tuple):
            logits_node = logits_node[0]

        # 插入一个占位符或自定算子代表从云端传来的 Q-strategy 向量
        # q2rl_vector: shape (vocab_size,) or (action_space,)
        with graph.inserting_before(output_node):
            # 获取一个运行时的 q2rl 偏置参数
            q2rl_vector = graph.call_function(
                torch.ops.cgc.get_q2rl_strategy_vector.default,
                args=(),
                kwargs={}
            )
            
            # alpha 权重系数 (控制 Q 值的引导强度)
            alpha = graph.call_function(
                torch.ops.cgc.get_q2rl_alpha.default,
                args=(),
                kwargs={}
            )
            
            # 缩放 Q 值向量
            scaled_q2rl = graph.call_function(
                torch.mul,
                args=(q2rl_vector, alpha)
            )
            
            # logits = logits + alpha * Q_vector
            biased_logits = graph.call_function(
                torch.add,
                args=(logits_node, scaled_q2rl)
            )
            
            # 将 output 节点的输入替换为加上 bias 后的 logits
            if isinstance(output_node.args[0], tuple):
                new_args = list(output_node.args[0])
                new_args[0] = biased_logits
                output_node.args = (tuple(new_args),)
            else:
                output_node.args = (biased_logits,)

        graph.lint()
        graph_module.recompile()
        
        print(f"[CGC] Q2RL Operator Injection Pass Applied: Biased logits with Q-strategy vector.")
        return graph_module

# 注册假算子以骗过 PyTorch 追踪
try:
    torch.library.define("cgc::get_q2rl_strategy_vector", "() -> Tensor")
    @torch.library.impl("cgc::get_q2rl_strategy_vector", "default")
    def _get_q2rl_strategy_vector():
        # Fake implementation
        return torch.zeros((1,), dtype=torch.float32)

    torch.library.define("cgc::get_q2rl_alpha", "() -> float")
    @torch.library.impl("cgc::get_q2rl_alpha", "default")
    def _get_q2rl_alpha():
        return 1.0
except Exception as e:
    pass # 忽略重复注册
