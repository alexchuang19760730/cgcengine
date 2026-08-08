"""Slime RL GRPO for MTP Head.

用 GRPO (Group Relative Policy Optimization) 优化 MTP head 的 accept rate.

核心思路:
  1. Rollout: 对每个 prompt, 用当前 MTP head (temperature sampling) 生成 G 条 draft 链,
     base model verify, 记录 (hidden, token, sampled_token, logprob, reward=accept_length)
  2. GRPO update: group-relative advantage + PPO clip + KL penalty

Reward 设计:
  - accept_length / num_draft: 归一化 accept rate (0~1)
  - 全部 accept bonus: +0.2 (鼓励长链 accept)
  - reject penalty: 0 (不额外惩罚, accept_length=0 已是惩罚)

GRPO advantage:
  A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

参考: DeepSeekMath GRPO, Slime RL framework
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repo root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import llama_cpp
from mtp_head.model import MTPHead, MTPHeadConfig


# ============================================================
# Config
# ============================================================

@dataclass
class GRPOConfig:
    """GRPO 训练配置."""
    # Model
    gguf_path: str = "/Users/alexchuang/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf"
    checkpoint: str = ""  # 初始 MTP head checkpoint (v4 scheduled sampling)
    embed_head_path: str = ""  # embed_head.pt
    hidden_size: int = 896
    vocab_size: int = 151936
    num_heads: int = 14
    head_dim: int = 64
    intermediate_size: int = 4864

    # GRPO
    num_draft: int = 4  # N=4 draft chain
    group_size: int = 8  # G: 每个 prompt 采样 G 条 draft 链
    num_prompts_per_iter: int = 4  # 每次迭代采样几个 prompt
    num_iterations: int = 50  # 总迭代次数
    temperature: float = 1.0  # 采样温度 (rollout 时)
    min_temperature: float = 0.5  # 温度下限 (防止太确定性)

    # PPO
    clip_ratio: float = 0.2  # PPO clip ε
    kl_penalty: float = 0.2  # KL 散度惩罚系数 (防 policy drift)
    lr: float = 1e-5  # 学习率 (RL 用小 lr)
    max_grad_norm: float = 1.0

    # Reward
    accept_bonus: float = 0.2  # 全部 accept 的 bonus
    reject_penalty: float = 0.0  # reject 额外惩罚

    # Device
    device: str = "mps"  # cpu or mps

    # Logging
    output_dir: str = "mtp_output/qwen25_0_5b_slime_rl"
    log_every: int = 1
    save_every: int = 10


# ============================================================
# Rollout Environment
# ============================================================

class MTPRolloutEnv:
    """MTP rollout 环境: 用 base model verify draft tokens.

    核心接口:
      - reset(prompt): prefill, 返回初始 hidden + token
      - step(draft_tokens): verify draft chain, 返回 (accept_length, rewards)

    注意: base model 的 KV cache 在 rollout 中管理, 每次 rollout 后 reset.
    """

    def __init__(self, gguf_path: str, n_ctx: int = 2048):
        print(f"[env] Loading base model: {gguf_path}", flush=True)
        self.llm = llama_cpp.Llama(
            model_path=gguf_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            n_batch=512,
            embedding=True,
            logits_all=False,
            verbose=False,
        )
        self.ctx = self.llm.ctx
        self.n_embd = self.llm.n_embd()
        self.n_vocab = self.llm.n_vocab()
        self.seq_id = 0
        self.n_past = 0
        self.n_ctx = n_ctx
        print(f"[env] n_embd={self.n_embd}, n_vocab={self.n_vocab}", flush=True)

    def _get_last_hidden(self) -> np.ndarray:
        """获取最后一个 token 的 hidden state."""
        emb = llama_cpp.llama_get_embeddings(self.ctx)
        if not emb:
            raise RuntimeError("embeddings is None, ensure embedding=True")
        arr_type = ctypes.c_float * self.n_embd
        return np.ctypeslib.as_array(
            ctypes.cast(emb, ctypes.POINTER(ctypes.c_float)), shape=(self.n_embd,)
        ).copy()

    def _get_last_logits(self) -> np.ndarray:
        """获取最后一个 token 的 logits."""
        logits_ptr = llama_cpp.llama_get_logits(self.ctx)
        arr_type = ctypes.c_float * self.n_vocab
        return np.ctypeslib.as_array(
            logits_ptr, shape=(self.n_vocab,)
        ).copy()

    def _get_hidden_ith(self, i: int) -> np.ndarray:
        emb = llama_cpp.llama_get_embeddings_ith(self.ctx, i)
        if not emb:
            raise RuntimeError(f"embeddings_ith({i}) is None")
        arr_type = ctypes.c_float * self.n_embd
        return np.ctypeslib.as_array(
            ctypes.cast(emb, ctypes.POINTER(ctypes.c_float)), shape=(self.n_embd,)
        ).copy()

    def _get_logits_ith(self, i: int) -> np.ndarray:
        logits_ptr = llama_cpp.llama_get_logits_ith(self.ctx, i)
        arr_type = ctypes.c_float * self.n_vocab
        return np.ctypeslib.as_array(
            logits_ptr, shape=(self.n_vocab,)
        ).copy()

    def _decode_single(self, token_id: int, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """单 token decode, 返回 (hidden, logits)."""
        token_arr = (llama_cpp.llama_token * 1)(token_id)
        batch = llama_cpp.llama_batch_get_one(token_arr, 1, pos, self.seq_id)
        ret = llama_cpp.llama_decode(self.ctx, batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed: {ret}")
        return self._get_last_hidden(), self._get_last_logits()

    def _decode_batch_verify(self, draft_tokens: List[int], pos: int):
        """Batch verify: 1 次 forward 验证所有 draft tokens."""
        n = len(draft_tokens)
        batch = llama_cpp.llama_batch_init(n, 0, 1)
        try:
            for i, tok in enumerate(draft_tokens):
                batch.token[i] = tok
                batch.pos[i] = pos + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = self.seq_id
                batch.logits[i] = 1
            batch.n_tokens = n
            ret = llama_cpp.llama_decode(self.ctx, batch)
            if ret != 0:
                raise RuntimeError(f"batch decode failed: {ret}")
            hiddens = [self._get_hidden_ith(i) for i in range(n)]
            logits_list = [self._get_logits_ith(i) for i in range(n)]
        finally:
            llama_cpp.llama_batch_free(batch)
        return hiddens, logits_list

    def _kv_seq_rm(self, p0: int, p1: int = -1):
        """删除 KV cache [p0, p1)."""
        if hasattr(llama_cpp, 'llama_memory_seq_rm'):
            mem = llama_cpp.llama_get_memory(self.ctx)
            llama_cpp.llama_memory_seq_rm(mem, self.seq_id, p0, p1)
        elif hasattr(llama_cpp, 'llama_kv_cache_seq_rm'):
            llama_cpp.llama_kv_cache_seq_rm(self.ctx, self.seq_id, p0, p1)
            if hasattr(llama_cpp, 'llama_kv_cache_update'):
                llama_cpp.llama_kv_cache_update(self.ctx)

    def reset(self, prompt: str) -> Tuple[np.ndarray, int, np.ndarray]:
        """Prefill prompt, 返回 (last_hidden, last_token, last_logits).

        注意: prefill 后的 embedding 不可靠 (embedding flag 只在最后 token 计算),
        需要像原 verify loop 一样 decode first_token 才能获取正确的 hidden.

        Returns:
            hidden: [n_embd] 最后一个 token 的 hidden state
            token: 最后一个 token ID
            logits: [n_vocab] 最后一个 token 的 logits (用于验证第一个 draft)
        """
        # Clear KV cache
        self._kv_seq_rm(0, -1)

        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        # 只过滤 EOS (151645), 保留 BOS (151643)
        tokens = [t for t in tokens if t != 151645]
        if len(tokens) < 3:
            tokens = tokens + [151644] * 3  # pad
        if len(tokens) > 256:
            tokens = tokens[:256]

        # Prefill (batch decode)
        pos = 0
        batch_size = min(self.llm.n_batch, len(tokens))
        while pos < len(tokens):
            end = min(pos + batch_size, len(tokens))
            bt = tokens[pos:end]
            token_arr = (llama_cpp.llama_token * len(bt))(*bt)
            batch = llama_cpp.llama_batch_get_one(token_arr, len(bt), pos, self.seq_id)
            ret = llama_cpp.llama_decode(self.ctx, batch)
            if ret != 0:
                raise RuntimeError(f"prefill decode failed: {ret}")
            pos = end

        self.n_past = len(tokens)

        # Get first token logits (for draft[0] verification)
        prefill_logits = self._get_last_logits()
        first_token = int(prefill_logits.argmax())

        # Decode first_token to get reliable hidden state
        # (prefill 的 embedding 不可靠, 需要单 token decode)
        hidden, logits = self._decode_single(first_token, self.n_past)
        self.n_past += 1

        return hidden, first_token, logits

    def verify_draft_chain(
        self, draft_tokens: List[int], current_logits: np.ndarray
    ) -> Tuple[int, List[np.ndarray]]:
        """Verify draft chain, 返回 (accept_length, verify_hiddens).

        Args:
            draft_tokens: MTP head 生成的 draft tokens
            current_logits: current token 的 logits (验证 draft[0])

        Returns:
            accept_length: 0 ~ len(draft_tokens)
            verify_hiddens: accept 的 draft tokens 的 hidden states (用于后续训练)
        """
        if not draft_tokens:
            return 0, []

        batch_pos = self.n_past
        try:
            batch_hiddens, batch_logits = self._decode_batch_verify(
                draft_tokens, self.n_past
            )
        except Exception:
            # Fallback: 逐个 verify
            return self._verify_sequential(draft_tokens, current_logits)

        self.n_past += len(draft_tokens)

        # Verify: current_logits 验证 draft[0], batch_logits[i] 验证 draft[i+1]
        verify_logits = current_logits
        accept_length = 0
        accepted_hiddens = []

        for i, draft_token in enumerate(draft_tokens):
            target_token = int(verify_logits.argmax())
            if target_token == draft_token:
                accept_length += 1
                accepted_hiddens.append(batch_hiddens[i])
                if i < len(draft_tokens) - 1:
                    verify_logits = batch_logits[i]
            else:
                # Reject: KV cache 回退
                rm_start = batch_pos + i
                self._kv_seq_rm(rm_start, -1)
                self.n_past = rm_start
                break

        return accept_length, accepted_hiddens

    def _verify_sequential(self, draft_tokens: List[int], current_logits: np.ndarray):
        """逐个 verify (fallback)."""
        verify_logits = current_logits
        accept_length = 0
        accepted_hiddens = []
        for i, draft_token in enumerate(draft_tokens):
            target = int(verify_logits.argmax())
            if target == draft_token:
                accept_length += 1
                hidden, verify_logits = self._decode_single(draft_token, self.n_past)
                self.n_past += 1
                accepted_hiddens.append(hidden)
            else:
                break
        return accept_length, accepted_hiddens


# ============================================================
# MTP Head with temperature sampling + logprob tracking
# ============================================================

class MTPHeadStochastic(nn.Module):
    """MTP head wrapper 支持 temperature sampling 和 logprob 记录.

    核心方法:
      - draft_chain_sample: 采样 draft chain, 记录 (hidden, token, logprob)
      - compute_logprob: 给定 draft chain, 计算 logprob (用于 importance ratio)
    """

    def __init__(self, mtp: MTPHead, embed_weight: torch.Tensor):
        super().__init__()
        self.mtp = mtp
        self.register_buffer("embed_weight", embed_weight)

    def draft_chain_sample(
        self,
        hidden: torch.Tensor,  # [hidden_size]
        token_id: int,
        num_draft: int,
        temperature: float,
        device: torch.device,
    ) -> dict:
        """采样 draft chain, 记录所有中间状态.

        Returns:
            {
                "draft_tokens": [num_draft],
                "logprobs": [num_draft],  # 采样 token 的 logprob
                "hiddens": [num_draft+1, hidden_size],  # h0=base, h1..hN=MTP输出
                "input_tokens": [num_draft],  # 每步输入的 token
            }
        """
        draft_tokens = []
        logprobs = []
        input_tokens = []
        hiddens = [hidden.detach()]

        current_hidden = hidden.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H]
        current_token = token_id

        with torch.no_grad():
            for i in range(num_draft):
                token_embed = self.embed_weight[current_token].unsqueeze(0).unsqueeze(0).to(device)

                # MTP forward
                x = torch.cat([current_hidden, token_embed], dim=-1)
                x = self.mtp.proj(x)
                h = x + self.mtp.attn(self.mtp.norm1(x))
                h = h + self.mtp.mlp(self.mtp.norm2(h))
                mtp_hidden = self.mtp.norm_out(h)

                logits = self.mtp.lm_head(mtp_hidden)  # [1, 1, vocab]
                logits = logits[:, 0, :]  # [1, vocab]

                # Temperature sampling
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    sample_token = torch.multinomial(probs, 1).squeeze(-1)  # [1]
                    logprob = torch.log(probs[0, sample_token] + 1e-10)
                else:
                    sample_token = logits.argmax(dim=-1)
                    logprob = torch.log_softmax(logits, dim=-1)[0, sample_token]

                draft_tokens.append(int(sample_token.item()))
                logprobs.append(float(logprob.item()))
                input_tokens.append(current_token)
                hiddens.append(mtp_hidden[0, 0].detach().cpu())

                current_hidden = mtp_hidden
                current_token = int(sample_token.item())

        return {
            "draft_tokens": draft_tokens,
            "logprobs": logprobs,
            "hiddens": torch.stack(hiddens),  # [num_draft+1, H]
            "input_tokens": input_tokens,
        }

    def compute_logprob(
        self,
        hiddens: torch.Tensor,  # [num_draft+1, H]
        input_tokens: List[int],  # [num_draft]
        draft_tokens: List[int],  # [num_draft]
        device: torch.device,
    ) -> torch.Tensor:
        """重新计算给定 draft chain 的 logprob (用于 policy update).

        Returns:
            logprobs: [num_draft] 当前 policy 对采样 token 的 logprob
        """
        logprobs = []
        current_hidden = hiddens[0].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H]

        for i in range(len(draft_tokens)):
            token_embed = self.embed_weight[input_tokens[i]].unsqueeze(0).unsqueeze(0).to(device)

            x = torch.cat([current_hidden, token_embed], dim=-1)
            x = self.mtp.proj(x)
            h = x + self.mtp.attn(self.mtp.norm1(x))
            h = h + self.mtp.mlp(self.mtp.norm2(h))
            mtp_hidden = self.mtp.norm_out(h)

            logits = self.mtp.lm_head(mtp_hidden)[:, 0, :]  # [1, vocab]
            logprob = F.log_softmax(logits, dim=-1)[0, draft_tokens[i]]
            logprobs.append(logprob)

            # Next step uses sampled token (not ground truth)
            current_hidden = mtp_hidden

        return torch.stack(logprobs)  # [num_draft]


# ============================================================
# GRPO Trainer
# ============================================================

class GRPOTrainer:
    """GRPO trainer for MTP head.

    流程:
      1. Sample prompts
      2. For each prompt: rollout G draft chains, record (hidden, tokens, logprob, reward)
      3. Compute group-relative advantage
      4. PPO clip update + KL penalty
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load embed + lm_head
        print(f"[grpo] Loading embed_head from {config.embed_head_path}...", flush=True)
        eh = torch.load(config.embed_head_path, map_location="cpu", weights_only=True)
        self.embed_weight = eh["embed_weight"].float().to(self.device)
        self.lm_head_weight = eh["lm_head_weight"].float().to(self.device)

        # Create MTP head
        mtp_config = MTPHeadConfig(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
        )
        self.mtp = MTPHead(mtp_config)
        self.mtp.set_shared_lm_head(self.lm_head_weight)
        self.mtp.to(self.device).to(torch.float32)

        # Load checkpoint
        if config.checkpoint and os.path.exists(config.checkpoint):
            print(f"[grpo] Loading checkpoint: {config.checkpoint}", flush=True)
            ckpt = torch.load(config.checkpoint, weights_only=False, map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt)
            filtered = {k: v for k, v in sd.items() if "lm_head" not in k}
            self.mtp.load_state_dict(filtered, strict=False)
            print(f"[grpo] Loaded {len(filtered)} tensors", flush=True)

        # Stochastic wrapper
        self.policy = MTPHeadStochastic(self.mtp, self.embed_weight).to(self.device)

        # Reference policy (frozen, for KL penalty)
        self.ref_mtp = copy.deepcopy(self.mtp)
        self.ref_mtp.eval()
        for p in self.ref_mtp.parameters():
            p.requires_grad = False
        self.ref_policy = MTPHeadStochastic(self.ref_mtp, self.embed_weight).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.mtp.parameters(), lr=config.lr, betas=(0.9, 0.95), weight_decay=0.01
        )

        # Rollout env
        self.env = MTPRolloutEnv(config.gguf_path)

        # Prompts
        self.prompts = self._generate_prompts()

        # Logging
        os.makedirs(config.output_dir, exist_ok=True)
        self.log = []

    def _generate_prompts(self) -> List[str]:
        """生成训练 prompts."""
        prompts = [
            "Write a Python function to check if a number is prime:",
            "def fibonacci(n):\n    ",
            "class Stack:\n    def __init__(self):\n        ",
            "import numpy as np\n\ndef normalize(x):\n    ",
            "def binary_search(arr, target):\n    ",
            "async def fetch_data(url):\n    ",
            "def merge_sort(arr):\n    ",
            "class TreeNode:\n    def __init__(self, val=0):\n        ",
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ",
            "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\nasync def root():\n    ",
            "def reverse_list(head):\n    ",
            "def is_palindrome(s):\n    ",
            "import torch\n\nclass SimpleModel(nn.Module):\n    def __init__(self):\n        ",
            "def train_model(X, y):\n    from sklearn.ensemble import RandomForestClassifier\n    ",
            "def connect_db(host, port):\n    ",
            "def parse_json(text):\n    ",
        ]
        return prompts * 10  # 160 prompts

    def rollout_one(
        self, prompt: str, group_size: int, num_draft: int, temperature: float,
        max_rounds: int = 10,
    ) -> List[dict]:
        """对一个 prompt rollout G 条 draft chain.

        每条 chain 跑 max_rounds 轮, 每轮 draft + verify.
        每轮独立作为一个 sample, reward = 该轮 accept_length / num_draft,
        这样 GRPO 能利用全部 10 轮信号 (而非只用最后一轮).

        同 prompt 的所有 (chain × round) sample 共享同一 group baseline,
        用于计算 group-relative advantage.

        Returns:
            samples: [{
                "hiddens": [num_draft+1, H],
                "input_tokens": [num_draft],
                "draft_tokens": [num_draft],
                "old_logprobs": [num_draft],
                "accept_length": int,   # 该轮 accept 数 (0~num_draft)
                "reward": float,        # 该轮 reward
                "chain_id": int,        # chain 标识 (同 chain 共享 KV cache)
            }, ...]
        """
        samples = []

        for g in range(group_size):
            # 每条 chain 重新 reset
            hidden, current_token, current_logits = self.env.reset(prompt)

            for round_idx in range(max_rounds):
                # MTP draft chain sampling
                result = self.policy.draft_chain_sample(
                    torch.from_numpy(hidden).float(),
                    current_token,
                    num_draft,
                    temperature,
                    self.device,
                )

                # Verify
                accept_length, accepted_hiddens = self.env.verify_draft_chain(
                    result["draft_tokens"], current_logits
                )

                # 每轮 reward = accept_length / num_draft + bonus
                reward = accept_length / num_draft
                if accept_length == num_draft:
                    reward += 0.2  # 全 accept bonus, 鼓励长链
                elif accept_length == 0:
                    reward -= 0.1  # 全 reject 轻微惩罚

                samples.append({
                    "hiddens": result["hiddens"],  # [num_draft+1, H]
                    "input_tokens": result["input_tokens"],
                    "draft_tokens": result["draft_tokens"],
                    "old_logprobs": result["logprobs"],
                    "accept_length": accept_length,
                    "reward": reward,
                    "chain_id": g,
                })

                # 更新 current state for next round
                if accept_length == num_draft:
                    # All accepted: decode bonus token (last accepted) 获取 hidden
                    if accepted_hiddens:
                        current_token = result["draft_tokens"][-1]
                        h, current_logits = self.env._decode_single(current_token, self.env.n_past)
                        self.env.n_past += 1
                        hidden = h
                    else:
                        break
                elif accept_length > 0:
                    # Partial accept: 从最后一个 accepted token 继续
                    current_token = result["draft_tokens"][accept_length - 1]
                    h, current_logits = self.env._decode_single(current_token, self.env.n_past)
                    self.env.n_past += 1
                    hidden = h
                else:
                    # All rejected: decode target (current_logits.argmax())
                    target_token = int(current_logits.argmax())
                    h, current_logits = self.env._decode_single(target_token, self.env.n_past)
                    self.env.n_past += 1
                    hidden = h
                    current_token = target_token

        return samples

    def compute_grpo_loss(
        self, samples: List[dict]
    ) -> Tuple[torch.Tensor, dict]:
        """计算 GRPO loss.

        GRPO:
          A_i = (r_i - mean(r)) / (std(r) + eps)
          ratio = exp(new_logprob - old_logprob)
          loss = -mean(min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)) + β * KL
        """
        cfg = self.config
        rewards = torch.tensor([s["reward"] for s in samples], dtype=torch.float32, device=self.device)
        group_mean = rewards.mean()
        group_std = rewards.std() + 1e-8
        advantages = (rewards - group_mean) / group_std  # [G]

        total_loss = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)
        total_clip_frac = 0.0
        n_valid = 0

        for i, s in enumerate(samples):
            # 全 group reward 相同 (无 advantage 信号) 才跳过
            if group_std.item() < 1e-6:
                continue

            # New policy logprob
            new_logprobs = self.policy.compute_logprob(
                s["hiddens"].to(self.device),
                s["input_tokens"],
                s["draft_tokens"],
                self.device,
            )  # [num_draft]

            # Old policy logprob
            old_logprobs = torch.tensor(
                s["old_logprobs"], device=self.device, dtype=torch.float32
            )

            # Importance ratio
            ratio = torch.exp(new_logprobs - old_logprobs)

            # Advantage (broadcast to num_draft)
            adv = advantages[i].expand_as(ratio)

            # PPO clipped objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL penalty (new policy vs reference policy)
            with torch.no_grad():
                ref_logprobs = self.ref_policy.compute_logprob(
                    s["hiddens"].to(self.device),
                    s["input_tokens"],
                    s["draft_tokens"],
                    self.device,
                )
            kl = (new_logprobs - ref_logprobs).mean()

            loss = policy_loss + cfg.kl_penalty * kl

            total_loss = total_loss + loss
            total_kl = total_kl + kl
            total_clip_frac += float(((ratio - 1).abs() > cfg.clip_ratio).float().mean().item())
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "n_valid": 0
            }

        total_loss = total_loss / n_valid
        total_kl = total_kl / n_valid
        total_clip_frac /= n_valid

        return total_loss, {
            "loss": float(total_loss.item()),
            "kl": float(total_kl.item()),
            "clip_frac": total_clip_frac,
            "n_valid": n_valid,
        }

    def train(self):
        """GRPO 训练主循环."""
        cfg = self.config
        print(f"\n[grpo] Starting GRPO training", flush=True)
        print(f"  num_draft={cfg.num_draft}, group_size={cfg.group_size}", flush=True)
        print(f"  num_iterations={cfg.num_iterations}, lr={cfg.lr}", flush=True)
        print(f"  clip_ratio={cfg.clip_ratio}, kl_penalty={cfg.kl_penalty}", flush=True)

        for iteration in range(cfg.num_iterations):
            t0 = time.time()
            self.mtp.train()

            # Anneal temperature
            temp = max(
                cfg.min_temperature,
                cfg.temperature * (1 - iteration / cfg.num_iterations),
            )

            # Sample prompts
            batch_prompts = random.sample(
                self.prompts, min(cfg.num_prompts_per_iter, len(self.prompts))
            )

            # Rollout
            all_samples = []
            t_rollout = time.time()
            for prompt in batch_prompts:
                samples = self.rollout_one(
                    prompt, cfg.group_size, cfg.num_draft, temp
                )
                all_samples.extend(samples)
            rollout_time = time.time() - t_rollout

            # Compute GRPO loss
            t_update = time.time()
            loss, metrics = self.compute_grpo_loss(all_samples)

            # Backward + step
            self.optimizer.zero_grad()
            if metrics["n_valid"] > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mtp.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
            update_time = time.time() - t_update

            # Stats
            # 每个 sample 是单轮 draft+verify, accept_length 已是单轮值 (0~num_draft)
            rewards = [s["reward"] for s in all_samples]
            accept_lengths = [s["accept_length"] for s in all_samples]
            mean_reward = np.mean(rewards) if rewards else 0
            mean_accept = np.mean(accept_lengths) if accept_lengths else 0
            accept_rate = mean_accept / cfg.num_draft  # 真实 accept rate (0~100%)

            total_time = time.time() - t0

            log_entry = {
                "iter": iteration,
                "temp": temp,
                "mean_reward": mean_reward,
                "mean_accept": mean_accept,
                "accept_rate": accept_rate,
                "loss": metrics["loss"],
                "kl": metrics["kl"],
                "clip_frac": metrics["clip_frac"],
                "n_valid": metrics["n_valid"],
                "rollout_time": rollout_time,
                "update_time": update_time,
                "total_time": total_time,
            }
            self.log.append(log_entry)

            if iteration % cfg.log_every == 0:
                print(
                    f"  [I{iteration:3d}] temp={temp:.2f} accept={mean_accept:.2f}/{cfg.num_draft} "
                    f"({accept_rate:.1%}) reward={mean_reward:.3f} loss={metrics['loss']:.4f} "
                    f"kl={metrics['kl']:.4f} clip={metrics['clip_frac']:.2f} "
                    f"valid={metrics['n_valid']}/{len(all_samples)} "
                    f"time={total_time:.1f}s",
                    flush=True,
                )

            # Save checkpoint
            if (iteration + 1) % cfg.save_every == 0 or iteration == cfg.num_iterations - 1:
                ckpt_path = os.path.join(
                    cfg.output_dir,
                    f"mtp_head_slime_rl_iter{iteration+1}.pt",
                )
                torch.save(
                    {
                        "model_state_dict": self.mtp.state_dict(),
                        "iteration": iteration + 1,
                        "config": vars(cfg),
                    },
                    ckpt_path,
                )
                print(f"  [save] {ckpt_path}", flush=True)

                # Save log
                log_path = os.path.join(cfg.output_dir, "slime_rl_log.json")
                with open(log_path, "w") as f:
                    json.dump(self.log, f, indent=2)

        print(f"\n[grpo] Training complete! {cfg.num_iterations} iterations", flush=True)


# ============================================================
# Main
# ============================================================

import ctypes  # needed by MTPRolloutEnv

def main():
    parser = argparse.ArgumentParser(
        description="Slime RL GRPO for MTP Head",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gguf", default="/Users/alexchuang/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf")
    parser.add_argument("--checkpoint", default="mtp_output/qwen25_0_5b_v4_scheduled/mtp_head_qwen25-0.5b_decode.pt")
    parser.add_argument("--embed-head", default="mtp_train_data/qwen25_0_5b_v4/embed_head.pt")
    parser.add_argument("--output-dir", default="mtp_output/qwen25_0_5b_slime_rl")
    parser.add_argument("--num-draft", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--kl-penalty", type=float, default=0.2)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    config = GRPOConfig(
        gguf_path=args.gguf,
        checkpoint=args.checkpoint,
        embed_head_path=args.embed_head,
        output_dir=args.output_dir,
        num_draft=args.num_draft,
        group_size=args.group_size,
        num_prompts_per_iter=args.num_prompts,
        num_iterations=args.iterations,
        temperature=args.temperature,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        kl_penalty=args.kl_penalty,
        device=args.device,
    )

    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
