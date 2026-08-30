#!/usr/bin/env python3
"""
DSH Trajectory → agent_harness SFT 数据集转换器

读取 DSH 的 session log (JSONL)，提取成功轨迹，
转换为 agent_harness 的 train.jsonl / valid.jsonl 格式。

用法:
    python dsh_to_sft.py --input <dsh_session_dir> --output <sft_output_dir>
    python dsh_to_sft.py --input results/dsh_20260830_120000 --output ../sft_data
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any


def parse_dsh_session(session_dir: str) -> List[Dict[str, Any]]:
    """解析 DSH session 目录，提取对话轨迹"""
    session_path = Path(session_dir)
    messages = []
    
    # DSH session log 是 append-only JSONL
    for jsonl_file in session_path.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    messages.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return messages


def extract_trajectory(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """从 DSH session messages 提取 user/assistant 对话"""
    trajectory = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role in ("user", "assistant") and content:
            # 清理内容
            if isinstance(content, list):
                # 多模态消息，提取文本
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "\n".join(text_parts)
            
            if content.strip():
                trajectory.append({
                    "role": role,
                    "content": content.strip()
                })
    
    return trajectory


def is_successful_trajectory(trajectory: List[Dict[str, str]]) -> bool:
    """判断轨迹是否成功（简化判断）"""
    if len(trajectory) < 2:
        return False
    
    # 最后一条 assistant 消息包含成功指标
    last_assistant = None
    for msg in reversed(trajectory):
        if msg["role"] == "assistant":
            last_assistant = msg["content"]
            break
    
    if not last_assistant:
        return False
    
    # 成功指标：包含 "rc=0"、"success"、"pass" 等
    success_indicators = ["rc=0", "success", "pass", "✅", "completed"]
    failure_indicators = ["rc=1", "error", "fail", "❌", "timeout", "no_command"]
    
    has_success = any(ind in last_assistant.lower() for ind in success_indicators)
    has_failure = any(ind in last_assistant.lower() for ind in failure_indicators)
    
    # 有成功指标且没有失败指标
    return has_success and not has_failure


def convert_to_sft_format(trajectory: List[Dict[str, str]]) -> Dict[str, Any]:
    """转换为 SFT 训练格式"""
    # 构建 system prompt
    system_prompt = """You are a coding agent. You have access to a terminal and file editor.
Execute the given task by running commands and editing files.
Always check your work and verify the results."""
    
    # 构建 messages
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(trajectory)
    
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="DSH Trajectory → SFT 数据集转换")
    parser.add_argument("--input", "-i", required=True, help="DSH session 目录")
    parser.add_argument("--output", "-o", required=True, help="SFT 输出目录")
    parser.add_argument("--split", default="0.9", help="训练/验证集分割比 (default: 0.9)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有 session
    all_sft_data = []
    session_dirs = list(input_path.glob("*"))
    
    for session_dir in session_dirs:
        if not session_dir.is_dir():
            continue
        
        print(f"处理 session: {session_dir.name}")
        messages = parse_dsh_session(str(session_dir))
        trajectory = extract_trajectory(messages)
        
        if is_successful_trajectory(trajectory):
            sft_entry = convert_to_sft_format(trajectory)
            all_sft_data.append(sft_entry)
            print(f"  ✅ 成功轨迹, {len(trajectory)} 条消息")
        else:
            print(f"  ⏭️ 跳过 (失败/不完整)")
    
    if not all_sft_data:
        print("警告: 没有找到成功轨迹", file=sys.stderr)
        sys.exit(1)
    
    # 分割训练/验证集
    split_ratio = float(args.split)
    split_idx = int(len(all_sft_data) * split_ratio)
    train_data = all_sft_data[:split_idx]
    valid_data = all_sft_data[split_idx:]
    
    # 写入文件
    train_file = output_path / "train.jsonl"
    valid_file = output_path / "valid.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    with open(valid_file, "w", encoding="utf-8") as f:
        for entry in valid_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n=== 转换完成 ===")
    print(f"总成功轨迹: {len(all_sft_data)}")
    print(f"训练集: {len(train_data)} 条 → {train_file}")
    print(f"验证集: {len(valid_data)} 条 → {valid_file}")


if __name__ == "__main__":
    main()
