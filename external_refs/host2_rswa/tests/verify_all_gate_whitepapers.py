#!/usr/bin/env python3
"""
验证所有 CGC Gate 白皮书的能力状态
检查 gate_map.json 与白皮书的一致性
生成完整的验证报告
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

def load_gate_map(gate_dir: Path) -> Dict[str, Any]:
    """加载 gate_map.json 文件"""
    gate_map_path = gate_dir / f"{gate_dir.name}_gate_map.json"
    if gate_map_path.exists():
        with open(gate_map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # 尝试其他命名格式
    alt_paths = list(gate_dir.glob("*gate_map*.json"))
    if alt_paths:
        with open(alt_paths[0], "r", encoding="utf-8") as f:
            return json.load(f)
    
    return None

def load_whitepaper(gate_dir: Path) -> str:
    """加载白皮书内容"""
    wp_paths = list(gate_dir.glob("*Technical_Whitepaper*.md"))
    if wp_paths:
        with open(wp_paths[0], "r", encoding="utf-8") as f:
            return f.read()
    return ""

def parse_whitepaper_capabilities(wp_content: str) -> List[Dict[str, str]]:
    """从白皮书表格中解析能力列表"""
    capabilities = []
    
    # 匹配能力表格（Markdown 表格格式）
    table_pattern = r'\|\s*capability_id\s*\|\s*名称\s*\|\s*当前状态\s*\|\s*gate_pass_claim\s*\|'
    if re.search(table_pattern, wp_content):
        lines = wp_content.split('\n')
        in_table = False
        header_found = False
        
        for line in lines:
            if 'capability_id' in line and '当前状态' in line and '名称' in line:
                header_found = True
                in_table = True
                continue
            
            if in_table and line.startswith('|'):
                if line.count('|') >= 4 and header_found:
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 4:
                        cap_id = parts[0].strip('`')
                        name = parts[1]
                        status = parts[2].strip('`')
                        claim = parts[3].strip('`') if len(parts) > 3 else "unknown"
                        capabilities.append({
                            "capability_id": cap_id,
                            "name": name,
                            "status": status,
                            "gate_pass_claim": claim
                        })
            elif in_table and line.startswith('---'):
                continue
            elif in_table and not line.startswith('|'):
                break
    
    return capabilities

def validate_gate(gate_dir: Path) -> Dict[str, Any]:
    """验证单个 Gate 的白皮书和 gate_map"""
    gate_name = gate_dir.name
    print(f"🔍 正在验证: {gate_name}")
    
    gate_map = load_gate_map(gate_dir)
    wp_content = load_whitepaper(gate_dir)
    wp_caps = parse_whitepaper_capabilities(wp_content)
    
    result = {
        "gate_name": gate_name,
        "gate_dir": str(gate_dir),
        "gate_map_exists": gate_map is not None,
        "whitepaper_exists": bool(wp_content),
        "capabilities": [],
        "issues": [],
        "summary": {
            "total": 0,
            "done": 0,
            "proof": 0,
            "target": 0,
            "integrated": 0,
            "unknown": 0
        }
    }
    
    if gate_map:
        gm_caps = gate_map.get("capabilities", [])
        result["gate_map_capabilities"] = len(gm_caps)
        
        # 检查每个能力
        for cap in gm_caps:
            cap_id = cap.get("capability_id")
            name = cap.get("name")
            status = cap.get("status", "unknown").lower()
            claim = cap.get("gate_pass_claim", "unknown")
            
            # 检查白皮书是否有对应能力
            wp_match = next((c for c in wp_caps if c["capability_id"] == cap_id), None)
            
            # 检查状态一致性
            status_match = True
            if wp_match:
                wp_status = wp_match["status"].lower()
                if status != wp_status:
                    result["issues"].append({
                        "type": "status_mismatch",
                        "capability_id": cap_id,
                        "gate_map_status": status,
                        "whitepaper_status": wp_status
                    })
                    status_match = False
            
            # 检查 claim 是否为 allowed
            if status == "done" and claim != "allowed":
                result["issues"].append({
                    "type": "claim_not_allowed",
                    "capability_id": cap_id,
                    "status": status,
                    "gate_pass_claim": claim
                })
            
            # 检查是否有验证证据
            if status == "done" and "validation_evidence" not in cap:
                result["issues"].append({
                    "type": "missing_evidence",
                    "capability_id": cap_id,
                    "name": name
                })
            
            result["capabilities"].append({
                "capability_id": cap_id,
                "name": name,
                "status": status,
                "gate_pass_claim": claim,
                "in_whitepaper": wp_match is not None,
                "status_consistent": status_match,
                "has_evidence": "validation_evidence" in cap
            })
            
            # 更新统计
            result["summary"]["total"] += 1
            if status in result["summary"]:
                result["summary"][status] += 1
            else:
                result["summary"]["unknown"] += 1
    
    else:
        result["issues"].append({
            "type": "missing_gate_map",
            "message": "gate_map.json 文件不存在"
        })
    
    # 检查白皮书能力数量
    result["whitepaper_capabilities"] = len(wp_caps)
    if gate_map and len(gm_caps) != len(wp_caps):
        result["issues"].append({
            "type": "cap_count_mismatch",
            "gate_map_count": len(gm_caps),
            "whitepaper_count": len(wp_caps)
        })
    
    return result

def generate_report(results: List[Dict[str, Any]]) -> str:
    """生成格式化的验证报告"""
    report = []
    report.append("=" * 80)
    report.append("CGC Gate 白皮书能力验证报告")
    report.append("=" * 80)
    report.append("")
    
    total_caps = 0
    total_done = 0
    total_proof = 0
    total_target = 0
    total_issues = 0
    
    for result in results:
        report.append(f"📋 {result['gate_name']}")
        report.append(f"   目录: {result['gate_dir']}")
        report.append(f"   gate_map: {'✅ 存在' if result['gate_map_exists'] else '❌ 缺失'}")
        report.append(f"   白皮书: {'✅ 存在' if result['whitepaper_exists'] else '❌ 缺失'}")
        report.append("")
        
        if result["capabilities"]:
            report.append("   能力状态:")
            for cap in result["capabilities"]:
                status_icon = {
                    "done": "✅",
                    "proof": "📋",
                    "target": "🎯",
                    "integrated": "🔄"
                }.get(cap["status"], "❓")
                
                evidence_icon = " ✨" if cap["has_evidence"] else ""
                report.append(f"     {status_icon} {cap['name']} {evidence_icon}")
                report.append(f"        ID: {cap['capability_id']}")
                report.append(f"        状态: {cap['status']}")
                report.append(f"        Claim: {cap['gate_pass_claim']}")
                report.append(f"        在白皮书: {'✅' if cap['in_whitepaper'] else '❌'}")
                if not cap["status_consistent"]:
                    report.append(f"        ⚠️ 状态不一致")
        
        # 汇总统计
        summary = result["summary"]
        report.append(f"   统计: 总计={summary['total']} done={summary['done']} proof={summary['proof']} target={summary['target']}")
        
        # 问题列表
        if result["issues