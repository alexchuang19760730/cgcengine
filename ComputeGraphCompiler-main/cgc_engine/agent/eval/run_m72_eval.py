import json
import yaml
import sys
import os
from pathlib import Path

def evaluate_m72_gate(cgc_report_path: str, yaml_config_path: str, out_dir: str):
    print(f"[*] 載入 M7.2 評測配置: {yaml_config_path}")
    with open(yaml_config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"[*] 讀取 CGC Pipeline 報告: {cgc_report_path}")
    with open(cgc_report_path, 'r') as f:
        report = json.load(f)

    m7_data = (report.get("gate_result") or {}).get("m7", {})
    if not m7_data:
        print("[!] 錯誤: 報告中找不到 M7 Gate 數據。")
        return False

    final_pass = True
    results = {}

    print("\n" + "="*40)
    print(f"  {config['name']} v{config['version']}")
    print("="*40)

    for metric_group in config['metrics']:
        group_name = metric_group['name']
        print(f"\n▶ 評測項目: {metric_group['description']}")
        
        group_pass = True
        
        # Mapping rules to real data extracted from M7
        for rule in metric_group['rules']:
            metric_id = rule['metric']
            op = rule['operator']
            threshold = float(rule['threshold'])
            actual_value = None

            # 1. Dynamic Trace
            if group_name == "dynamic_trace_l1":
                dt = m7_data.get("dynamic_trace_l1", {})
                if not dt:
                    dt = m7_data.get("dynamic_trace", {})

                if metric_id in ("compile_success_rate", "cache_hit_rate", "correctness_consistency"):
                    if isinstance(dt, dict) and any(k in dt for k in ("compile_success_rate", "cache_hit_rate", "correctness_consistency")):
                        if metric_id == "compile_success_rate":
                            actual_value = dt.get("compile_success_rate", 0.0)
                        elif metric_id == "cache_hit_rate":
                            actual_value = dt.get("cache_hit_rate", 0.0)
                        elif metric_id == "correctness_consistency":
                            actual_value = dt.get("correctness_consistency", 0.0)
                    else:
                        compile_variants = dt.get("compile_variants", []) if isinstance(dt, dict) else []
                        correctness = dt.get("correctness", []) if isinstance(dt, dict) else []

                        if metric_id == "compile_success_rate":
                            if isinstance(compile_variants, list) and len(compile_variants) > 0:
                                ok = 0
                                for x in compile_variants:
                                    if isinstance(x, dict) and str(x.get("status") or "") == "PASS":
                                        ok += 1
                                actual_value = float(ok / len(compile_variants))
                            else:
                                actual_value = 0.0
                        elif metric_id == "cache_hit_rate":
                            if isinstance(compile_variants, list) and len(compile_variants) > 0:
                                hit = 0
                                for x in compile_variants:
                                    if isinstance(x, dict) and bool(x.get("cache_hit")):
                                        hit += 1
                                actual_value = float(hit / len(compile_variants))
                            else:
                                actual_value = 0.0
                        elif metric_id == "correctness_consistency":
                            if isinstance(correctness, list) and len(correctness) > 0:
                                ok = 0
                                for x in correctness:
                                    if isinstance(x, dict) and bool(x.get("repeat_consistent")):
                                        ok += 1
                                actual_value = float(ok / len(correctness))
                            else:
                                actual_value = 0.0

            # 2. State Compression
            elif group_name == "state_compression":
                sc = m7_data.get("state_compression", {})
                if not sc:
                    sc = m7_data.get("state_compression_summary", {})
                if metric_id == "compression_ratio":
                    actual_value = sc.get("compression_ratio", 1.0)
                elif metric_id == "restore_consistency":
                    actual_value = sc.get("restore_consistency", 0.0)
                elif metric_id == "dedup_expansion_ratio":
                    actual_value = sc.get("dedup_expansion_ratio", 999.0)

            # 3. Soft-RT Replay
            elif group_name == "soft_rt_replay":
                rp = m7_data.get("soft_rt_replay", {})
                if not rp:
                    rp = m7_data.get("replay", {})
                if metric_id == "deadline_ms":
                    actual_value = rp.get("deadline_ms", 999.0)
                elif metric_id == "p99_latency_ms":
                    if "p99_latency_ms" in rp:
                        actual_value = rp.get("p99_latency_ms", 999.0)
                    else:
                        actual_value = ((rp.get("latency_ms") or {}).get("p99")) if isinstance(rp.get("latency_ms"), dict) else 999.0
                elif metric_id == "miss_rate":
                    actual_value = rp.get("miss_rate", 1.0)

            # 4. Industrial Audit
            elif group_name == "industrial_audit":
                au = m7_data.get("industrial_audit", {})
                if not au:
                    a2 = m7_data.get("audit", {})
                    au = {
                        "event_integrity": 1.0 if str(a2.get("status") or "") == "PASS" else 0.0,
                        "hash_chain_valid": 1.0 if bool(a2.get("verify_ok")) else 0.0,
                    }
                if metric_id == "event_integrity":
                    actual_value = au.get("event_integrity", 0.0)
                elif metric_id == "hash_chain_valid":
                    actual_value = au.get("hash_chain_valid", 0.0)

            if actual_value is None:
                print(f"  [?] {metric_id}: 數據缺失")
                group_pass = False
                continue

            # Evaluate
            passed = False
            if op == ">=": passed = actual_value >= threshold
            elif op == "<=": passed = actual_value <= threshold
            elif op == "==": passed = actual_value == threshold

            status_str = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status_str} | {metric_id} (閾值: {op} {threshold}) -> 實際: {actual_value}")
            if not passed:
                group_pass = False

        results[group_name] = "PASS" if group_pass else "FAIL"
        if not group_pass:
            final_pass = False

    print("\n" + "="*40)
    print(f"最終 M7.2 Gate 驗收結果: {'✅ 通過' if final_pass else '❌ 失敗'}")
    print("="*40)
    
    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir_p / str(config['output']['report_file']))

    with open(out_file, 'w', encoding="utf-8") as f:
        json.dump({
            "name": config['name'],
            "status": "PASS" if final_pass else "FAIL",
            "metrics": results
        }, f, ensure_ascii=False, indent=2)
    print(f"[*] 已生成評測報告: {out_file}")
    
    return final_pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cgc-report", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    evaluate_m72_gate(args.cgc_report, args.config, args.out_dir)
