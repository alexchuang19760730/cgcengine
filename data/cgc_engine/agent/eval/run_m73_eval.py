import argparse
import json
from pathlib import Path

import yaml


def evaluate_m73_gate(cgc_report_path: str, yaml_config_path: str, out_dir: str) -> bool:
    print(f"[*] 載入 M7.3 評測配置: {yaml_config_path}")
    with open(yaml_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[*] 讀取 CGC Pipeline 報告: {cgc_report_path}")
    with open(cgc_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    m73_data = (report.get("gate_result") or {}).get("m73", {})
    results = {}

    print("\n" + "=" * 40)
    print(f"  {config['name']} v{config['version']}")
    print("=" * 40)

    if not m73_data:
        final_pass = False
        reason = "missing_gate_result_m73"
        print(f"\n▶ 評測項目: M7.3 Gate")
        print(f"  ❌ FAIL | {reason}")
        results["m73"] = "FAIL"
    else:
        final_pass = True
        for metric_group in config["metrics"]:
            group_name = metric_group["name"]
            print(f"\n▶ 評測項目: {metric_group['description']}")

            group_pass = True
            for rule in metric_group["rules"]:
                metric_id = rule["metric"]
                op = rule["operator"]
                threshold = float(rule["threshold"])
                actual_value = None

                if group_name == "cloud_training_psi0":
                    dt = m73_data.get("cloud_training_psi0", {})
                    if metric_id == "compile_success_rate":
                        actual_value = dt.get("compile_success_rate", 0.0)
                    elif metric_id == "cache_hit_rate":
                        actual_value = dt.get("cache_hit_rate", 0.0)

                elif group_name == "edge_inference_bridge":
                    br = m73_data.get("edge_inference_bridge", {})
                    if metric_id == "bridge_export_success":
                        actual_value = br.get("bridge_export_success", 0.0)
                    elif metric_id == "edge_latency_ms":
                        actual_value = br.get("edge_latency_ms", 999.0)

                elif group_name == "state_compression":
                    sc = m73_data.get("state_compression", {})
                    if metric_id == "compression_ratio":
                        actual_value = sc.get("compression_ratio", 1.0)
                    elif metric_id == "restore_consistency":
                        actual_value = sc.get("restore_consistency", 0.0)

                elif group_name == "industrial_audit":
                    au = m73_data.get("industrial_audit", {})
                    if not au:
                        a2 = m73_data.get("audit", {})
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

                passed = False
                if op == ">=":
                    passed = actual_value >= threshold
                elif op == "<=":
                    passed = actual_value <= threshold
                elif op == "==":
                    passed = actual_value == threshold

                status_str = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {status_str} | {metric_id} (閾值: {op} {threshold}) -> 實際: {actual_value}")
                if not passed:
                    group_pass = False

            results[group_name] = "PASS" if group_pass else "FAIL"
            if not group_pass:
                final_pass = False

    print("\n" + "=" * 40)
    print(f"最終 M7.3 Gate 驗收結果: {'✅ 通過' if final_pass else '❌ 未通過'}")
    print("=" * 40)

    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir_p / str(config["output"]["report_file"]))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {"name": config["name"], "status": "PASS" if final_pass else "FAIL", "metrics": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[*] 已生成評測報告: {out_file}")
    return bool(final_pass)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cgc-report", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    ok = evaluate_m73_gate(args.cgc_report, args.config, args.out_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

