import os
import json

print("=== [Psi-Zero] 端雲一體性能對比測試 ===")
model_path = "./models/Psi0"

print("\n--- 1. 基線測試 (Baseline): 原生 MLX 推理 ---")
cmd_baseline = f"PYTHONPATH=./MagiCompiler-main python3 MagiCompiler-main/cgc_engine/agent/cli.py pipeline --backend mlx --task-type inference --exec-mode native --model {model_path}"
print(f"Running: {cmd_baseline}")
os.system(cmd_baseline)

print("\n--- 2. 端側編譯與微調: mlx-tune ---")
cmd_edge_tune = f"PYTHONPATH=./MagiCompiler-main python3 MagiCompiler-main/cgc_engine/agent/cli.py pipeline --backend mlx-tune --task-type tune --exec-mode compile --model {model_path}"
print(f"Running: {cmd_edge_tune}")
os.system(cmd_edge_tune)

print("\n--- 3. 雲側編譯與訓練: megatrain ---")
cmd_cloud_train = f"PYTHONPATH=./MagiCompiler-main python3 MagiCompiler-main/cgc_engine/agent/cli.py pipeline --backend megatrain --task-type train --exec-mode compile --model {model_path}"
print(f"Running: {cmd_cloud_train}")
os.system(cmd_cloud_train)

# 讀取並打印結果
try:
    with open('/tmp/cgc_engine_pipeline_report.json') as f:
        report = json.load(f)
        print("\n=== 測試報告 ===")
        print(f"整體狀態: {'PASS' if report.get('ok') else 'FAIL'}")
        
        speedup = report.get('speedup_ratio', {})
        if speedup:
            print(f"編譯加速比: {speedup}")
            
        print("詳細步驟結果:")
        for step, data in report.get('steps', {}).items():
            print(f"  - {step}: {data.get('status', 'UNKNOWN')}")
            if 'note' in data:
                print(f"    Note: {data['note']}")
except Exception as e:
    print(f"讀取報告失敗: {e}")
