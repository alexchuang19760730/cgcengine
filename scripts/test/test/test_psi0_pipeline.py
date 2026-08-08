import os

print("=== [Psi-Zero / Qwen] 端雲一體性能對比測試 ===")
print("說明: 由於 Psi0-8B 權重需要申請權限 (HTTP 401)，我們以架構相近的模型進行基準與編譯比較。")

local_model_path = "./qwen_vllm_ready"

print("\n--- 1. 基線測試 (Baseline): 原生 MLX 推理 ---")
cmd_baseline = f"PYTHONPATH=./MagiCompiler-main python3 MagiCompiler-main/cgc_engine/agent/cli.py pipeline --backend mlx --task-type inference --exec-mode native --model {local_model_path}"
print(f"Running: {cmd_baseline}")
os.system(cmd_baseline)

print("\n--- 2. 端側編譯與微調: mlx-tune ---")
cmd_edge_tune = f"PYTHONPATH=./MagiCompiler-main python3 MagiCompiler-main/cgc_engine/agent/cli.py pipeline --backend mlx-tune --task-type tune --exec-mode compile --model {local_model_path}"
print(f"Running: {cmd_edge_tune}")
os.system(cmd_edge_tune)

print("\n--- 3. 雲側編譯與訓練: megatrain ---")
cmd_cloud_train = f"PYTHONPATH=./MagiCompiler-main python3 MagiCompiler-main/cgc_engine/agent/cli.py pipeline --backend megatrain --task-type train --exec-mode compile --model {local_model_path}"
print(f"Running: {cmd_cloud_train}")
os.system(cmd_cloud_train)
