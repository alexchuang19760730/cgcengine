# Scripts 目录说明

本目录包含所有项目相关的 Shell 脚本，按功能分类存放。

## 目录结构

```
scripts/
├── install/      # 安装相关脚本
├── check/        # 检查相关脚本
├── benchmark/    # 基准测试脚本
├── sync/         # 同步相关脚本
├── download/     # 下载相关脚本
├── test/         # 测试相关脚本
├── fix/          # 修复相关脚本
└── misc/         # 其他杂项脚本
```

## 各目录说明

### install/
- `install_vllm.sh` - 安装 vLLM
- `install_transformers.sh` - 安装 Transformers
- `install_pytorch.sh` - 安装 PyTorch
- `install_editable.sh` - 可编辑模式安装
- 以及其他安装脚本...

### check/
- `check_server.sh` - 检查服务器状态
- `check_vllm.sh` - 检查 vLLM 安装
- `check_torch.sh` - 检查 PyTorch
- 以及其他检查脚本...

### benchmark/
- `run_phi3_moe_benchmark.sh` - 运行 Phi-3 MoE 基准测试
- `full_benchmark.sh` - 完整基准测试
- 以及其他基准测试脚本...

### sync/
- `sync_code.sh` - 同步代码
- `deploy_and_run.sh` - 部署并运行
- 以及其他同步脚本...

### download/
- `download_model.sh` - 下载模型
- `download_vllm.sh` - 下载 vLLM
- 以及其他下载脚本...

### test/
- `test_cgc.sh` - 测试 CGC
- `test_import.sh` - 测试导入
- 以及其他测试脚本...

### fix/
- `fix_numpy彻底.sh` - 修复 NumPy 问题
- 以及其他修复脚本...

### misc/
- `clear_cache.sh` - 清除缓存
- `server_install.sh` - 服务器安装
- 以及其他杂项脚本...

## 使用说明

```bash
# 运行安装脚本
bash scripts/install/install_vllm.sh

# 运行基准测试
bash scripts/benchmark/run_phi3_moe_benchmark.sh

# 同步代码
bash scripts/sync/sync_code.sh
```
