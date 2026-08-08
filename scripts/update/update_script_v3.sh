#!/bin/bash
ssh root@8.209.247.134 "cat << 'INNER_EOF' > /workspace/psi0_system/run_test_in_docker.sh
#!/bin/bash
set -e
echo \"=========================================\"
echo \"🐳 [Docker] 初始化環境與 conda (dexbotic)\"
echo \"=========================================\"
source /opt/conda/etc/profile.d/conda.sh || source ~/.bashrc || echo \"無法找到 conda 啟動腳本\"
conda activate dexbotic || true

cd /workspace/dexbotic
echo \"🔄 修改 DexBotic 的 pyproject.toml 避免覆蓋 PyTorch 11.8...\"
sed -i 's/\"torch>=2.6.0\"/\"torch>=2.1.0\"/g' pyproject.toml

echo \"🔄 安裝 DexBotic 依賴...\"
pip install -e . --default-timeout=1000

echo \"=========================================\"
echo \"🤖 [Docker] 啟動 RoboTwin2.0 仿真 WebSocket\"
echo \"=========================================\"
cd /workspace/dexbotic
bash scripts/env_sh/robotwin2.sh &
sleep 15

echo \"=========================================\"
echo \"🚀 [Docker] 啟動 Q2RL x FLASH 測試腳本\"
echo \"=========================================\"
cd /workspace
python q2rl_vla_integration.py
INNER_EOF"
