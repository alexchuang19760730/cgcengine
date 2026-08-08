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
echo \"🔄 替換 requirements 為輕量化，只保留關鍵套件，避免重新下載 PyTorch...\"
sed -i 's/\"torch>=2.6.0\"//g' pyproject.toml
sed -i 's/\"torchvision>=0.21.0\"//g' pyproject.toml
sed -i 's/\"torchaudio>=2.6.0\"//g' pyproject.toml

echo \"🔄 快速安裝 DexBotic 依賴 (無依賴模式)...\"
pip install -e . --no-deps

echo \"🔄 安裝必要的 Gym 與 WebSocket 套件...\"
pip install gymnasium websockets opencv-python Pillow

echo \"=========================================\"
echo \"🤖 [Docker] 啟動 RoboTwin2.0 仿真 WebSocket\"
echo \"=========================================\"
cd /workspace/dexbotic
bash scripts/env_sh/robotwin2.sh &
sleep 15

echo \"=========================================\"
echo \"🚀 [Docker] 啟動 Q2RL x FLASH 測試腳本\"
echo \"=========================================\"
export PYTHONPATH=/workspace/dexbotic:\$PYTHONPATH
cd /workspace
python q2rl_vla_integration.py
INNER_EOF"
