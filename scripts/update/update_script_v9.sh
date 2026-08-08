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
echo \"🔄 確保關鍵依賴完全安裝...\"
pip install \"numpy<2.0.0\" gymnasium websockets opencv-python Pillow

echo \"=========================================\"
echo \"🤖 [Docker] 啟動 RoboTwin2.0 仿真 WebSocket\"
echo \"=========================================\"
cd /workspace/dexbotic
bash scripts/env_sh/robotwin2.sh &
sleep 5

echo \"=========================================\"
echo \"🚀 [Docker] 啟動 Q2RL x FLASH 測試腳本\"
echo \"=========================================\"
export PYTHONPATH=/workspace/dexbotic:\$PYTHONPATH
cd /workspace
python q2rl_vla_integration.py
INNER_EOF"
