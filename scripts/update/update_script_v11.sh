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
pip install \"numpy<2.0.0\" gymnasium websockets opencv-python Pillow --index-url https://pypi.org/simple --default-timeout=1000

echo \"=========================================\"
echo \"🤖 [Docker] 啟動 RoboTwin2.0 仿真 WebSocket\"
echo \"=========================================\"
cd /workspace/dexbotic
# 強制在背景啟動 RoboTwin2.0 的 websocket server，並將日誌輸出到 robotwin.log 方便除錯
bash scripts/env_sh/robotwin2.sh > /workspace/robotwin.log 2>&1 &
echo \"正在等待 RoboTwin 模擬器啟動...\"
sleep 20

echo \"=========================================\"
echo \"🚀 [Docker] 啟動 Q2RL x FLASH 測試腳本\"
echo \"=========================================\"
export PYTHONPATH=/workspace/dexbotic:\$PYTHONPATH
cd /workspace
python q2rl_vla_integration.py
INNER_EOF"
