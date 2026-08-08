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
# 因為沒有原始腳本，我們自己啟動一個輕量的 websocket 伺服器來模擬 3D 引擎的推流
cat << 'MOCK_EOF' > mock_websocket_server.py
import asyncio
import websockets
import json
import time

async def handler(websocket, path):
    print(\"🔌 [3D Engine] 收到連接請求！\")
    while True:
        try:
            # 模擬 3D 渲染與推流 (發送假圖片與環境狀態)
            dummy_state = {
                \"image\": [0] * 10, # 簡化
                \"prompt\": \"pick up the red cube\",
                \"time\": time.time()
            }
            await websocket.send(json.dumps(dummy_state))
            # 接收 AI 動作
            msg = await websocket.recv()
            print(f\"🤖 [3D Engine] 收到 AI 動作指令: {msg[:50]}...\")
            await asyncio.sleep(0.033) # 30Hz 模擬
        except websockets.exceptions.ConnectionClosed:
            print(\"🔌 [3D Engine] 連接中斷\")
            break

start_server = websockets.serve(handler, \"0.0.0.0\", 8765)
print(\"🚀 [3D Engine] WebSocket 模擬伺服器啟動於 ws://0.0.0.0:8765\")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
MOCK_EOF

python mock_websocket_server.py > /workspace/robotwin.log 2>&1 &
echo \"正在等待 RoboTwin 模擬器啟動...\"
sleep 5

echo \"=========================================\"
echo \"🚀 [Docker] 啟動 Q2RL x FLASH 測試腳本\"
echo \"=========================================\"
export PYTHONPATH=/workspace/dexbotic:\$PYTHONPATH
cd /workspace
python q2rl_vla_integration.py
INNER_EOF"
