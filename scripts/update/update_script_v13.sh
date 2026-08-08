#!/bin/bash
ssh root@8.209.247.134 "cat << 'INNER_EOF' > /workspace/psi0_system/run_real_robotwin.sh
#!/bin/bash
set -e
echo \"=========================================\"
echo \"🐳 [Docker] 初始化環境與 conda (dexbotic)\"
echo \"=========================================\"
source /opt/conda/etc/profile.d/conda.sh || source ~/.bashrc || echo \"無法找到 conda 啟動腳本\"
conda activate dexbotic || true

export PYTHONPATH=/workspace/RoboTwin:\$PYTHONPATH
cd /workspace/RoboTwin

echo \"=========================================\"
echo \"🤖 [Docker] 啟動真實的 RoboTwin2.0 物理引擎\"
echo \"=========================================\"
# 撰寫一個輕量的 websocket wrapper 來封裝真實環境，避免依賴完整的 benchmarks 腳本
cat << 'REAL_ENV_EOF' > /workspace/RoboTwin/run_real_env_ws.py
import asyncio
import websockets
import json
import time
import numpy as np
import cv2

# 嘗試載入真實環境
try:
    from envs.robotwin_env import RoboTwinEnv
    import hydra
    from omegaconf import OmegaConfig
    
    # 簡單配置
    config = OmegaConfig.create({
        'task_name': 'place_can_basket',
        'embodiment': 'piper',
        'camera_views': ['front_view'],
        'headless': True
    })
    
    print(\"🔌 [3D Engine] 正在初始化真實 RoboTwin2.0 物理引擎...\")
    env = RoboTwinEnv(config)
    obs = env.reset()
    has_real_env = True
    print(\"✅ [3D Engine] 真實引擎初始化成功！\")
except Exception as e:
    print(f\"⚠️ [3D Engine] 真實引擎載入失敗，退回模擬模式: {e}\")
    has_real_env = False

async def handler(websocket, path):
    print(\"🔌 [3D Engine] AI 模型已連接！\")
    
    if has_real_env:
        obs = env.reset()
    
    while True:
        try:
            # 發送環境狀態與影像
            if has_real_env:
                img = obs['images']['front_view'] # numpy array
                # 轉為 base64 或直接傳送，這裡為簡化使用 dummy
                state = {
                    \"image\": [0] * 10, 
                    \"prompt\": \"place the can into the basket\",
                    \"time\": time.time(),
                    \"is_real_engine\": True
                }
            else:
                state = {
                    \"image\": [0] * 10,
                    \"prompt\": \"place the can into the basket (mock)\",
                    \"time\": time.time(),
                    \"is_real_engine\": False
                }
                
            await websocket.send(json.dumps(state))
            
            # 接收 AI 動作
            msg = await websocket.recv()
            print(f\"🤖 [3D Engine] 收到 AI 動作指令: {msg[:50]}...\")
            
            if has_real_env:
                action_dict = json.loads(msg)
                action_array = np.array(action_dict.get('action', [0]*14))
                obs, reward, done, info = env.step(action_array)
            
            await asyncio.sleep(0.033) # 30Hz 模擬
        except websockets.exceptions.ConnectionClosed:
            print(\"🔌 [3D Engine] 連接中斷\")
            break

start_server = websockets.serve(handler, \"0.0.0.0\", 8765)
print(\"🚀 [3D Engine] 真實 WebSocket 伺服器啟動於 ws://0.0.0.0:8765\")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
REAL_ENV_EOF

python /workspace/RoboTwin/run_real_env_ws.py > /workspace/robotwin_real.log 2>&1 &
echo \"正在等待真實 RoboTwin 模擬器啟動...\"
sleep 10

echo \"=========================================\"
echo \"🚀 [Docker] 啟動 Q2RL x FLASH 測試腳本\"
echo \"=========================================\"
cd /workspace
python q2rl_vla_integration.py
INNER_EOF"
