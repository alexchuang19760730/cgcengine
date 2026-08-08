"""
EgoPoseFormer + GMR 整合脚本
功能：
1. 读取 EgoPoseFormer 输出的 3D 关键点
2. 转换成 GMR 能接受的格式（body_name -> (position, rotation)）
3. 调用 GMR 进行运动重定向
4. 输出机器人关节角度

使用方式：
python egoposeformer_gmr_integration.py \
    --egoposeformer_data /path/to/epf_output.npy \
    --output /path/to/robot_motion.pkl

目录结构：
EgoPosePredictor/
├── models/
│   ├── EgoPoseFormer/     # EgoPoseFormer 模型
│   └── GMR/               # GMR 模型
├── databases/
│   ├── action_atoms/      # 动作原子数据库
│   ├── grasp_points/      # 抓取点数据库
│   └── demo_features/     # 演示特征数据库
└── egoposeformer_gmr_integration.py  # 本脚本
"""

import numpy as np
import argparse
import sys
import os

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义模型路径（相对于脚本位置）
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
GMR_PATH = os.path.join(MODELS_DIR, 'GMR')
EGOPOSEFORMER_PATH = os.path.join(MODELS_DIR, 'EgoPoseFormer')

# 添加模型路径到 sys.path
if GMR_PATH not in sys.path:
    sys.path.insert(0, GMR_PATH)

# 尝试导入 GMR，如果失败则回退到旧路径
try:
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    print(f"✅ GMR loaded from: {GMR_PATH}")
except ImportError:
    # 回退到旧路径
    OLD_GMR_PATH = '/root/TWIST2/GMR'
    if OLD_GMR_PATH not in sys.path:
        sys.path.insert(0, OLD_GMR_PATH)
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    print(f"⚠️  Falling back to GMR at: {OLD_GMR_PATH}")


def load_egoposeformer_data(data_path):
    """
    读取 EgoPoseFormer 输出的数据
    支持格式: .npy (dict 或 array)
    """
    print(f"Loading EgoPoseFormer data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"EgoPoseFormer data not found: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    
    if isinstance(data, dict):
        print("Keys in EgoPoseFormer data:", list(data.keys()))
        
        keypoints_3d = data.get('keypoints_3d', None)
        rotations = data.get('rotations', None)
        confidence = data.get('confidence', None)
        
        if keypoints_3d is not None:
            print(f"Loaded keypoints shape: {keypoints_3d.shape}")
            print(f"Confidence range: {confidence.min():.3f} - {confidence.max():.3f}" if confidence is not None else "")
            return keypoints_3d, rotations, confidence
        else:
            raise ValueError("EgoPoseFormer data does not contain 'keypoints_3d'")
    
    elif isinstance(data, np.ndarray):
        print(f"Loaded array shape: {data.shape}")
        return data, None, None
    
    else:
        raise ValueError("Unknown data format")


def convert_to_gmr_format(keypoints_3d, rotations=None):
    """
    把 EgoPoseFormer 的 3D 关键点转换成 GMR 格式
    GMR 期望的格式: dict, 键是身体部位名称，值是 (position, rotation)
    position: np.array([x, y, z])
    rotation: np.array([w, x, y, z]) 或 3x3 旋转矩阵
    """
    n_frames = keypoints_3d.shape[0]
    print(f"Converting {n_frames} frames to GMR format...")
    
    # EgoPoseFormer 关节索引到 GMR 身体部位名称的映射
    EPF_JOINT_MAP = {
        0: 'pelvis',      # 骨盆
        1: 'left_hip',    # 左髋
        2: 'left_knee',   # 左膝
        3: 'left_ankle',  # 左踝
        4: 'left_foot',   # 左脚
        5: 'right_hip',   # 右髋
        6: 'right_knee',  # 右膝
        7: 'right_ankle', # 右踝
        8: 'right_foot',  # 右脚
        9: 'spine1',      # 脊柱1
        10: 'spine2',     # 脊柱2
        11: 'neck',       # 脖子
        12: 'head',       # 头部
        13: 'left_shoulder',   # 左肩
        14: 'left_elbow',      # 左肘
        15: 'left_wrist',      # 左腕
        16: 'left_hand',       # 左手
        17: 'right_shoulder',  # 右肩
        18: 'right_elbow',     # 右肘
        19: 'right_wrist',     # 右腕
        20: 'right_hand',      # 右手
    }
    
    # 单位四元数 [w, x, y, z]
    UNIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
    
    # 转换后的帧数据列表
    gmr_frames = []
    
    for frame_idx in range(n_frames):
        frame_data = {}
        
        for epf_idx, body_name in EPF_JOINT_MAP.items():
            try:
                pos_3d = keypoints_3d[frame_idx, epf_idx]
                
                if rotations is not None and frame_idx < rotations.shape[0]:
                    if rotations.ndim == 3 and rotations.shape[-1] == 4:
                        quat = rotations[frame_idx, epf_idx]
                    elif rotations.ndim == 4 and rotations.shape[-2:] == (3, 3):
                        from scipy.spatial.transform import Rotation as R
                        quat = R.from_matrix(rotations[frame_idx, epf_idx]).as_quat()
                        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
                    else:
                        quat = UNIT_QUAT
                else:
                    quat = UNIT_QUAT
                
                frame_data[body_name] = (pos_3d.astype(np.float64), quat.astype(np.float64))
                
            except Exception as e:
                print(f"Warning: Could not process joint {body_name} (idx {epf_idx}): {e}")
                frame_data[body_name] = (np.zeros(3, dtype=np.float64), UNIT_QUAT.copy())
        
        gmr_frames.append(frame_data)
    
    print(f"Successfully converted {len(gmr_frames)} frames")
    return gmr_frames


def retarget_frames(gmr_frames, src_human='smpl', tgt_robot='unitree_g1', verbose=True):
    """
    使用 GMR 对多帧数据进行运动重定向
    """
    print(f"\nInitializing GMR with {src_human} -> {tgt_robot}...")
    
    gmr = GeneralMotionRetargeting(
        src_human=src_human,
        tgt_robot=tgt_robot,
        verbose=verbose
    )
    
    print(f"Retargeting {len(gmr_frames)} frames...")
    
    robot_motions = []
    
    for frame_idx, human_data in enumerate(gmr_frames):
        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx}/{len(gmr_frames)}...")
        
        gmr.retarget(human_data)
        
        qpos = gmr.configuration.qpos.copy()
        
        motion_dict = {
            'frame_idx': frame_idx,
            'qpos': qpos,
            'dof_names': list(gmr.robot_dof_names.keys()),
            'motor_names': list(gmr.robot_motor_names.keys())
        }
        
        robot_motions.append(motion_dict)
    
    print(f"Successfully retargeted {len(robot_motions)} frames")
    return robot_motions, gmr


def save_robot_motion(robot_motions, output_path):
    """
    保存机器人运动数据
    """
    import pickle
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(robot_motions, f)
    
    print(f"Robot motion saved to: {output_path}")


def init_models():
    """
    初始化模型目录结构
    """
    print(f"\n📁 Checking model directories...")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    
    # 创建模型目录（如果不存在）
    os.makedirs(GMR_PATH, exist_ok=True)
    os.makedirs(EGOPOSEFORMER_PATH, exist_ok=True)
    
    # 检查 GMR 是否存在
    if os.listdir(GMR_PATH):
        print(f"✅ GMR model exists in: {GMR_PATH}")
    else:
        print(f"⚠️  GMR model not found in {GMR_PATH}")
        print(f"   Please copy GMR to: {GMR_PATH}")
        print(f"   Example: cp -r /root/TWIST2/GMR/* {GMR_PATH}/")
    
    # 检查 EgoPoseFormer 是否存在
    if os.listdir(EGOPOSEFORMER_PATH):
        print(f"✅ EgoPoseFormer model exists in: {EGOPOSEFORMER_PATH}")
    else:
        print(f"⚠️  EgoPoseFormer model not found in {EGOPOSEFORMER_PATH}")
        print(f"   Please copy EgoPoseFormer to: {EGOPOSEFORMER_PATH}")


def main():
    parser = argparse.ArgumentParser(description="EgoPoseFormer + GMR Integration")
    parser.add_argument("--egoposeformer_data", type=str, required=True,
                        help="Path to EgoPoseFormer output data (.npy file)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save robot motion (.pkl file)")
    parser.add_argument("--src_human", type=str, default='smpl',
                        help="Source human model type (default: smpl)")
    parser.add_argument("--tgt_robot", type=str, default='unitree_g1',
                        help="Target robot name (default: unitree_g1)")
    parser.add_argument("--verbose", action='store_true', default=True,
                        help="Enable verbose output")
    parser.add_argument("--init", action='store_true', default=False,
                        help="Initialize model directories")
    args = parser.parse_args()
    
    if args.init:
        init_models()
        return
    
    try:
        keypoints_3d, rotations, confidence = load_egoposeformer_data(args.egoposeformer_data)
        
        gmr_frames = convert_to_gmr_format(keypoints_3d, rotations)
        
        robot_motions, gmr = retarget_frames(
            gmr_frames,
            src_human=args.src_human,
            tgt_robot=args.tgt_robot,
            verbose=args.verbose
        )
        
        save_robot_motion(robot_motions, args.output)
        
        print("\n✅ Integration completed successfully!")
        print(f"Input: {args.egoposeformer_data}")
        print(f"Output: {args.output}")
        print(f"Robot: {args.tgt_robot}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()