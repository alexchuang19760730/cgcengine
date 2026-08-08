
"""
EgoPoseFormer 到 GMR 转换脚本
功能：
1. 读取 EgoPoseFormer 输出的高精度 3D 关键点
2. 转换成 GMR 能接受的 SMPLX 格式
3. 调用 GMR 进行运动重定向
"""

import numpy as np
import argparse
from pathlib import Path

# EgoPoseFormer 到 SMPLX 关节的映射
# 这里只是示意，需要根据实际 EgoPoseFormer 输出格式调整
EGOPOSEFORMER_TO_SMPLX_MAPPING = {
    'root': 'pelvis',
    'hip_l': 'left_hip',
    'knee_l': 'left_knee',
    'ankle_l': 'left_ankle',
    'foot_l': 'left_foot',
    'hip_r': 'right_hip',
    'knee_r': 'right_knee',
    'ankle_r': 'right_ankle',
    'foot_r': 'right_foot',
    'spine': 'spine1',
    'spine1': 'spine2',
    'neck': 'neck',
    'head': 'head',
    'shoulder_l': 'left_shoulder',
    'elbow_l': 'left_elbow',
    'wrist_l': 'left_wrist',
    'hand_l': 'left_hand',
    'shoulder_r': 'right_shoulder',
    'elbow_r': 'right_elbow',
    'wrist_r': 'right_wrist',
    'hand_r': 'right_hand',
}

# 单位四元数 [w, x, y, z]
UNIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def load_egoposeformer_data(data_path):
    """
    读取 EgoPoseFormer 输出的数据
    注意: 需要根据实际 EgoPoseFormer 输出格式调整此函数
    """
    print(f"Loading EgoPoseFormer data from: {data_path}")

    # 示意实现 - 需要根据 EgoPoseFormer 实际输出调整
    # 假设 EgoPoseFormer 输出 npy 文件或者 h5 文件
    data = np.load(data_path, allow_pickle=True)

    if isinstance(data, dict):
        print("Keys in data:", list(data.keys()))
        # 假设包含 'keypoints_3d' 和 'rotations'
        if 'keypoints_3d' in data:
            keypoints_3d = data['keypoints_3d']
            rotations = data.get('rotations', None)
            print(f"Loaded keypoints shape: {keypoints_3d.shape}")
            return keypoints_3d, rotations
    elif isinstance(data, np.ndarray):
        print(f"Loaded array shape: {data.shape}")
        return data, None
    else:
        raise ValueError("Unknown data format")


def convert_to_gmr_format(keypoints_3d, rotations=None):
    """
    把 EgoPoseFormer 的数据转换成 GMR 能接受的 SMPLX 格式
    """
    n_frames = keypoints_3d.shape[0]

    # 构造 GMR 格式的数据
    gmr_data = []

    for frame_idx in range(n_frames):
        frame_data = {}

        # 示意: 需要根据实际数据结构调整
        # 假设 keypoints_3d 的格式是 (n_frames, n_joints, 3)
        frame_keypoints = keypoints_3d[frame_idx]

        # 如果有旋转数据，就用真实的旋转；否则用单位四元数
        if rotations is not None:
            frame_rotations = rotations[frame_idx]
        else:
            frame_rotations = None

        for epf_joint, smplx_joint in EGOPOSEFORMER_TO_SMPLX_MAPPING.items():
            # 示意: 需要根据实际索引调整
            # 假设 EgoPoseFormer 的关节顺序和我们的映射一致
            try:
                # 这里只是示意，需要根据实际数据结构调整索引
                joint_idx = list(EGOPOSEFORMER_TO_SMPLX_MAPPING.keys()).index(epf_joint)
                pos_3d = frame_keypoints[joint_idx]

                if frame_rotations is not None:
                    quat = frame_rotations[joint_idx]
                else:
                    quat = UNIT_QUAT

                frame_data[smplx_joint] = (pos_3d, quat)
            except Exception as e:
                print(f"Warning: Could not process joint {epf_joint}: {e}")
                frame_data[smplx_joint] = (np.zeros(3), UNIT_QUAT)

        gmr_data.append(frame_data)

    print(f"Converted {n_frames} frames to GMR format")
    return gmr_data


def main():
    parser = argparse.ArgumentParser(description="EgoPoseFormer -> GMR -> Robot")
    parser.add_argument("--egoposeformer_data", type=str, required=True,
                        help="Path to EgoPoseFormer output data")
    parser.add_argument("--robot", type=str, default="unitree_g1",
                        help="Robot name (e.g., unitree_g1)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save robot motion")
    parser.add_argument("--rot_weight", type=float, default=1.0,
                        help="Rotation weight (default: 1.0, use both position and rotation)")
    args = parser.parse_args()

    # 1. 加载 EgoPoseFormer 数据
    keypoints_3d, rotations = load_egoposeformer_data(args.egoposeformer_data)

    # 2. 转换成 GMR 格式
    gmr_input_data = convert_to_gmr_format(keypoints_3d, rotations)

    # 3. 调用 GMR 重定向
    print("Calling GMR motion retargeting...")
    print("Note: Full integration requires importing GMR modules")
    print("This is a template script - please integrate with actual GMR API")

    # 实际调用时，类似:
    # from general_motion_retargeting import GeneralMotionRetargeting
    # gmr = GeneralMotionRetargeting(robot_name=args.robot, rot_weight=args.rot_weight)
    # robot_motion = gmr.retarget(gmr_input_data)
    # robot_motion.save(args.save_path)
    # ...

    print("Script completed.")


if __name__ == "__main__":
    main()

