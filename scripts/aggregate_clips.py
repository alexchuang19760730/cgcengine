#!/usr/bin/env python3
"""
GeoMirror 数据聚合脚本
自动扫描 robot 数据和 video/EGE 数据，按操作轮次聚合到 clips 目录

输出目录结构 (符合 GeoMirror 01_data_collection 规格):
  data/clips/
  ├── manifests/
  │   └── episodes.json       # 所有 episode 的汇总信息
  ├── raw/                    # 原始数据
  │   └── <task>_round<num>/
  │       ├── data/           # 动捕CSV、rosbag、topic映射
  │       │   ├── merged_data.csv
  │       │   ├── rosbag/
  │       │   └── topic_mapping.txt
  │       └── video/          # 视频数据
  │           ├── ego/        # EGE数据 (Transform/RGB_Images/SLAM_Poses/Calibration/IMU)
  │           └── robot/      # Robot相机图像
  └── synced/                 # 处理后的聚合包
      └── <task>_round<num>/
          ├── episode.npz
          ├── merged_data.csv
          └── metadata.json
"""

import os
import sys
import json
import shutil
import glob
import re
from pathlib import Path

# 根目录配置
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
ROBOT_SOURCE_DIR = DATA_DIR / 'source/robot/data'
VIDEO_SOURCE_DIR = DATA_DIR / 'source/video/data/fastumi/DATA'
OUTPUT_DIR = DATA_DIR / 'clips'
TOPIC_MAPPING_FILE = DATA_DIR / 'archive/Robot/topic/topic_list.txt'


def find_robot_rounds():
    """扫描所有 robot 操作轮次"""
    rounds = []

    if not ROBOT_SOURCE_DIR.exists():
        print(f"警告: Robot 数据源目录不存在: {ROBOT_SOURCE_DIR}")
        return rounds

    for task_dir in ROBOT_SOURCE_DIR.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        m = re.match(r'(\d{4})-(.+)-syn', task_name)
        if m:
            date_str = m.group(1)
            clean_task_name = m.group(2)
        else:
            date_str = 'unknown'
            clean_task_name = task_name

        for round_dir in task_dir.iterdir():
            if not round_dir.is_dir():
                continue

            try:
                round_num = int(round_dir.name)
            except ValueError:
                continue

            csv_files = list(round_dir.glob('*.csv'))
            bag_dirs = list(round_dir.glob('bag_*'))
            photo_dirs = [p for p in round_dir.glob('photo_*') if p.is_dir()]

            if not csv_files and not bag_dirs:
                continue

            rounds.append({
                'task_dir': task_dir,
                'round_dir': round_dir,
                'task_name': clean_task_name,
                'date_str': date_str,
                'round_num': round_num,
                'csv_files': csv_files,
                'bag_dirs': bag_dirs,
                'photo_dirs': photo_dirs
            })

    return sorted(rounds, key=lambda x: (x['date_str'], x['task_name'], x['round_num']))


def find_video_sessions():
    """扫描所有 video/EGE 会话"""
    sessions = []

    if not VIDEO_SOURCE_DIR.exists():
        print(f"警告: Video 数据源目录不存在: {VIDEO_SOURCE_DIR}")
        return sessions

    for multi_session_dir in VIDEO_SOURCE_DIR.iterdir():
        if not multi_session_dir.is_dir():
            continue
        if not multi_session_dir.name.startswith('multi_session_'):
            continue

        date_str = multi_session_dir.name.replace('multi_session_', '')

        for session_dir in multi_session_dir.iterdir():
            if not session_dir.is_dir():
                continue
            if not session_dir.name.startswith('session_'):
                continue

            session_id = session_dir.name.replace('session_', '')

            transform_dir = session_dir / 'Transform'
            rgb_images_dir = session_dir / 'RGB_Images'
            slam_poses_dir = session_dir / 'SLAM_Poses'

            sessions.append({
                'session_dir': session_dir,
                'date_str': date_str,
                'session_id': session_id,
                'has_transform': transform_dir.exists(),
                'has_rgb': rgb_images_dir.exists(),
                'has_slam': slam_poses_dir.exists()
            })

    return sorted(sessions, key=lambda x: (x['date_str'], x['session_id']))


def match_robot_video(robot_rounds, video_sessions):
    """匹配 robot 轮次与 video 会话（基于日期）"""
    matches = []

    for robot in robot_rounds:
        matched_sessions = []
        robot_date = robot['date_str']

        for video in video_sessions:
            video_date = video['date_str']

            if robot_date == video_date:
                matched_sessions.append(video)
            elif len(robot_date) == 4 and len(video_date) == 8:
                if video_date.endswith(robot_date):
                    matched_sessions.append(video)
            elif len(robot_date) == 8 and len(video_date) == 4:
                if robot_date.endswith(video_date):
                    matched_sessions.append(video)

        matches.append({
            'robot': robot,
            'videos': matched_sessions
        })

    return matches


def create_raw_data(match):
    """创建 raw/ 目录下的原始数据"""
    robot = match['robot']
    task_name = robot['task_name']
    round_num = robot['round_num']
    clip_name = f"{task_name}_round{round_num}"

    # raw/<clip_name>/ 目录
    raw_dir = OUTPUT_DIR / 'raw' / clip_name
    raw_data_dir = raw_dir / 'data'
    raw_video_dir = raw_dir / 'video'
    raw_rosbag_dir = raw_data_dir / 'rosbag'

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    raw_video_dir.mkdir(parents=True, exist_ok=True)
    raw_rosbag_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制动捕 CSV -> raw/<clip_name>/data/merged_data.csv
    if robot['csv_files']:
        csv_src = robot['csv_files'][0]
        csv_dst = raw_data_dir / 'merged_data.csv'
        shutil.copy(csv_src, csv_dst)
        print(f"    复制原始动捕数据 -> raw/{clip_name}/data/")

    # 2. 复制 rosbag -> raw/<clip_name>/data/rosbag/
    if robot['bag_dirs']:
        bag_dir = robot['bag_dirs'][0]
        for bag_file in bag_dir.iterdir():
            if bag_file.suffix == '.db3':
                shutil.copy(bag_file, raw_rosbag_dir / 'rosbag_0.db3')
            elif bag_file.name == 'metadata.yaml':
                shutil.copy(bag_file, raw_rosbag_dir / 'metadata.yaml')
        print(f"    复制 rosbag -> raw/{clip_name}/data/rosbag/")

    # 3. 复制 topic_mapping.txt -> raw/<clip_name>/data/
    if TOPIC_MAPPING_FILE.exists():
        shutil.copy(TOPIC_MAPPING_FILE, raw_data_dir / 'topic_mapping.txt')
        print(f"    复制 topic_mapping.txt -> raw/{clip_name}/data/")

    # 4. 复制 Robot 相机图像 -> raw/<clip_name>/video/robot/
    robot_video_dir = raw_video_dir / 'robot'
    robot_video_dir.mkdir(exist_ok=True)

    if robot['photo_dirs']:
        for photo_dir in robot['photo_dirs']:
            photo_dst = robot_video_dir / photo_dir.name
            if photo_dst.exists():
                shutil.rmtree(photo_dst)
            shutil.copytree(photo_dir, photo_dst)
        print(f"    复制 Robot 图像 -> raw/{clip_name}/video/robot/")

    # 5. 复制 EGE Video 数据 -> raw/<clip_name>/video/ego/
    ego_video_dir = raw_video_dir / 'ego'
    ego_video_dir.mkdir(exist_ok=True)

    for video in match['videos']:
        session_dir = video['session_dir']

        # Transform
        transform_src = session_dir / 'Transform'
        if transform_src.exists():
            transform_dst = ego_video_dir / 'Transform'
            if transform_dst.exists():
                shutil.rmtree(transform_dst)
            shutil.copytree(transform_src, transform_dst)

        # RGB_Images
        rgb_src = session_dir / 'RGB_Images'
        if rgb_src.exists():
            rgb_dst = ego_video_dir / 'RGB_Images'
            if rgb_dst.exists():
                shutil.rmtree(rgb_dst)
            shutil.copytree(rgb_src, rgb_dst)

        # SLAM_Poses
        slam_src = session_dir / 'SLAM_Poses'
        if slam_src.exists():
            slam_dst = ego_video_dir / 'SLAM_Poses'
            if slam_dst.exists():
                shutil.rmtree(slam_dst)
            shutil.copytree(slam_src, slam_dst)

        # Calibration
        calib_src = session_dir / 'Calibration'
        if calib_src.exists():
            calib_dst = ego_video_dir / 'Calibration'
            if calib_dst.exists():
                shutil.rmtree(calib_dst)
            shutil.copytree(calib_src, calib_dst)

        # IMU
        imu_src = session_dir / 'IMU'
        if imu_src.exists():
            imu_dst = ego_video_dir / 'IMU'
            if imu_dst.exists():
                shutil.rmtree(imu_dst)
            shutil.copytree(imu_src, imu_dst)

    print(f"    复制 EGE 数据 -> raw/{clip_name}/video/ego/")
    return raw_dir


def create_synced_package(match):
    """创建 synced/ 目录下的处理后聚合包"""
    robot = match['robot']
    task_name = robot['task_name']
    round_num = robot['round_num']
    clip_name = f"{task_name}_round{round_num}"

    # synced/<clip_name>/ 目录
    synced_dir = OUTPUT_DIR / 'synced' / clip_name
    synced_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制动捕 CSV -> synced/<clip_name>/merged_data.csv
    if robot['csv_files']:
        csv_src = robot['csv_files'][0]
        csv_dst = synced_dir / 'merged_data.csv'
        shutil.copy(csv_src, csv_dst)
        print(f"    复制动捕数据 -> synced/{clip_name}/")

    # 2. 创建 metadata.json
    metadata = {
        "session_id": clip_name,
        "episode_id": clip_name,
        "robot_id": "gsrobot",
        "task_name": task_name,
        "scene_name": "unknown",
        "operator": "unknown",
        "round_number": round_num,
        "date_str": robot['date_str'],
        "source": {
            "robot": str(robot['round_dir']),
            "videos": [str(v['session_dir']) for v in match['videos']]
        }
    }

    with open(synced_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    创建 metadata.json")

    # 3. 生成 episode.npz
    generate_episode_npz(synced_dir)

    return synced_dir


def generate_episode_npz(clip_dir):
    """从 merged_data.csv 生成 episode.npz"""
    import numpy as np
    import csv

    csv_path = clip_dir / 'merged_data.csv'
    if not csv_path.exists():
        print(f"    警告: 未找到 merged_data.csv，跳过 episode.npz 生成")
        return

    timestamps = []
    xrobot_pelvis_px = []
    xrobot_pelvis_py = []
    xrobot_pelvis_pz = []
    left_hand_wrist_px = []
    left_hand_wrist_py = []
    left_hand_wrist_pz = []
    right_hand_wrist_px = []
    right_hand_wrist_py = []
    right_hand_wrist_pz = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(int(row.get('record_time_ns', row.get('sdk_record_time_ns', 0))))
                xrobot_pelvis_px.append(float(row.get('xrobot_Pelvis_px', 0)))
                xrobot_pelvis_py.append(float(row.get('xrobot_Pelvis_py', 0)))
                xrobot_pelvis_pz.append(float(row.get('xrobot_Pelvis_pz', 0)))
                left_hand_wrist_px.append(float(row.get('left_LeftHandWrist_px', row.get('left_wrist_px', 0))))
                left_hand_wrist_py.append(float(row.get('left_LeftHandWrist_py', row.get('left_wrist_py', 0))))
                left_hand_wrist_pz.append(float(row.get('left_LeftHandWrist_pz', row.get('left_wrist_pz', 0))))
                right_hand_wrist_px.append(float(row.get('right_RightHandWrist_px', row.get('right_wrist_px', 0))))
                right_hand_wrist_py.append(float(row.get('right_RightHandWrist_py', row.get('right_wrist_py', 0))))
                right_hand_wrist_pz.append(float(row.get('right_RightHandWrist_pz', row.get('right_wrist_pz', 0))))

        np.savez(clip_dir / 'episode.npz',
                 timestamps=np.array(timestamps, dtype=np.int64),
                 xrobot_Pelvis_px=np.array(xrobot_pelvis_px, dtype=np.float32),
                 xrobot_Pelvis_py=np.array(xrobot_pelvis_py, dtype=np.float32),
                 xrobot_Pelvis_pz=np.array(xrobot_pelvis_pz, dtype=np.float32),
                 left_hand_wrist_px=np.array(left_hand_wrist_px, dtype=np.float32),
                 left_hand_wrist_py=np.array(left_hand_wrist_py, dtype=np.float32),
                 left_hand_wrist_pz=np.array(left_hand_wrist_pz, dtype=np.float32),
                 right_hand_wrist_px=np.array(right_hand_wrist_px, dtype=np.float32),
                 right_hand_wrist_py=np.array(right_hand_wrist_py, dtype=np.float32),
                 right_hand_wrist_pz=np.array(right_hand_wrist_pz, dtype=np.float32))

        print(f"    生成 episode.npz")
    except Exception as e:
        print(f"    警告: 生成 episode.npz 失败: {e}")


def create_episodes_json(matches):
    """创建 manifests/episodes.json 汇总文件"""
    episodes = []

    for match in matches:
        robot = match['robot']
        task_name = robot['task_name']
        round_num = robot['round_num']
        clip_name = f"{task_name}_round{round_num}"

        episode_npz_path = OUTPUT_DIR / 'synced' / clip_name / 'episode.npz'
        source_csv_path = OUTPUT_DIR / 'raw' / clip_name / 'data' / 'merged_data.csv'
        episode_dir_path = OUTPUT_DIR / 'synced' / clip_name

        episode_info = {
            "episode_id": clip_name,
            "session_id": clip_name,
            "source_csv": str(source_csv_path),
            "episode_dir": str(episode_dir_path),
            "episode_npz": str(episode_npz_path)
        }

        episodes.append(episode_info)

    manifests_dir = OUTPUT_DIR / 'manifests'
    manifests_dir.mkdir(parents=True, exist_ok=True)

    episodes_json = {
        "dataset_root": str(OUTPUT_DIR),
        "source": str(DATA_DIR / 'source'),
        "format_version": "geomirror_dataset_v1",
        "episodes": episodes
    }

    with open(manifests_dir / 'episodes.json', 'w') as f:
        json.dump(episodes_json, f, indent=2)

    print(f"\n  ✓ 创建 manifests/episodes.json")


def main():
    print("=" * 60)
    print("GeoMirror 数据聚合脚本")
    print("=" * 60)

    # 扫描数据源
    print("\n1. 扫描 Robot 操作轮次...")
    robot_rounds = find_robot_rounds()
    print(f"   找到 {len(robot_rounds)} 个 robot 操作轮次")
    for r in robot_rounds:
        print(f"     - {r['task_name']} round{r['round_num']} ({r['date_str']})")

    print("\n2. 扫描 Video/EGE 会话...")
    video_sessions = find_video_sessions()
    print(f"   找到 {len(video_sessions)} 个 video 会话")
    for v in video_sessions:
        print(f"     - session_{v['session_id']} ({v['date_str']})")

    # 匹配数据
    print("\n3. 匹配 Robot 与 Video 数据...")
    matches = match_robot_video(robot_rounds, video_sessions)

    # 创建目录结构
    print("\n4. 创建目录结构...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, match in enumerate(matches):
        robot = match['robot']
        clip_name = f"{robot['task_name']}_round{robot['round_num']}"
        print(f"\n   [{i+1}/{len(matches)}] 处理: {clip_name}")

        # 创建 raw 目录
        print(f"   - 创建 raw 数据...")
        create_raw_data(match)

        # 创建 synced 目录
        print(f"   - 创建 synced 聚合包...")
        create_synced_package(match)

        print(f"   ✓ 完成: {clip_name}")

    # 创建 episodes.json
    create_episodes_json(matches)

    print("\n" + "=" * 60)
    print("聚合完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()