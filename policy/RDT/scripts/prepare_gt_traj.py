import os
import json
import numpy as np
import cv2
import h5py
import transforms3d as t3d

def xyzrpy_to_matrix(x, y, z, roll, pitch, yaw):
    """将xyzrpy转换为4x4变换矩阵"""
    rotation_matrix = t3d.euler.euler2mat(roll, pitch, yaw, 'rxyz')
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [x, y, z]
    return transform_matrix

def matrix_to_xyzrpy(matrix):
    """将4x4变换矩阵转换为xyzrpy"""
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    
    # 提取欧拉角（固定轴顺序：roll(x), pitch(y), yaw(z))
    roll, pitch, yaw = t3d.euler.mat2euler(rotation_matrix, 'rxyz')
    
    return np.array([translation[0], translation[1], translation[2], roll, pitch, yaw])

def world_to_camera_transform(world_pose, camera_matrix):
    """
    将世界坐标系下的位姿转换到相机坐标系下
    
    参数:
        world_pose: [x, y, z, roll, pitch, yaw] 或 [x, y, z, roll, pitch, yaw, gripper]
        camera_matrix: 相机在世界坐标系下的4x4变换矩阵
    
    返回:
        camera_pose: 在相机坐标系下的位姿 [x, y, z, roll, pitch, yaw]
    """
    # 提取位姿部分（忽略可能的gripper值）
    if len(world_pose) == 7:
        world_pose_6d = world_pose[:6]
        gripper = world_pose[6]
    else:
        world_pose_6d = world_pose
        gripper = None
    
    # 如果camera_matrix不是4x4矩阵，则补全为4x4齐次变换矩阵
    if camera_matrix.shape == (3, 4):
        camera_matrix = np.concatenate([camera_matrix, np.array([[0, 0, 0, 1]])], axis=0)
    elif camera_matrix.shape == (3, 3):
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = camera_matrix
    
    # 世界坐标系下的变换矩阵
    T_world_end = xyzrpy_to_matrix(*world_pose_6d)
    
    # 计算相机坐标系下的变换矩阵: T_camera_end = T_camera_world * T_world_end
    # 其中 T_camera_world = inv(T_world_camera)
    T_camera_world = np.linalg.inv(camera_matrix)
    T_camera_end = T_camera_world @ T_world_end
    
    # 转换回xyzrpy
    camera_pose_6d = matrix_to_xyzrpy(T_camera_end)
    
    # 如果需要，重新添加gripper值
    if gripper is not None:
        return np.concatenate([camera_pose_6d, [gripper]])
    else:
        return camera_pose_6d

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper_all, left_arm_all = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper_all, right_arm_all = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        qpos = []
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )
            state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)  # joint
            state = state.astype(np.float32)
            qpos.append(state)
        
        qpos = np.array(qpos, dtype=np.float32)

        left_endpose_world, left_gripper_endpose = (
            root["/endpose/left_endpose"][:,:-1],
            root["/endpose/left_gripper"][()],
        )
        right_endpose_world, right_gripper_endpose = (
            root["/endpose/right_endpose"][:,:-1],
            root["/endpose/right_gripper"][()],
        )
        head_cam_extrinsic = root["/observation/head_camera/extrinsic_cv"][()]

        left_endpose_cam = []
        right_endpose_cam = []
        for i in range(left_endpose_world.shape[0]):
            # left_6d_cam = world_to_camera_transform(left_endpose_world[i], head_cam_extrinsic[i])
            # right_6d_cam = world_to_camera_transform(right_endpose_world[i], head_cam_extrinsic[i])
            left_6d_cam = left_endpose_world[i]
            right_6d_cam = right_endpose_world[i]
            left_6d_cam = np.concatenate([left_6d_cam, [left_gripper_endpose[i]]])
            right_6d_cam = np.concatenate([right_6d_cam, [right_gripper_endpose[i]]])
            left_endpose_cam.append(left_6d_cam)
            right_endpose_cam.append(right_6d_cam)
        left_endpose_cam = np.array(left_endpose_cam)
        right_endpose_cam = np.array(right_endpose_cam)
        # 拼接left_endpose_cam和right_endpose_cam，组成dual_endpose_cam
        dual_endpose_cam = np.concatenate([left_endpose_cam, right_endpose_cam], axis=1)

    return qpos, dual_endpose_cam

def generate_gt_traj_label(task_path, seed_dir, episode_count, label_size):
    """
    生成 Ground Truth 轨迹标签
    """

    #加载seed列表
    with open(seed_dir, "r") as f:
        content = f.read().strip()
        # 支持空格分隔或换行分隔
        seeds = [int(s) for s in content.split() if s.strip()]
    seeds = seeds[:50] 

    traj_label_dict = {}
    hdf5_path = os.path.join(task_path, "data")
    print(f"Processing {episode_count} episodes from {hdf5_path}...")
    for episode_idx in range(episode_count):
        print(f"Processing episode {episode_idx}...")
        # 获取每个 episode 数据
        qpos, dual_endpose_cam = load_hdf5(hdf5_path + f"/episode{episode_idx}.hdf5")

        num_steps = qpos.shape[0]
        # We skip the first few still steps
        EPS = 1e-2
        # Get the idx of the first qpos whose delta exceeds the threshold
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")

        if first_idx < 3:
            first_idx = 3


        # 采样
        start_idx = first_idx - 1
        sampled_traj_labels = dual_endpose_cam[start_idx:]

        # 处理不足按 label_size 倍数的情况：在末尾补充最后一帧
        if len(sampled_traj_labels) % 64 != 0:
            pad_cnt = (64 - (len(sampled_traj_labels) % 64))
            sampled_traj_labels = np.concatenate(
            [sampled_traj_labels, np.tile(sampled_traj_labels[-1:], (pad_cnt, 1))],
            axis=0
            )        
        LABEL_SIZE = label_size
        NUM_LABEL  = len(sampled_traj_labels) // label_size

        def bucketize(delta, pos_eps=0.02, rot_eps=0.05, grip_eps=0.01):
            """Map continuous delta to {0,1,2} where 0: negative, 1: zero, 2: positive."""
            out = np.ones_like(delta, dtype=np.int64)  # 中性默认 1
            idx = np.arange(delta.shape[-1]) % 7
            pos_mask  = idx < 3           # xyz
            rot_mask  = (idx >= 3) & (idx < 6)  # rpy
            grip_mask = idx == 6          # gripper
            out[pos_mask  & (delta >  pos_eps)]  = 2
            out[pos_mask  & (delta < -pos_eps)]  = 0
            out[rot_mask & (delta >  rot_eps)]  = 2
            out[rot_mask & (delta < -rot_eps)]  = 0
            out[grip_mask & (delta >  grip_eps)] = 2
            out[grip_mask & (delta < -grip_eps)] = 0
            return out


        traj_label = []
        for i in range(NUM_LABEL):
            s0 = i * LABEL_SIZE
            s1 = s0 + LABEL_SIZE - 1
            delta = sampled_traj_labels[s1] - sampled_traj_labels[s0]
            traj_label.append(bucketize(delta))
        traj_label_txt = "_".join([str(label) for label in traj_label])
        traj_label_dict[str(seeds[episode_idx])] = traj_label_txt


        print(f"Processed episode {episode_idx} successfully!")

    # 将 traj_label_dict 转换为可 JSON 序列化的结构（将 numpy 数组转换为 Python 列表/整数）
    json_data = []
    json_data.append(traj_label_dict)

    json_save_path = os.path.join(task_path, "traj_label_gt.json")
    with open(json_save_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Saved gt traj labels to {json_save_path}")

if __name__ == "__main__":
    # 运行脚本
    dataset_path = '/mnt/data-1/data/shaojiangnan/code/MAT-VLA/datasets/RoboTwin2.0/dataset'
    
    task_name = 'stack_bowls_three'
    task_level = 'clean'
    episode_num = 50

    task_path = os.path.join(dataset_path, task_name, f"aloha-agilex_{task_level}_{episode_num}")
    seed_dir = os.path.join(task_path, "seed.txt")

    generate_gt_traj_label(task_path, seed_dir, episode_count=50, label_size=8)
