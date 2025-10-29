import sys

sys.path.append("./")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml
from scripts.encode_lang_batch_once import encode_lang
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
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

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
            left_6d_cam = world_to_camera_transform(left_endpose_world[i], head_cam_extrinsic[i])
            right_6d_cam = world_to_camera_transform(right_endpose_world[i], head_cam_extrinsic[i])
            left_6d_cam = np.concatenate([left_6d_cam, [left_gripper_endpose[i]]])
            right_6d_cam = np.concatenate([right_6d_cam, [right_gripper_endpose[i]]])
            left_endpose_cam.append(left_6d_cam)
            right_endpose_cam.append(right_6d_cam)
        left_endpose_cam = np.array(left_endpose_cam)
        right_endpose_cam = np.array(right_endpose_cam)
        # 拼接left_endpose_cam和right_endpose_cam，组成dual_endpose_cam
        dual_endpose_cam = np.concatenate([left_endpose_cam, right_endpose_cam], axis=1)
        
    return left_gripper, left_arm, right_gripper, right_arm, image_dict, dual_endpose_cam


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict, dual_endpose_cam = (load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []
        dual_endpose = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)  # joint
            state = state.astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:

                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])
                dual_endpose.append(dual_endpose_cam[j])
                

        if not os.path.exists(os.path.join(save_path, f"episode_{i}")):
            os.makedirs(os.path.join(save_path, f"episode_{i}"))
        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            f.create_dataset("dual_endpose", data=np.array(dual_endpose))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("/mnt/pfs/users/jiangnan.shao/code/RoboTwin/datasets/RoboTwin2.0/dataset", str(task_name), f"{task_config}_{expert_data_num}", "data")

    print(f"read data from path: {load_dir}")
    begin = data_transform(
        load_dir,
        expert_data_num,
        f"./processed_data_traj/{task_name}-{task_config}-{expert_data_num}",
    )
    tokenizer, text_encoder = None, None
    for idx in range(expert_data_num):
        print(f"Processing Language: {idx}", end="\r")
        data_file_path = (f"/mnt/pfs/users/jiangnan.shao/code/RoboTwin/datasets/RoboTwin2.0/dataset/{task_name}/{task_config}_{expert_data_num}/instructions/episode{idx}.json")
        target_dir = (f"processed_data_traj/{task_name}-{task_config}-{expert_data_num}/episode_{idx}")
        tokenizer, text_encoder = encode_lang(
            DATA_FILE_PATH=data_file_path,
            TARGET_DIR=target_dir,
            GPU=0,
            desc_type="seen",
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
