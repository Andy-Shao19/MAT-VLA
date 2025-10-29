import os
import json
import numpy as np
import cv2
import h5py
import transforms3d as t3d

def to_bgr_image(img_data):
    """Convert various image representations to an OpenCV BGR uint8 ndarray."""
    # Case 1: compressed bytes from HDF5 (variable-length)
    if isinstance(img_data, (bytes, np.void)):
        buf = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image bytes.")
    else:
        # Ensure numpy array
        if not isinstance(img_data, np.ndarray):
            img = np.asarray(img_data)
        else:
            img = img_data

    # Handle channel-first (C,H,W) -> (H,W,C)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.transpose(img, (1, 2, 0))

    # Convert dtype to uint8 if needed
    if img.dtype != np.uint8:
        # Heuristic: assume [0,1] floats
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # If RGBA -> BGR
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # If RGB -> BGR
    elif img.ndim == 3 and img.shape[2] == 3:
        # Heuristic: many datasets store RGB; OpenCV expects BGR.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

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
        
    return qpos, image_dict, dual_endpose_cam

def generate_vlm_data_with_sampling(hdf5_dir, instructions_dir, save_dir, task_name, task_level, episode_count=10, segment_size=8, total_segments=32, total_frames=256):
    """
    生成 Qwen3-VL 微调数据（包含多图+指令），并确保足够帧数后再进行补充。
    """

    special_tokens = [
        "<X_L_NEG>", "<X_L_ZERO>", "<X_L_POS>",
        "<Y_L_NEG>", "<Y_L_ZERO>", "<Y_L_POS>",
        "<Z_L_NEG>", "<Z_L_ZERO>", "<Z_L_POS>",
        "<ROLL_L_NEG>", "<ROLL_L_ZERO>", "<ROLL_L_POS>",
        "<PITCH_L_NEG>", "<PITCH_L_ZERO>", "<PITCH_L_POS>",
        "<YAW_L_NEG>", "<YAW_L_ZERO>", "<YAW_L_POS>",
        "<GR_L_NEG>", "<GR_L_ZERO>", "<GR_L_POS>",
        
        "<X_R_NEG>", "<X_R_ZERO>", "<X_R_POS>",
        "<Y_R_NEG>", "<Y_R_ZERO>", "<Y_R_POS>",
        "<Z_R_NEG>", "<Z_R_ZERO>", "<Z_R_POS>",
        "<ROLL_R_NEG>", "<ROLL_R_ZERO>", "<ROLL_R_POS>",
        "<PITCH_R_NEG>", "<PITCH_R_ZERO>", "<PITCH_R_POS>",
        "<YAW_R_NEG>", "<YAW_R_ZERO>", "<YAW_R_POS>",
        "<GR_R_NEG>", "<GR_R_ZERO>", "<GR_R_POS>",

        # Repeat the above 14 tokens for all segments (0-31)
        # for SEG2...SEG31.
    ]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    episode_info_list = []
    print(f"Processing {episode_count} episodes from {hdf5_dir}...")
    for episode_idx in range(episode_count):
        print(f"Processing episode {episode_idx}...")
        # 获取每个 episode 数据
        qpos, image_dict, dual_endpose_cam = load_hdf5(hdf5_dir + f"/episode{episode_idx}.hdf5")

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
        
        sample_num = 100
        for sample_idx in range(sample_num):
            # 随机采样
            start_idx = np.random.randint(first_idx - 1, num_steps)
            sampled_traj_labels = dual_endpose_cam[start_idx:start_idx + total_frames]

            # 处理不足32帧的情况：补充最后一帧
            if len(sampled_traj_labels) < total_frames:
                sampled_traj_labels = np.concatenate(
                    [sampled_traj_labels, np.tile(sampled_traj_labels[-1:], (total_frames - len(sampled_traj_labels), 1))],
                    axis=0
                )
            
            LABEL_SIZE = segment_size
            NUM_LABEL  = total_segments

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
            
            traj_label_txt = []
            txt_formal = ['<X_L_', '<Y_L_', '<Z_L_', '<ROLL_L_', '<PITCH_L_', '<YAW_L_', '<GR_L_',
                          '<X_R_', '<Y_R_', '<Z_R_', '<ROLL_R_', '<PITCH_R_', '<YAW_R_', '<GR_R_']
            for label in traj_label:
                for i, val in enumerate(label):
                    if val == 0:
                        traj_label_txt.append(txt_formal[i] + 'NEG>')
                    elif val == 1:
                        traj_label_txt.append(txt_formal[i] + 'ZERO>')
                    else:
                        traj_label_txt.append(txt_formal[i] + 'POS>')

            # 随机选择该 episode 对应的指令文件
            instruction_file = os.path.join(instructions_dir, f"episode{episode_idx}.json")
            with open(instruction_file, "r") as f:
                instructions = json.load(f)

            # 从 Robotwin 数据中提取图像
            image_paths = []
            for i in range(2):
                time_step = start_idx-(2-i)
                raw_img = image_dict["head_camera"][time_step]
                img_bgr = to_bgr_image(raw_img)

                image_path_to_save = f"{save_dir}/images/{task_name}/{task_level}/episode_{episode_idx}"
                os.makedirs(image_path_to_save, exist_ok=True)

                image_path = os.path.join(image_path_to_save, f"{time_step}_head_camera.jpg")
                ok = cv2.imwrite(image_path, img_bgr)
                if not ok:
                    raise IOError(f"Failed to write image to {image_path}")
                image_paths.append(image_path)

            # 构造每条数据的 `conversations` 部分
            conversation = {
                "from": "human",
                "value": f"<image>\nThis is the head camera image of the previous frame.\n<image>\nThis is the head camera image of the current frame.\n"
                        f"instruction: {instructions['seen'][sample_idx]}\n"
                        f"Please predict the sequence of 32 trajectory labels that the robotic arm will execute next based on the image and instructions. Each trajectory label contains 14 tokens, and the format of one trajectory label is "
                        f"[x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l, x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r]. "
                        f"Please output them in order, using special tokens to represent: { ' '.join(special_tokens) }."
            }

            conversation_gpt = {
                "from": "gpt",
                "value": f"{' '.join(traj_label_txt)}"  # 转换为 JSON 可处理格式
            }

            episode_info = {
                "image": image_paths,
                "conversations": [conversation, conversation_gpt]
            }
            episode_info_list.append(episode_info)
            
        print(f"Processed episode {episode_idx + 1}/{episode_count}.")

    # 保存为 `JSON`
    json_data = episode_info_list

    json_save_path = os.path.join(save_dir, f"{task_name}_{task_level}_{episode_count}.json")
    with open(json_save_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Saved VLM data to {json_save_path}")

# 运行脚本
hdf5_dir = '/mnt/pfs/users/jiangnan.shao/code/RoboTwin/datasets/RoboTwin2.0/dataset/stack_bowls_two/aloha-agilex_clean_50/data'
instructions_dir = '/mnt/pfs/users/jiangnan.shao/code/RoboTwin/datasets/RoboTwin2.0/dataset/stack_bowls_two/aloha-agilex_clean_50/instructions'
save_dir = '/mnt/pfs/users/jiangnan.shao/code/RoboTwin/datasets/qwen3_vl_data/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
task_name = 'stack_bowls_two'
task_level = 'clean'

generate_vlm_data_with_sampling(hdf5_dir, instructions_dir, save_dir, task_name, task_level, episode_count=2, segment_size=8, total_segments=32, total_frames=256)
