import os
import json
import numpy as np
import cv2
import h5py
import transforms3d as t3d
import concurrent.futures
import logging  # 新增

# 日志配置
def setup_logger(save_dir):
    log_path = os.path.join(save_dir, "process_data.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler()
        ]
    )

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

    return qpos, image_dict

def process_single_episode(episode_idx, hdf5_dir, instructions_dir, save_dir, task_name, task_level, segment_size, total_segments, total_frames):
    try:
        qpos, image_dict = load_hdf5(hdf5_dir + f"/episode{episode_idx}.hdf5")
        num_steps = qpos.shape[0]
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")

        if first_idx < 3:
            first_idx = 3

        dual_endpose_cam = qpos
        sample_num = len(dual_endpose_cam)-first_idx
        episode_info_list = []
        for sample_idx in range(sample_num):
            start_idx = first_idx-1 + sample_idx
            sampled_traj_labels = dual_endpose_cam[start_idx:start_idx + total_frames]
            if len(sampled_traj_labels) < total_frames:
                sampled_traj_labels = np.concatenate(
                    [sampled_traj_labels, np.tile(sampled_traj_labels[-1:], (total_frames - len(sampled_traj_labels), 1))],
                    axis=0
                )
            LABEL_SIZE = segment_size
            NUM_LABEL  = total_segments

            def bucketize(delta, pos_eps=0.01, rot_eps=0.01, grip_eps=0.01):
                out = np.ones_like(delta, dtype=np.int64)
                idx = np.arange(delta.shape[-1]) % 7
                pos_mask  = idx < 3
                rot_mask  = (idx >= 3) & (idx < 6)
                grip_mask = idx == 6
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
            for label in traj_label:
                traj_label_txt.append(''.join(label.astype(str).tolist()))

            instruction_file = os.path.join(instructions_dir, f"episode{episode_idx}.json")
            with open(instruction_file, "r") as f:
                instructions = json.load(f)

            image_paths = []
            for cam_name in ["left_camera", "head_camera", "right_camera"]:
                raw_img = image_dict[cam_name][start_idx]
                img_bgr = to_bgr_image(raw_img)
                image_path_to_save = f"{save_dir}/images/{task_name}/{task_level}/episode_{episode_idx}"
                os.makedirs(image_path_to_save, exist_ok=True)
                image_path = os.path.join(image_path_to_save, f"{start_idx}_{cam_name}.jpg")
                ok = cv2.imwrite(image_path, img_bgr)
                if not ok:
                    raise IOError(f"Failed to write image to {image_path}")
                image_paths.append(image_path)

            instruct_id = np.random.randint(1, len(instructions['seen']))
            endpose_txt = np.array2string(dual_endpose_cam[start_idx-2: start_idx+1], separator=', ', formatter={'float_kind': lambda x: f"{x:.6f}"},).replace('\n', '')

            conversation = {
                "from": "human",
                "value": f"You are a Aloha-AgileX robot using Joint-position control. The instruction is \"{instructions['seen'][instruct_id]}\".<image>\nThis is the left camera image of the current frame.\n<image>\nThis is the head camera image of the current frame.\n<image>\nThis is the right camera image of the current frame.\n"
                        f"The format of action is [left_arm_1, left_arm_2, left_arm_3, left_arm_4, left_arm_5, left_arm_6, left_gripper, right_arm_1, right_arm_2, right_arm_3, right_arm_4, right_arm_5, right_arm_6, right_gripper] and the previous three (including current) frames' actions are: {endpose_txt}.\n"
                        f"We define a meta action label as follows: every 8 frames, take the action differences and each component is mapped to 0 if below 0.01, 1 if within [-0.01, 0.01], and 2 if above 0.01.\n"
                        f"Please predict the next 8 meta action labels based on the images, instruction and the previous three frames' actions. Please output them in order using \"_\" to connect different trajectory labels like \"11010211111111_10000211111111_…\""
            }
            conversation_gpt = {
                "from": "gpt",
                "value": f"{'_'.join(traj_label_txt)}"
            }
            episode_info = {
                "image": image_paths,
                "conversations": [conversation, conversation_gpt]
            }
            episode_info_list.append(episode_info)
        print(f"Processed episode {episode_idx}.")
        logging.info(f"Processed episode {episode_idx}.")
        return episode_info_list
    except Exception as e:
        logging.error(f"Error processing episode {episode_idx}: {e}", exc_info=True)
        return []

def generate_vlm_data_with_sampling(hdf5_dir, instructions_dir, save_dir, task_name, task_level, episode_count=10, segment_size=8, total_segments=32, total_frames=256):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    episode_info_list = []
    logging.info(f"Processing {episode_count} episodes from {hdf5_dir}...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_single_episode,
                episode_idx,
                hdf5_dir,
                instructions_dir,
                save_dir,
                task_name,
                task_level,
                segment_size,
                total_segments,
                total_frames
            )
            for episode_idx in range(episode_count)
        ]
        for f in concurrent.futures.as_completed(futures):
            episode_info_list.extend(f.result())

    json_data = episode_info_list
    json_save_path = os.path.join(save_dir, f"{task_name}_{task_level}_{episode_count}.json")
    with open(json_save_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    logging.info(f"Saved VLM data to {json_save_path}")

# 运行脚本
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate VLM data from RoboTwin hdf5 datasets.")
    parser.add_argument("--root-data-dir", type=str, default='/mnt/pfs/users/jiangnan.shao/code/RoboTwin/datasets/RoboTwin2.0/dataset',
                        help="Root directory containing task folders.")
    parser.add_argument("--save-dir", type=str, default='/mnt/pfs/users/jiangnan.shao/code/RoboTwin/datasets/qwen3_vl_data_all/',
                        help="Directory to save processed data and images.")
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated list of task folder names to process. If empty, process all tasks.")
    parser.add_argument("--task-file", type=str, default="",
                        help="Path to a text file containing one task name per line (overrides --tasks if provided).")
    parser.add_argument("--levels", type=str, default="clean",
                        help="Comma-separated levels to process among {clean,randomized}. Default: clean,randomized")
    parser.add_argument("--segment-size", type=int, default=8)
    parser.add_argument("--total-segments", type=int, default=8)
    parser.add_argument("--total-frames", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=8, help="Max threads for processing (ThreadPoolExecutor).")
    args = parser.parse_args()

    root_data_dir = args.root_data_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setup_logger(save_dir)  # 初始化日志

    # clean和randomized目录名修正
    level_dir_map = {
        "clean": "aloha-agilex_clean_50",
        "randomized": "aloha-agilex_randomized_500"
    }

    # 解析要处理的任务列表
    if args.task_file:
        if not os.path.isfile(args.task_file):
            logging.error(f"Task file not found: {args.task_file}")
            sys.exit(1)
        with open(args.task_file, "r") as tf:
            requested_tasks = [line.strip() for line in tf if line.strip()]
    elif args.tasks:
        requested_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        requested_tasks = []

    # 解析要处理的 levels
    requested_levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    valid_levels = {"clean", "randomized"}
    for lv in requested_levels:
        if lv not in valid_levels:
            logging.error(f"Invalid level '{lv}'. Valid options: clean, randomized")
            sys.exit(1)

    # 枚举任务目录（如果未指定，则遍历所有任务）
    if requested_tasks:
        task_list = []
        missing_tasks = []
        for t in requested_tasks:
            tpath = os.path.join(root_data_dir, t)
            if os.path.isdir(tpath):
                task_list.append(t)
            else:
                missing_tasks.append(t)
        if missing_tasks:
            logging.warning(f"Requested tasks not found under {root_data_dir}: {missing_tasks}")
        if not task_list:
            logging.error("No valid tasks to process. Exiting.")
            sys.exit(1)
    else:
        task_list = [d for d in os.listdir(root_data_dir) if os.path.isdir(os.path.join(root_data_dir, d))]
        if not task_list:
            logging.error(f"No tasks found under {root_data_dir}. Exiting.")
            sys.exit(1)

    logging.info(f"Tasks to process: {task_list}")
    logging.info(f"Levels to process: {requested_levels}")

    # 遍历任务与指定的 level（clean/randomized）
    for task_name in task_list:
        task_path = os.path.join(root_data_dir, task_name)
        if not os.path.isdir(task_path):
            logging.warning(f"Skip {task_name}: not a directory.")
            continue

        for task_level in requested_levels:
            level_key = task_level
            if level_key not in level_dir_map:
                logging.warning(f"Unknown level '{task_level}' for task {task_name}, skipping.")
                continue
            level_path = os.path.join(task_path, level_dir_map[level_key])
            hdf5_dir = os.path.join(level_path, "data")
            instructions_dir = os.path.join(level_path, "instructions")
            if not (os.path.isdir(hdf5_dir) and os.path.isdir(instructions_dir)):
                msg = f"Skip {task_name} {task_level}: missing data or instructions."
                print(msg)
                logging.warning(msg)
                continue

            # 自动检测 episode 文件数量
            hdf5_files = sorted(
                [f for f in os.listdir(hdf5_dir) if f.startswith("episode") and f.endswith(".hdf5")]
            )
            hdf5_files = hdf5_files[:2]
            if len(hdf5_files) == 0:
                msg = f"No episode .hdf5 files found in {hdf5_dir}, skip."
                print(msg)
                logging.warning(msg)
                continue
            episode_count = len(hdf5_files)
            msg = f"Processing {task_name} {task_level}: {episode_count} episodes."
            print(msg)
            logging.info(msg)

            # 调用原有的数据生成函数（并传递线程数限制）
            # 注意：generate_vlm_data_with_sampling 内部使用 ThreadPoolExecutor()，要改最大并发请修改函数定义或全局参数。
            generate_vlm_data_with_sampling(
                hdf5_dir,
                instructions_dir,
                save_dir,
                task_name,
                task_level,
                episode_count=episode_count,
                segment_size=args.segment_size,
                total_segments=args.total_segments,
                total_frames=args.total_frames,
            )
