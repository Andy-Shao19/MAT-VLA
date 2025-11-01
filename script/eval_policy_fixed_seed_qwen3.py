from transformers import AutoModelForImageTextToText, AutoProcessor
import cv2
import transforms3d as t3d
import re

import json
import sys
import os
import subprocess
import time

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
import torch

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _parse_qwen_labels(text: str) -> np.ndarray:
    """
    把 Qwen3-VL 返回的 '110102..._100002...' 解析成 (8, 14) np.ndarray(int64)
    """
    parts = [p for p in text.strip().split("_") if p]
    label_mat = []
    for p in parts[:8]:  # 保险起见只取 8 段
        label_mat.append([int(c) for c in p.strip()[:14]])  # 每段 14 位 0/1/2
    # 若不足 8 行，用 0 补齐
    while len(label_mat) < 8:
        label_mat.append([0]*14)
    return np.asarray(label_mat, dtype=np.int64)

def validate_traj_label_text(
    text: str,
    expected_segments: int = 8,
    segment_length: int = 14,
) -> bool:
    """
    校验 Qwen3-VL 生成的 trajectory-label 字符串是否合法。

    Parameters
    ----------
    text : str
        待校验字符串，如 '11010211111111_10000211111111_……'
    expected_segments : int, optional
        期望的最小段数，默认 8。
    segment_length : int, optional
        每段应有的字符长度，默认 14。

    Returns
    -------
    (bool)
    """
    if text is None:
        return False, "text is None"

    s = text.strip()
    if not s:
        return False

    # 1. 不能以 '_' 开头或结尾，且不能有连续 '__'
    if s.startswith("_") or s.endswith("_") or "__" in s:
        return False

    segments = s.split("_")
    if len(segments) != expected_segments:
        return False

    # 2. 校验每段长度 & 字符合法性
    pattern = re.compile(f"^[0-2]{{{segment_length}}}$")
    for idx, seg in enumerate(segments[:expected_segments], 1):
        if not pattern.match(seg):
            if len(seg) != segment_length:
                return (
                    False
                )
            return False

    return True

def predict_traj_labels(imgs: list, poses: list, instr: str,
                        model, processor) -> np.ndarray:
    """
    imgs  : list[PIL.Image]，长度 3，按时间排序 (t-2, t-1, t)
    poses : list[np.ndarray]，长度 3，每个 shape=(14,)
    instr : 当前 task 指令
    返回  : (8, 14) np.ndarray(int64)
    """
    # 构造系统 prompt ----------------------------------
    # endpose_txt = str(np.vstack(poses).tolist())
    endpose_txt = np.array2string(np.vstack(poses), separator=', ', formatter={'float_kind': lambda x: f"{x:.6f}"},).replace('\n', '')
    
    user_prompt = (
        f"You are a Aloha-AgileX robot using end-effector control. The instruction is \"{instr}\".<image>\nThis is the left camera image of the current frame.\n<image>\nThis is the head camera image of the current frame.\n<image>\nThis is the right camera image of the current frame.\n"
        f"The format of endpose is [x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l, x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r] and the previous three (including current) frames' endposes are: {endpose_txt}.\n"
        f"We define a trajectory label as follows: every 8 frames, take the endpose differences. Each component is mapped to 0 if below the threshold, 1 if within [-threshold, threshold], and 2 if above the threshold. The thresholds for xyz are 0.02, for rpy are 0.05, and for gripper are 0.01.\n"
        f"Please predict the next 8 trajectory labels based on the images, instruction and the previous three frames' endposes. Please output them in order using \"_\" to connect different trajectory labels like \"11010211111111_10000211111111_…\""
    )
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": imgs[0]},
            {"type": "image", "image": imgs[1]},
            {"type": "image", "image": imgs[2]},
            {"type": "text",  "text": user_prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=119)
    gen_trimmed = gen_ids[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(
        gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # if text != '11010211111111_10000211111111_10000211111111_10000211111111_10000211111111_10000211111111_10000211111111_10000211111111':
    #     text = '11010211111111_10000211111111_10000211111111_10000211111111_10000211111111_10000211111111_10000211111111_10000211111111'
    while not validate_traj_label_text(text):
        text = processor.batch_decode(
            gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(text)

    return _parse_qwen_labels(text)

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
    
    # 提取欧拉角（固定轴顺序：roll(x), pitch(y), yaw(z)）
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


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    seed_file = usr_args.get("seed_file", None)  # 添加种子文件路径参数
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"/shared_disk/users/jiangnan.shao/eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    # 读取种子文件
    if seed_file and os.path.exists(seed_file):
        with open(seed_file, "r") as f:
            content = f.read().strip()
            # 支持空格分隔或换行分隔
            seeds = [int(s) for s in content.split() if s.strip()]
        seeds = seeds[:50]  # 只取前50个
        print(f"\033[96mLoaded {len(seeds)} seeds from {seed_file}\033[0m")
        print(f"\033[96mSeeds: {seeds}\033[0m")
    else:
        # 如果没有指定文件或文件不存在,使用默认方式
        seed = usr_args.get("seed", 0)
        st_seed = 100000 * (1 + seed)
        seeds = list(range(st_seed, st_seed + 50))
        print(f"\033[93mUsing default seed range starting from {st_seed}\033[0m")
        
    suc_nums = []
    test_num = len(seeds)  # 测试数量改为种子数量
    topk = 1

    model = get_model(usr_args)
    suc_num = eval_policy(task_name,
                         TASK_ENV,
                         args,
                         model,
                         seeds,  # 传入种子列表
                         test_num=test_num,
                         video_size=video_size,
                         instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Seeds used: {seeds}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                seeds,  # 改为接收种子列表
                test_num=50,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    print(f"\033[34mTesting {test_num} seeds\033[0m")

    # 加载Qwen3-VL=========
    QWEN_PATH = "/mnt/data-1/data/shaojiangnan/code/MAT-VLA/Qwen3-VL-2B-Instruct"

    qwen_model = AutoModelForImageTextToText.from_pretrained(
        QWEN_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='auto',
    )

    qwen_processor = AutoProcessor.from_pretrained(QWEN_PATH)
    #=======================
    
    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    
    
    

    for seed_idx, now_seed in enumerate(seeds):
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                print(f"\033[91mUnstable seed {now_seed}, skipping...\033[0m")
                continue
            except Exception as e:
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                print(f"\033[91mError with seed {now_seed}, skipping...\033[0m")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            suc_test_seed_list.append(now_seed)
        else:
            args["render_freq"] = render_freq
            print(f"\033[91mExpert check failed for seed {now_seed}, skipping...\033[0m")
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        # if TASK_ENV.eval_video_path is not None:
        #     ffmpeg = subprocess.Popen(
        #         [
        #             "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        #             "-pixel_format", "rgb24", "-video_size", video_size,
        #             "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
        #             "-vcodec", "libx264", "-crf", "23",
        #             f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
        #         ],
        #         stdin=subprocess.PIPE,
        #     )
        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "mpeg2video",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        
        dual_endpose_list = []
        observation = TASK_ENV.get_obs()
        dual_endpose = observation['endpose']
        for i in range(3):
            dual_endpose_list.append(dual_endpose)
        print('endpose:', dual_endpose_list[-1])
        
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            
            # 把dual_endpose_list转到head camera坐标系下
            head_cam_extrinsic = observation["observation"]["head_camera"]["extrinsic_cv"]
            endpose_3 = []
            for i in range(len(dual_endpose_list)):
                left_6d_cam = world_to_camera_transform(dual_endpose_list[i][:6], head_cam_extrinsic)
                right_6d_cam = world_to_camera_transform(dual_endpose_list[i][7:-1], head_cam_extrinsic)
                dual_endpose_cam = np.concatenate([left_6d_cam, [dual_endpose_list[i][6]], right_6d_cam, [dual_endpose_list[i][-1]]], axis=0)
                endpose_3.append(dual_endpose_cam)

            # 保存测试rgb图片
            # for cam_name in ["left_camera", "head_camera", "right_camera"]:
            #     raw_img = observation["observation"][cam_name]["rgb"]
            #     img_bgr = to_bgr_image(raw_img)
            #     image_path_to_save = f"/mnt/data-1/data/shaojiangnan/code/MAT-VLA/test_img"
            #     os.makedirs(image_path_to_save, exist_ok=True)
            #     image_path = os.path.join(image_path_to_save, f"{cam_name}.jpg")
            #     ok = cv2.imwrite(image_path, img_bgr)
            
            images_list= [
                (observation["observation"]["left_camera"]["rgb"]),
                (observation["observation"]["head_camera"]["rgb"]),
                (observation["observation"]["right_camera"]["rgb"]),
            ]
            traj_label = predict_traj_labels(images_list, endpose_3, instruction, qwen_model, qwen_processor)
            traj_label = torch.from_numpy(traj_label).unsqueeze(0)
            
            start_time = time.time()
            dual_endpose_list = eval_func(TASK_ENV, model, observation, traj_label)
            print('endpose:', dual_endpose_list[-1])
            end_time = time.time()
            print("一个chunk执行时间: ", end_time - start_time)
            if TASK_ENV.eval_success:
                succ = True
                break

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((seed_idx + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Progress: \033[96m{seed_idx + 1}/{test_num}\033[0m, Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )

    return TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
