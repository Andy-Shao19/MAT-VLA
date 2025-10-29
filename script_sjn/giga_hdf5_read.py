import os
import h5py
import numpy as np
import cv2

def read_video_and_depth_from_hdf5(file_path, cam_name):
    with h5py.File(file_path, 'r') as f:
        video = f['observations']['images'][cam_name][:]
        depth = f['observations']['images_depth'][cam_name][:]
    return video, depth

def read_video_from_hdf5(file_path, cam_name):
    with h5py.File(file_path, 'r') as f:
        video = f['observations']['images'][cam_name][:]
    return video

def read_action_from_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            actions = f['observations']['qpos'][:]
    except:
        return None
    return actions

src_path = '/mnt/pfs/users/jiangnan.shao/code/RoboTwin/datasets/Agilex_robot'
all_data_list = [
    'stack_bowls_two_clean',
]

if __name__ == '__main__':
    for data_name in all_data_list:
        data_path = os.path.join(src_path, data_name)
        hdf5_list = [f for f in os.listdir(data_path) if f.endswith('.hdf5')]
        for hdf5_name in hdf5_list:
            hdf5_path = os.path.join(data_path, hdf5_name)
            actions = read_action_from_hdf5(hdf5_path)
            videos = read_video_from_hdf5(hdf5_path, 'cam_high')  # cam_name: 'cam_high', 'cam_left_wrist', 'cam_right_wrist'

            mp4_save_path = os.path.join(src_path, data_name, "videos", f"{hdf5_name.split('.')[0]}.mp4")
            os.makedirs(os.path.dirname(mp4_save_path), exist_ok=True)
            
            height, width = videos[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(mp4_save_path, fourcc, 20.0, (width, height))
            for frame in videos:
                if frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            out.release()
            
            print(f"Saved video to {mp4_save_path}")
