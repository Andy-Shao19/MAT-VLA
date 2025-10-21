import os
import h5py
import numpy as np
import cv2

hdf5_file = "/mnt/pfs/users/jiangnan.shao/code/RoboTwin/policy/RDT/processed_data_traj/stack_bowls_two-aloha-agilex_clean-50/episode_0/episode_0.hdf5"

if __name__ == '__main__':
    with h5py.File(hdf5_file, 'r') as f:
        print(f.keys())
        print(f['action'].shape)
        print(f['dual_endpose'].shape)