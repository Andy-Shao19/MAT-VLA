import numpy as np
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

def camera_to_world_transform(camera_pose, camera_matrix):
    """
    将相机坐标系下的位姿转换回世界坐标系
    
    参数:
        camera_pose: [x, y, z, roll, pitch, yaw] 在相机坐标系下
        camera_matrix: 相机在世界坐标系下的4x4变换矩阵
    
    返回:
        world_pose: 在世界坐标系下的位姿 [x, y, z, roll, pitch, yaw]
    """
    # 提取位姿部分
    if len(camera_pose) == 7:
        camera_pose_6d = camera_pose[:6]
        gripper = camera_pose[6]
    else:
        camera_pose_6d = camera_pose
        gripper = None
    
    # 相机坐标系下的变换矩阵
    T_camera_end = xyzrpy_to_matrix(*camera_pose_6d)
    
    # 计算世界坐标系下的变换矩阵: T_world_end = T_world_camera * T_camera_end
    T_world_end = camera_matrix @ T_camera_end
    
    # 转换回xyzrpy
    world_pose_6d = matrix_to_xyzrpy(T_world_end)
    
    # 如果需要，重新添加gripper值
    if gripper is not None:
        return np.concatenate([world_pose_6d, [gripper]])
    else:
        return world_pose_6d


head_camera_matrix = np.array([
    [1.0, 0.0, 0.0, 0.03200001],
    [0.0, -0.8, -0.6, 0.45],
    [0.0, 0.6, -0.8, 1.35],
    [0.0, 0.0, 0.0, 1.0]
])

# 假设的left_endpose（世界坐标系下）
left_endpose_world = [-2.97923714e-01, -3.13803881e-01,  9.41998601e-01,  7.00000708e-01,
  -7.09328127e-06, -3.79383585e-06,  8.00000000e-01]  # [x,y,z,roll,pitch,yaw,gripper]

print("原始世界坐标系位姿:", left_endpose_world)

# 转换到head_camera坐标系
left_endpose_camera = world_to_camera_transform(left_endpose_world, head_camera_matrix)
print("相机坐标系位姿:", left_endpose_camera)

# 再转换回世界坐标系
left_endpose_world_back = camera_to_world_transform(left_endpose_camera, head_camera_matrix)
print("转换回世界坐标系:", left_endpose_world_back)

# 验证转换的准确性
print("转换误差:", np.array(left_endpose_world) - np.array(left_endpose_world_back))