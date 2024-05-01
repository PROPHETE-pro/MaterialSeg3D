import json
import numpy as np

# JSON文件路径
json_file_path = '/root/data2/zeyu_li/GET3D/GET3D-master/render_shapenet_data/camera_angle/transforms.json'

# 读取JSON文件
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 提取"transform_matrix"
transform_matrices = [frame['transform_matrix'] for frame in data['frames']]

# 将列表转换为NumPy数组
B_Blender = np.array(transform_matrices)

# 保存为.npy文件
np.save('/root/data2/zeyu_li/GET3D/GET3D-master/render_shapenet_data/camera_angle/original_40.npy', B_Blender)

# import numpy as np
# import json 
# from typing import Tuple

# def P2B(R: np.ndarray, T: np.ndarray) -> np.ndarray:
#     P2B_R1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
#     P2B_R2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
#     P2B_T = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
#     vec4w = np.array([[0, 0, 0, 1]], dtype=np.float64)

#     B = np.zeros((30, 4, 4))  # 创建一个适应30个视角的空矩阵
# # 从文件中载入保存的 R 和 T 数组
# R_data = np.load('/data/zeyu_li/data/test/RT/R_data.npy')
# T_data = np.load('/data/zeyu_li/data/test/RT/T_data.npy')
# print("R_data.shape:",R_data.shape)
# print("T_data.shape:",T_data.shape)

# B = np.load('/data/zeyu_li/data/test/blender/data/B_Blender.npy')
# print("B.shape:",B.shape)
# # 假设 B_all 的形状为 (30, 4, 4)
# print("0----------------",B[0])
# print("1----------------",B[1])

# B_30 = np.load('/data/zeyu_li/data/test/blender/data/all_B_array.npy')
# import pdb;pdb.set_trace()
# # # 假设要设置第一个相机的位置
# # index = 0
# # print(R_data[index][0].shape)
# # print(T_data[index][0].shape)
# # R = R_data[index][0]
# # T = np.expand_dims(T_data[index][0], axis=1)
# # for i in range(R_data.shape[0]):
# #     R_i = R_data[i][0].flatten()
# #     T_i = T_data[i][0].flatten()

# #     # 打印形状以确保它们匹配
# #     print(f"Shapes - R_i: {R_i.shape}, T_i: {T_i.shape}")


