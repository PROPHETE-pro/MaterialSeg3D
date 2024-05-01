#该函数的作用是将单通道的材质label图转化为46通道的一组图片，用于进行语义分割的预测

import os
import numpy as np
from PIL import Image

# 46种材质的ID
dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
    24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
    50, 51, 52, 53, 56]

# 输入和输出路径
input_folder = '/data/zeyu_li/DMS/DMS_v1/test'
output_folder = '/data/zeyu_li/DMS/DMS_v1/labels_processed'

def convert_labels(input_path, output_path):
    # 打开原始标签图像
    img = Image.open(input_path)
    img_array = np.array(img)

    # 创建一个空的46通道的one-hot编码图像
    h, w = img_array.shape
    one_hot_img = np.zeros((h, w, len(dms46)), dtype=np.uint8)

    # 对于每个材质，将标签映射为对应的one-hot通道
    for i, material_id in enumerate(dms46):
        one_hot_img[:, :, i] = (img_array == material_id).astype(np.uint8)

    # 保存单个one-hot编码的标签图像
    output_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_folder, output_filename)
    np.save(output_path, one_hot_img)

    # 为每个通道分别保存图像
    for i, material_id in enumerate(dms46):
        channel_img = one_hot_img[:, :, i]
        channel_output_path = os.path.join(output_folder, f"{output_filename}_channel_{material_id}.png")
        Image.fromarray(channel_img * 255).convert("L").save(channel_output_path)

    return one_hot_img

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 处理输入文件夹下的所有图像
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        convert_labels(input_path, output_folder)

print("All labels processed and saved in the output folder.")





# #遍历图像文件夹，并计算每个图像中唯一标签的数量，然后复制包含超过20种唯一标签的图像到新的目录。
# import os
# import cv2
# import numpy as np
# import shutil

# # 输入和输出路径
# input_folder = '/data/zeyu_li/DMS/DMS_v1/labels'
# output_folder = '/data/zeyu_li/DMS/DMS_v1/long_label'

# # 创建输出文件夹
# os.makedirs(output_folder, exist_ok=True)

# def count_unique_labels(image_path):
#     # 读取图像
#     img = cv2.imread(image_path)

#     # 计算图像中唯一标签的数量
#     unique_labels = np.unique(img[:, :, 2])
#     return len(unique_labels)

# # 遍历图像文件夹
# for root, dirs, files in os.walk(input_folder):
#     for file in files:
#         if file.endswith(".png"):
#             image_path = os.path.join(root, file)
#             num_unique_labels = count_unique_labels(image_path)

#             # 如果图像中唯一标签的数量大于20，复制到输出文件夹
#             if num_unique_labels > 20:
#                 shutil.copy(image_path, os.path.join(output_folder, file))

# print("Images with more than 20 unique labels copied to the output folder.")



