#这个py文件就是集成视角转换和材质预测的主文件
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# This example shows how to predict materials.
#

import argparse
import torchvision.transforms as TTR
import os
import glob
import random
import json
import cv2
import numpy as np
import torch
import math
from PIL import Image


"""
DMS部分的命令行是
$ python inference.py --jit_path /data/zeyu_li/DMS/DMS46_v1.pt --image_folder /data/zeyu_li/DMS/test/input --output_folder /data/zeyu_li/DMS/test/out
"""
random.seed(112)
#选择列举的46类材料种类
dms46 = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
    24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
    50, 51, 52, 53, 56, ]
t = json.load(open(os.path.expanduser('/data/zeyu_li/DMS/taxonomy.json'), 'rb')) #加载颜色映射数据
                                                
srgb_colormap = [      
    t['srgb_colormap'][i] for i in range(len(t['srgb_colormap'])) if i in dms46   #为46个种类对应颜色映射
]
srgb_colormap = np.array(srgb_colormap, dtype=np.uint8)

#颜色映射函数，接受一个类标签的掩码（一个二维数组），并使用颜色映射将其转换为可视化颜色。
def apply_color(label_mask):       
    # translate labels to visualization colors
    vis = np.take(srgb_colormap, label_mask, axis=0)
    return vis[..., ::-1]


def get_material(image_f): 
    args = argparse.Namespace(
        jit_path='/data/zeyu_li/DMS/DMS46_v1.pt',
        image_folder=image_f,
        output_folder='/data/zeyu_li/Text2Tex/outputs/car/test'
    )
    is_cuda = torch.cuda.is_available()  
    model = torch.jit.load(args.jit_path)    #加载预训练的DMS46_v1.pt，用于预测像素的 46 种材料
    if is_cuda:
        model = model.cuda() #如果 GPU 可用，模型将被移到 GPU 上进行后续计算。
    
    # images_list = glob.glob(f'{args.image_folder}/*') #通过 glob 模块获取指定文件夹 args.image_folder 下的所有图像文件路径，并将它们存储在 images_list 列表中。
    # 如果提供的是单个图像文件路径而不是文件夹
    if os.path.isfile(args.image_folder):
        images_list = [args.image_folder]  # 将单个文件路径放入列表中
    else:
        # 如果提供的是文件夹，则获取文件夹下的所有图像文件路径
        images_list = glob.glob(f'{args.image_folder}/*')
    
    #对图像进行预处理，以下变量定义了图像预处理所需的参数，包括图像像素值的缩放因子 value_scale，均值 mean，和标准差 std
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    os.makedirs(args.output_folder, exist_ok=True) #根据命令行参数 args.output_folder 创建用于存储输出结果的文件夹，如果该文件夹已经存在，则不会重复创建
    #处理图像列表中的每个图像，包括了图像的加载、尺寸调整、归一化、模型推断、结果可视化以及结果保存
    for image_path in images_list:
        print(image_path)
        #读取和调整图像
        img = cv2.imread(image_path, cv2.IMREAD_COLOR) #使用OpenCV (cv2)库读取图像，IMREAD_COLOR 标志表示以彩色模式读取
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #将读取的图像从BGR颜色空间转换为RGB颜色空间，以确保颜色通道的正确性
        #调整图像的尺寸，将图像的高度和宽度缩放到一个新的目标尺寸 new_dim，以确保输入尺寸适合模型
        new_dim = 768    
        h, w = img.shape[0:2]
        scale_x = float(new_dim) / float(h)
        scale_y = float(new_dim) / float(w)
        scale = min(scale_x, scale_y)
        new_h = math.ceil(scale * h)
        new_w = math.ceil(scale * w)
    
        img = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS) ##使用PIL库的 Image.fromarray 来进行图像缩放，采用 Image.LANCZOS 采样方法以获得更好的质量
        img = np.array(img) #将PIL图像转换为NumPy数组，以便后续对图像进行处理和归一化，然后传递给深度学习模型进行推断
        #归一化和GPU移动
        image = torch.from_numpy(img.transpose((2, 0, 1))).float() #图像数据转换为PyTorch张量，通道顺序被重新排列为(C, H, W)
        image = TTR.Normalize(mean, std)(image)#图像被归一化，使用了定义的均值 mean 和标准差 std
        if is_cuda:
            image = image.cuda() #图像移动到GPU上
        image = image.unsqueeze(0) #形状将变为(1, C, H, W)，其中第一个维度是批次大小。这样做是为了适应深度学习模型通常要求的输入数据形状
        
        #使用模型进行推断
        #输入：image（H,W,C），输出：prediction（H,W）
        with torch.no_grad(): #使用模型进行图像推断，将输入图像传递给模型并获取输出结果,with torch.no_grad() 确保在此过程中不进行梯度计算，以加快推断速度
            prediction = model(image)[0].data.cpu()[0, 0].numpy() #输出是一个单通道的图像，其中每个像素的值表示该像素属于预测的哪个类别，将其转化为NumPy数组
        print("prediction.shape:",prediction.shape)
        print(prediction)
        #结果可视化,将原始图像（original_image）和经过颜色映射处理的模型预测结果（predicted_colored）准备好，以便叠加在一起
        original_image = img[..., ::-1]
        predicted_colored = apply_color(prediction)
        print("predicted_colored.shape:",predicted_colored.shape)
        print(predicted_colored)
        #结果保存
        #始图像和预测结果在水平方向上叠加，然后使用OpenCV的 cv2.imwrite 将结果保存为图像文件
        # stacked_img = np.concatenate(
        #     (np.uint8(original_image), predicted_colored), axis=1
        # )
        #图像文件的文件名基于输入图像的文件名，并保存在指定的输出文件夹中
        # cv2.imwrite(
        #     f'{args.output_folder}/{os.path.splitext(os.path.basename(image_path))[0]}.png',
        #     stacked_img,
        # )
        cv2.imwrite(
            f'{args.output_folder}/{os.path.splitext(os.path.basename(image_path))[0]}.png',
            predicted_colored,
        )


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()   #创建一个argparse解析器对象，用于定义和解析命令行参数
    # # 使用add_argument方法定义命令行参数的选项。在这个代码中，定义了三个选项：--jit_path、--image_folder和--output_folder。每个选项都具有不同的类型、默认值和帮助文本
    # parser.add_argument(  
    #     '--jit_path',
    #     type=str,
    #     default='',
    #     help='path to the pretrained model',
    # )
    # parser.add_argument(
    #     '--image_folder',
    #     type=str,
    #     default='',
    #     help='overwrite the data_path for local runs',
    # )
    # parser.add_argument(
    #     '--output_folder',
    #     type=str,
    #     default='',
    #     help='overwrite the data_path for local runs',
    # )
    # args = parser.parse_args()   #调用parse_args() 方法来解析命令行参数，并将结果存储在args变量中。args变量将包含传递给脚本的命令行参数的值

    get_material('/data/zeyu_li/DMS/DMS_v1/images/train/22519.jpg')

