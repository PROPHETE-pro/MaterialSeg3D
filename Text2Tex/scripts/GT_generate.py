#利用pytorch3d渲染3D物体的2d渲染图，用于构建语义分割网络的dataset
# common utils
import os
import argparse
import time
# pytorch3d
from pytorch3d.renderer import TexturesUV
# torch
import torch
from torchvision import transforms
# numpy
import numpy as np
# image
from PIL import Image

# customized
import sys
sys.path.append("/data2/zeyu_li/Text2Tex")
from lib.projection_helper import (
    render_one_view_and_build_masks_2,
)
from lib.camera_helper import init_viewpoints
from lib.mesh_helper import (
    init_mesh,
    init_mesh_2
)
import trimesh
# 指定GPU来渲染
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:2")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

def modify_image(init_texture, init_image):
       # 转换图像为 NumPy 数组
    init_texture_np = np.array(init_texture)
    init_image_np = np.array(init_image)

    # 将 [255, 255, 255] 的像素值替换为 [0, 0, 0]
    mask = np.all(init_image_np == [255, 255, 255], axis=2)
    init_image_np[mask] = [0, 0, 0]

    # 获取所有唯一的像素值列表
    unique_pixels = np.unique(init_texture_np.reshape(-1, init_texture_np.shape[2]), axis=0)

    # 逐像素查找最接近的颜色值进行替换
    for i in range(init_image_np.shape[0]):
        for j in range(init_image_np.shape[1]):
            pixel_value = init_image_np[i, j]

            # 找到最接近的颜色值进行替换
            if not np.array_equal(pixel_value, [0, 0, 0]):
                nearest_val = min(unique_pixels, key=lambda x: np.linalg.norm(pixel_value - x))
                init_image_np[i, j] = nearest_val

    # 将修改后的 NumPy 数组转换为 PIL 图像
    modified_init_image = Image.fromarray(init_image_np.astype(np.uint8))

    return modified_init_image


def get_data(input_dir,output_dir,choose = 0):
    # 获取 obj_name
    obj_name = os.path.basename(input_dir)
    textureUV_path =  os.path.join(input_dir, 'Image_0.png')
    # 拼接 obj_file 的路径
    obj_file = os.path.join( obj_name + '.obj')
    dist = 1
    elev = 0
    num_viewpoints = 36             #可以选1/2/4/6/12/20/36/68。相机参数都是写好固定的
    viewpoint_mode = 'predefined'
    fragment_k = 1
    #控制生成的图像大小
    image_size = 1024
    use_shapenet = True
    use_objaverse = True

    os.makedirs(output_dir, exist_ok=True)
    # print("=> OUTPUT_DIR:", output_dir)
    B_all = np.load('/data2/zeyu_li/data/test/blender/data/B_40.npy')
    #初始化各类用于生成和更新3D模型纹理的资源
    # mesh, _, faces, aux, principle_directions, mesh_center, mesh_scale = init_mesh(os.path.join(input_dir, obj_file), DEVICE)
    #经过预处理的mesh无需再normal调整尺寸
    mesh, _, faces, aux, principle_directions, mesh_center, mesh_scale = init_mesh_2(os.path.join(input_dir, obj_file), DEVICE)
    #d导入纹理UV
    init_texture= Image.open(textureUV_path).convert("RGB")
    (
        dist_list, 
        elev_list, 
        azim_list, 
        sector_list,
        view_punishments
    ) = init_viewpoints(viewpoint_mode, num_viewpoints, dist, elev, principle_directions, 
                            use_principle=True, 
                            use_shapenet= use_shapenet,
                            use_objaverse=use_objaverse)

    # NOTE no update / refinement
    if choose == 0:     #如果是生成Image
        #一级目录，/Data/Image或者/Data/GT
        generate_dir = os.path.join(output_dir, "Image")
    else:   
        #一级目录，/Data/Image或者/Data/GT
        generate_dir = os.path.join(output_dir, "GT")
    os.makedirs(generate_dir, exist_ok=True)
    #二级目录，加上资产名字，生成的图片都在这个子文件中
    inpainted_image_dir = os.path.join(generate_dir, obj_name)
    os.makedirs(inpainted_image_dir, exist_ok=True)


    # 使用视点信息构建相似性纹理缓存
    NUM_PRINCIPLE = 40 #选择36个点中前30个初始生成视点，位置都是固定的
    pre_dist_list = dist_list[:NUM_PRINCIPLE]
    pre_elev_list = elev_list[:NUM_PRINCIPLE]
    pre_azim_list = azim_list[:NUM_PRINCIPLE]
    pre_sector_list = sector_list[:NUM_PRINCIPLE]

    ###############开始生成纹理的过程：生成图像 + 创建纹理UV ########################
    # start generation
    print("=> start generating...")
    # 初始化一个空数组来存储结果
    # all_B = []
    for view_idx in range(NUM_PRINCIPLE):  #对每个视点进行处理
        # print("=> processing view {}...".format(view_idx))
        # 通过 pre_dist_list, pre_elev_list, pre_azim_list, pre_sector_list获取当前视点的位置信息
        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx] 

        (
            init_image, 
            init_images_tensor
        ) = render_one_view_and_build_masks_2(dist, elev, azim, 
            mesh, 
            image_size, fragment_k,
            DEVICE,B_all[view_idx]
        )
        # all_B.append(B)
        if choose == 0:
            #生成Image时用下面两行
            generate_image_before = init_image      #不用用后处理，忽略插值问题，Image处理时使用
            generate_image_before.save(os.path.join(inpainted_image_dir, obj_name + "_{}.png".format(view_idx)))
        else:
            #调用后处理函数消除插值问题（GT图时使用下面两行）
            generate_image_before = modify_image(init_texture,init_image)
            # import pdb;pdb.set_trace()
            generate_image_before = generate_image_before.convert("L")  # 转换为单通道图像，便于后续网络训练
            generate_image_before.save(os.path.join(inpainted_image_dir, obj_name + "_{}_222.png".format(view_idx)))
        # print("=> generated for view {}".format(view_idx))
    # # 将所有结果堆叠成一个数组
    # all_B = np.stack(all_B)

    # # 保存数组到 .npy 文件
    # np.save('/data/zeyu_li/data/test/blender/data/all_B_array.npy', all_B)
        
#批量处理资产
def process_folders(input_parent_dir, output_parent_dir,choose):
    #批量资产的地址
    choose = choose
    label_dir =   input_parent_dir
    label_folders = os.listdir(label_dir)
    output_dir = output_parent_dir
    number = 1
    for folder in label_folders:
        input_dir = os.path.join(label_dir, folder)
        start_time = time.time()
        print(f"开始处理{input_dir}")
        get_data(input_dir, output_dir, choose)
        print(f"已经处理了{number}/{len(label_folders)}的资产")
        print(f"=> Processing {input_dir}: {time.time() - start_time} s")
        number += 1


if __name__ == "__main__":
    start_time = time.time()
    choose = 1     #选0是构造Image；选1是构造GT
    input_dir = '/data/zeyu_li/data/UE/table_sofa_cabinet/normal_label'  
    output_dir = '/data/zeyu_li/data/UE/table_sofa_cabinet'
    process_folders(input_dir, output_dir, choose)
    # get_data(input_dir, output_dir,choose)
    print(f"=>total time: {time.time() - start_time} s")

