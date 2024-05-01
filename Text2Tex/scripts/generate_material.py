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


#输入图片或文件夹，返回对应的材质预测图
def get_material(image_f):  
    args = argparse.Namespace(
        jit_path='/data/zeyu_li/DMS/DMS46_v1.pt',
        image_folder=image_f,
        
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

    # os.makedirs(args.output_folder, exist_ok=True) #根据命令行参数 args.output_folder 创建用于存储输出结果的文件夹，如果该文件夹已经存在，则不会重复创建
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
        import pdb;pdb.set_trace()
        with torch.no_grad(): #使用模型进行图像推断，将输入图像传递给模型并获取输出结果,with torch.no_grad() 确保在此过程中不进行梯度计算，以加快推断速度
            prediction = model(image)[0].data.cpu()[0, 0].numpy() #输出是一个单通道的图像，其中每个像素的值表示该像素属于预测的哪个类别，将其转化为NumPy数组

        #结果可视化,将原始图像（original_image）和经过颜色映射处理的模型预测结果（predicted_colored）准备好，以便叠加在一起
        original_image = img[..., ::-1]
        predicted_colored = apply_color(prediction) #将单通道的png图像转化为彩色映射的图像
        #结果保存
        #始图像和预测结果在水平方向上叠加，然后使用OpenCV的 cv2.imwrite 将结果保存为图像文件
        # stacked_img = np.concatenate(
        #     (np.uint8(original_image), predicted_colored), axis=1
        # )
        #图像文件的文件名基于输入图像的文件名，并保存在指定的输出文件夹中
        # cv2.imwrite(
        #     f'{args.output_folder}/{os.path.splitext(os.path.basename(image_path))[0]}.png',
        #     predicted_colored,
        # )

        return predicted_colored



"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
以下部分是Text2Tex的视角转换的texturing贴图生成的代码
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

"""




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
sys.path.append(".")

from lib.mesh_helper import (
    init_mesh,
    apply_offsets_to_mesh,
    adjust_uv_map
)
from lib.render_helper import render
from lib.io_helper import (
    save_backproject_obj,
    save_args,
    save_viewpoints
)
from lib.vis_helper import (
    visualize_outputs, 
    visualize_principle_viewpoints, 
    visualize_refinement_viewpoints
)
from lib.diffusion_helper import (
    get_controlnet_depth,
    get_inpainting,
    apply_controlnet_depth,
    apply_inpainting_postprocess
)
from lib.projection_helper import (
    backproject_from_image,
    render_one_view_and_build_masks,
    select_viewpoint,
    build_similarity_texture_cache_for_all_views
)
from lib.camera_helper import init_viewpoints

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:2")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()


"""
    Use Diffusion Models conditioned on depth input to back-project textures on 3D mesh.
    在3D网格上使用基于深度输入的扩散模型来反向投影纹理。
    The inputs should be constructed as follows(输入应构造如下):
        - <input_dir>/
            |- <obj_file> # name of the input OBJ file
    
    此脚本的输出将存储在' outputs/ '下，并使用配置参数作为文件夹名称。具体来说，应该有以下文件在这样的文件夹:
        - outputs/
            |- <configs>/                       # configurations of the run
                |- generate/                    # assets generated in generation stage
                    |- depth/                   # depth map
                    |- inpainted/               # images generated by diffusion models
                    |- intermediate/            # renderings of textured mesh after each step
                    |- mask/                    # generation mask
                    |- mesh/                    # textured mesh
                    |- normal/                  # normal map
                    |- rendering/               # input renderings
                    |- similarity/              # simiarity map
                |- update/                      # assets generated in refinement stage
                    |- ...                      # the structure is the same as generate/
                |- args.json                    # all arguments for the run
                |- viewpoints.json              # all viewpoints
                |- principle_viewpoints.png     # principle viewpoints
                |- refinement_viewpoints.png    # refinement viewpoints

"""

def init_args():
    print("=> initializing input arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--obj_name", type=str, required=True)
    parser.add_argument("--obj_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--a_prompt", type=str, default="best quality, high quality, extremely detailed, good geometry")
    parser.add_argument("--n_prompt", type=str, default="deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke")
    parser.add_argument("--new_strength", type=float, default=1)
    parser.add_argument("--update_strength", type=float, default=0.5)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=10)
    parser.add_argument("--output_scale", type=float, default=1)
    parser.add_argument("--view_threshold", type=float, default=0.1)
    parser.add_argument("--num_viewpoints", type=int, default=8)
    parser.add_argument("--viewpoint_mode", type=str, default="predefined", choices=["predefined", "hemisphere"])
    parser.add_argument("--update_steps", type=int, default=8)
    parser.add_argument("--update_mode", type=str, default="heuristic", choices=["sequential", "heuristic", "random"])
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_patch", action="store_true", help="apply repaint during refinement to patch up the missing regions")
    parser.add_argument("--use_multiple_objects", action="store_true", help="operate on multiple objects")
    parser.add_argument("--use_principle", action="store_true", help="poperate on multiple objects")
    parser.add_argument("--use_shapenet", action="store_true", help="operate on ShapeNet objects")
    parser.add_argument("--use_objaverse", action="store_true", help="operate on Objaverse objects")
    parser.add_argument("--use_unnormalized", action="store_true", help="save unnormalized mesh")

    parser.add_argument("--add_view_to_prompt", action="store_true", help="add view information to the prompt")
    parser.add_argument("--post_process", action="store_true", help="post processing the texture")

    parser.add_argument("--smooth_mask", action="store_true", help="smooth the diffusion mask")

    parser.add_argument("--force", action="store_true", help="forcefully generate more image")

    # negative options
    parser.add_argument("--no_repaint", action="store_true", help="do NOT apply repaint")
    parser.add_argument("--no_update", action="store_true", help="do NOT apply update")

    # device parameters
    # parser.add_argument("--device", type=str, choices=["a6000", "2080"], default="a6000")
    parser.add_argument("--device", type=str, choices=["A40", "2080"], default="a6000")
    # camera parameters NOTE need careful tuning!!!
    parser.add_argument("--test_camera", action="store_true")
    parser.add_argument("--dist", type=float, default=1, 
        help="distance to the camera from the object")
    parser.add_argument("--elev", type=float, default=0,
        help="the angle between the vector from the object to the camera and the horizontal plane")
    parser.add_argument("--azim", type=float, default=180,
        help="the angle between the vector from the object to the camera and the vertical plane")

    args = parser.parse_args()

    if args.device == "a6000":  #后续正式运行时应该改为"A40"
        setattr(args, "render_simple_factor", 12) #图像渲染的简化因子。这个值通常用于控制渲染过程中的精细程度。较大的值可以提高图像的精细度
        setattr(args, "fragment_k", 1)
        setattr(args, "image_size", 768)
        setattr(args, "uv_size", 3000)
    else:
        setattr(args, "render_simple_factor", 4)
        setattr(args, "fragment_k", 1)
        setattr(args, "image_size", 768)
        setattr(args, "uv_size", 1000)

    return args


if __name__ == "__main__":
    args = init_args()

    # save输出和保存路径的文件路径和格式
    output_dir = os.path.join(
        args.output_dir, 
        "{}-{}-{}-{}-{}-{}".format(   #42-p36-h20-1.0-0.3-0.1
            str(args.seed),
            args.viewpoint_mode[0]+str(args.num_viewpoints),
            args.update_mode[0]+str(args.update_steps),
            str(args.new_strength),
            str(args.update_strength),
            str(args.view_threshold)
        ),
    )
    if args.no_repaint: output_dir += "-norepaint"
    if args.no_update: output_dir += "-noupdate"

    os.makedirs(output_dir, exist_ok=True)
    print("=> OUTPUT_DIR:", output_dir)

    #初始化各类用于生成和更新3D模型纹理的资源
    # init mesh初始化3D物体的网格，通过加载指定路径下的OBJ文件来创建模型的三角网格
    mesh, _, faces, aux, principle_directions, mesh_center, mesh_scale = init_mesh(os.path.join(args.input_dir, args.obj_file), DEVICE)

    # gradient texture打开一个空白的假纹理图像dummy.png，用于初始化纹理
    init_texture = Image.open("./samples/textures/dummy.png").convert("RGB").resize((args.uv_size, args.uv_size))
    init_material= Image.open("./samples/textures/dummy.png").convert("RGB").resize((args.uv_size, args.uv_size))
    # HACK adjust UVs for multiple materials
    if args.use_multiple_objects:          #是否是处理多个物体的场景
        new_verts_uvs, init_texture = adjust_uv_map(faces, aux, init_texture, args.uv_size)
    else:           
        new_verts_uvs = aux.verts_uvs    #如果只是单个物体，则保留原始的UV映射

    # update the mesh 在更新过程中，将模型的顶点信息与 UV 图上的信息进行对照，以确保纹理图像————>3D模型mesh表面。
    mesh.textures = TexturesUV(    
        maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE), #maps 参数接受了一个纹理图像，这个图像会被映射到模型的表面上。这个图像经过了处理，以确保它的格式适用于 PyTorch 3D 库。
        faces_uvs=faces.textures_idx[None, ...],  #faces_uvs 参数包含了三角面片的 UV 坐标索引。每个三角面片都有一个对应的 UV 坐标索引，用于确定在 UV 图上的哪个位置应用纹理。
        verts_uvs=new_verts_uvs[None, ...]  #verts_uvs 参数包含了模型的顶点的 UV 坐标。每个顶点都有一个 UV 坐标，用于确定在 UV 图上的哪个位置应用纹理
    )
    

    # back-projected faces
    #创建了一个全零的二维张量，用于保存控制深度到图像的投影。它是一个用于深度到图像映射的辅助张量
    exist_texture = torch.from_numpy(np.zeros([args.uv_size, args.uv_size]).astype(np.float32)).to(DEVICE)

    # initialize viewpoints 初始化了视点（观察角度）。它包括主要的用于生成的原始视点（principle viewpoints），以及用于更新的细化视点（refinement viewpoints）
    # including: principle viewpoints for generation + refinement viewpoints for updating
    (
        dist_list, 
        elev_list, 
        azim_list, 
        sector_list,
        view_punishments
    ) = init_viewpoints(args.viewpoint_mode, args.num_viewpoints, args.dist, args.elev, principle_directions, 
                            use_principle=True, 
                            use_shapenet=args.use_shapenet,
                            use_objaverse=args.use_objaverse)

    # save args将初始化参数保存到指定的输出目录中，以便将来参考和记录
    save_args(args, output_dir)

    # initialize depth2image model 初始化了用于控制深度到图像的投影的模型，以及一个采样器。这将在模型生成和更新过程中用于深度到图像的计算
    controlnet, ddim_sampler = get_controlnet_depth() 


    # ------------------- 下面开始为操作的代码区 ------------------------

    # 1. generate texture with RePaint 生成纹理
    # NOTE no update / refinement

    generate_dir = os.path.join(output_dir, "generate")
    os.makedirs(generate_dir, exist_ok=True)

    update_dir = os.path.join(output_dir, "update")
    os.makedirs(update_dir, exist_ok=True)

    init_image_dir = os.path.join(generate_dir, "rendering")
    os.makedirs(init_image_dir, exist_ok=True)

    normal_map_dir = os.path.join(generate_dir, "normal")
    os.makedirs(normal_map_dir, exist_ok=True)

    mask_image_dir = os.path.join(generate_dir, "mask")
    os.makedirs(mask_image_dir, exist_ok=True)

    depth_map_dir = os.path.join(generate_dir, "depth")
    os.makedirs(depth_map_dir, exist_ok=True)

    similarity_map_dir = os.path.join(generate_dir, "similarity")
    os.makedirs(similarity_map_dir, exist_ok=True)

    inpainted_image_dir = os.path.join(generate_dir, "inpainted")
    os.makedirs(inpainted_image_dir, exist_ok=True)

    mesh_dir = os.path.join(generate_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    interm_dir = os.path.join(generate_dir, "intermediate")
    os.makedirs(interm_dir, exist_ok=True)

    # prepare viewpoints and cache 使用视点信息构建相似性纹理缓存
    NUM_PRINCIPLE = 10 if args.use_shapenet or args.use_objaverse else 6
    pre_dist_list = dist_list[:NUM_PRINCIPLE]
    pre_elev_list = elev_list[:NUM_PRINCIPLE]
    pre_azim_list = azim_list[:NUM_PRINCIPLE]
    pre_sector_list = sector_list[:NUM_PRINCIPLE]
    pre_view_punishments = view_punishments[:NUM_PRINCIPLE]

    pre_similarity_texture_cache = build_similarity_texture_cache_for_all_views(mesh, faces, new_verts_uvs,
        pre_dist_list, pre_elev_list, pre_azim_list,
        args.image_size, args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
        DEVICE
    )

    ###############开始生成纹理的过程：生成图像 + 创建纹理UV ########################
    # start generation
    print("=> start generating texture...")
    start_time = time.time()
    for view_idx in range(NUM_PRINCIPLE):  #对每个视点进行处理
        print("=> processing view {}...".format(view_idx))

        # sequentially pop the viewpoints  
        # 通过 pre_dist_list, pre_elev_list, pre_azim_list, pre_sector_list获取当前视点的位置信息
        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx] 
        #如果 add_view_to_prompt 为真，则将视点信息添加到提示中
        prompt = " the {} view of {}".format(sector, args.prompt) if args.add_view_to_prompt else args.prompt
        print("=> generating image for prompt: {}...".format(prompt))

        # 1.1. render and build masks（生成部分的核心代码）
        #render_one_view_and_build_masks: 这个函数用于渲染一个视点并构建蒙版。它接受一系列参数，包括距离（dist）、仰角（elev）、方位角（azim）等，用来定义视点的位置和特性。
        #然后，它使用这些参数渲染视点，生成图像、法线图、深度图等。这个函数还会构建多个蒙版，包括保留蒙版（keep_mask_image）、更新蒙版（update_mask_image）、生成蒙版（generate_mask_image）等
        
        (
            view_score,
            renderer, cameras, fragments,
            init_image, normal_map, depth_map, 
            init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
            keep_mask_image, update_mask_image, generate_mask_image, 
            keep_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor,
        ) = render_one_view_and_build_masks(dist, elev, azim, 
            view_idx, view_idx, view_punishments, # => actual view idx and the sequence idx 
            pre_similarity_texture_cache, exist_texture,
            mesh, faces, new_verts_uvs,
            args.image_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
            DEVICE, save_intermediate=True, smooth_mask=args.smooth_mask, view_threshold=args.view_threshold
        )

        # 1.2. generate missing region生成缺失区域的纹理
        # NOTE first view still gets the mask for consistent ablations
        if args.no_repaint and view_idx != 0: #如果 args.no_repaint 为 True 并且 view_idx 不等于 0，也就是对于后续的视点（非第一个视点），则创建一个全白的图像作为 actual_generate_mask_image。
            actual_generate_mask_image = Image.fromarray((np.ones_like(np.array(generate_mask_image)) * 255.).astype(np.uint8))
        else:
            actual_generate_mask_image = generate_mask_image  #对于第一个视点或者重新绘制的视点，将 generate_mask_image 赋值给 actual_generate_mask_image，会根据生成蒙版重新绘制图像

        print("=> generate for view {}".format(view_idx))

        #@@@@@@@@@！！！！！！使用 apply_controlnet_depth 函数，根据输入参数，以及视点的图像、初始掩模、深度地图等信息，生成缺失区域的纹理########
        generate_image, generate_image_before, generate_image_after = apply_controlnet_depth(controlnet, ddim_sampler, 
            init_image.convert("RGBA"), prompt, args.new_strength, args.ddim_steps,
            actual_generate_mask_image, keep_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
            args.a_prompt, args.n_prompt, args.guidance_scale, args.seed, args.eta, 1, DEVICE, args.blend)
        #生成前、生成后、和生成的差异图像
        generate_image.save(os.path.join(inpainted_image_dir, "{}.png".format(view_idx)))
        generate_image_before.save(os.path.join(inpainted_image_dir, "{}_before.png".format(view_idx)))
        generate_image_after.save(os.path.join(inpainted_image_dir, "{}_after.png".format(view_idx)))

        # 1.2.2 back-project and create texture
        # NOTE projection mask = generate mask

        ##@@@！！！！！！！！！！！使用 backproject_from_image 函数，将生成的图像重新投影到 3D 模型上，同时更新初始纹理。
        #这一步涉及纹理映射的处理，以确保生成的纹理正确映射到 3D 模型的 UV 坐标上
        init_texture, project_mask_image, exist_texture = backproject_from_image(
            mesh, faces, new_verts_uvs, cameras, 
            generate_image, generate_mask_image, generate_mask_image, init_texture, exist_texture, 
            args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
            DEVICE
        )

        project_mask_image.save(os.path.join(mask_image_dir, "{}_project.png".format(view_idx)))
        
        # #添加材质预测图进行UV映射和纹理更新!!!!!!!!!!!!!
        material = get_material(os.path.join(inpainted_image_dir, "{}_after.png".format(view_idx)))
        # print("material.shape:",material.shape)

        init_material, project_mask_image, exist_texture = backproject_from_image(
            mesh, faces, new_verts_uvs, cameras, 
            material, generate_mask_image, generate_mask_image, init_material, exist_texture, 
            args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
            DEVICE
        )

        
        # update the mesh 更新 mesh 对象的纹理信息，包括颜色纹理（textures）、UV 坐标等。mesh 对象现在包含了新生成的纹理信息
        mesh.textures = TexturesUV(
            maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=new_verts_uvs[None, ...]
        )


        #保存时将texturing换为material

        mesh.textures = TexturesUV(
            maps=transforms.ToTensor()(init_material)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=new_verts_uvs[None, ...]
        )

        # 1.2.3. re: render 
        # NOTE only the rendered image is needed - masks should be re-used
        (
            view_score,
            renderer, cameras, fragments,
            init_image, *_,
        ) = render_one_view_and_build_masks(dist, elev, azim, 
            view_idx, view_idx, view_punishments, # => actual view idx and the sequence idx 
            pre_similarity_texture_cache, exist_texture,
            mesh, faces, new_verts_uvs,
            args.image_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
            DEVICE, save_intermediate=False, smooth_mask=args.smooth_mask, view_threshold=args.view_threshold
        )

        # 1.3. update blurry region
        # only when: 1) use update flag; 2) there are contents to update; 3) there are enough contexts.
        """
        首先，检查是否允许更新 (not args.no_update)。
        检查是否有需要更新的内容 (update_mask_tensor.sum() > 0)。
        检查更新区域的像素数量是否占总区域的5%以上 (update_mask_tensor.sum() / (all_mask_tensor.sum()) > 0.05)。
        如果上述条件都满足，进入生成纹理的步骤
        
        """
        if not args.no_update and update_mask_tensor.sum() > 0 and update_mask_tensor.sum() / (all_mask_tensor.sum()) > 0.05:
            print("=> update {} pixels for view {}".format(update_mask_tensor.sum().int(), view_idx))
            diffused_image, diffused_image_before, diffused_image_after = apply_controlnet_depth(controlnet, ddim_sampler, 
                init_image.convert("RGBA"), prompt, args.update_strength, args.ddim_steps,
                update_mask_image, keep_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
                args.a_prompt, args.n_prompt, args.guidance_scale, args.seed, args.eta, 1, DEVICE, args.blend)
            #将生成的纹理保存为图像文件，包括更新前后的图像。
            diffused_image.save(os.path.join(inpainted_image_dir, "{}_update.png".format(view_idx)))
            diffused_image_before.save(os.path.join(inpainted_image_dir, "{}_update_before.png".format(view_idx)))
            diffused_image_after.save(os.path.join(inpainted_image_dir, "{}_update_after.png".format(view_idx)))
        
            # 1.3.2. back-project and create texture更新网格纹理
            # NOTE projection mask = generate mask
            
            """
            调用 backproject_from_image 函数重新投影纹理,将生成的 diffused_image 重新投影到 3D 模型上，其中 update_mask_image 用于确定投影的区域
            """
            
            init_texture, project_mask_image, exist_texture = backproject_from_image(
                mesh, faces, new_verts_uvs, cameras, 
                diffused_image, update_mask_image, update_mask_image, init_texture, exist_texture, 
                args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
                DEVICE
            )
            
            #添加材质预测图进行UV映射和纹理更新!!!!!!!!!!!!!
            #将update的图作为材质预测函数的输入图片
            material = get_material(os.path.join(inpainted_image_dir, "{}_update_after.png".format(view_idx)))

            init_material, project_mask_image, exist_texture = backproject_from_image(
                mesh, faces, new_verts_uvs, cameras, 
                material, update_mask_image, update_mask_image, init_material, exist_texture, 
                args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
                DEVICE
            )



            """
            更新网格的纹理信息：
            这部分代码用TexturesUV更新了模型的纹理信息，将新生成的纹理映射到模型的 UV 坐标上，从而更新了模型的外观。
            """
            # update the mesh
            mesh.textures = TexturesUV(
                maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
                faces_uvs=faces.textures_idx[None, ...],
                verts_uvs=new_verts_uvs[None, ...]
            )
            
            #保存时将texturing换为material
            mesh.textures = TexturesUV(
                maps=transforms.ToTensor()(init_material)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
                faces_uvs=faces.textures_idx[None, ...],
                verts_uvs=new_verts_uvs[None, ...]
            )


        # 1.4. save generated assets
        # save backprojected OBJ file保存反投影的 OBJ 文件，包括了模型的顶点、UV坐标、纹理信息等


        # save_backproject_obj(
        #     mesh_dir, "{}.obj".format(view_idx),
        #     mesh_scale * mesh.verts_packed() + mesh_center if args.use_unnormalized else mesh.verts_packed(),
        #     faces.verts_idx, new_verts_uvs, faces.textures_idx, init_texture, 
        #     DEVICE
        # )


        save_backproject_obj(
            mesh_dir, "{}.obj".format(view_idx),
            mesh_scale * mesh.verts_packed() + mesh_center if args.use_unnormalized else mesh.verts_packed(),
            faces.verts_idx, new_verts_uvs, faces.textures_idx, init_material, 
            DEVICE
        )



        # save the intermediate view 保存中间渲染视图，即模型在生成过程中的中间结果，保存为图像文件
        inter_images_tensor, *_ = render(mesh, renderer)
        inter_image = inter_images_tensor[0].cpu()
        inter_image = inter_image.permute(2, 0, 1)
        inter_image = transforms.ToPILImage()(inter_image).convert("RGB")
        inter_image.save(os.path.join(interm_dir, "{}.png".format(view_idx)))

        # save texture mask将纹理蒙版（texture mask）保存为图像文件，以便进一步的分析和可视化
        exist_texture_image = exist_texture * 255. 
        exist_texture_image = Image.fromarray(exist_texture_image.cpu().numpy().astype(np.uint8)).convert("L")
        exist_texture_image.save(os.path.join(mesh_dir, "{}_texture_mask.png".format(view_idx)))

    print("=> total generate time: {} s".format(time.time() - start_time))

    # visualize viewpoints
    visualize_principle_viewpoints(output_dir, pre_dist_list, pre_elev_list, pre_azim_list)

    # # 2. update texture with RePaint 

    # if args.update_steps > 0:

    #     update_dir = os.path.join(output_dir, "update")
    #     os.makedirs(update_dir, exist_ok=True)

    #     init_image_dir = os.path.join(update_dir, "rendering")
    #     os.makedirs(init_image_dir, exist_ok=True)

    #     normal_map_dir = os.path.join(update_dir, "normal")
    #     os.makedirs(normal_map_dir, exist_ok=True)

    #     mask_image_dir = os.path.join(update_dir, "mask")
    #     os.makedirs(mask_image_dir, exist_ok=True)

    #     depth_map_dir = os.path.join(update_dir, "depth")
    #     os.makedirs(depth_map_dir, exist_ok=True)

    #     similarity_map_dir = os.path.join(update_dir, "similarity")
    #     os.makedirs(similarity_map_dir, exist_ok=True)

    #     inpainted_image_dir = os.path.join(update_dir, "inpainted")
    #     os.makedirs(inpainted_image_dir, exist_ok=True)

    #     mesh_dir = os.path.join(update_dir, "mesh")
    #     os.makedirs(mesh_dir, exist_ok=True)

    #     interm_dir = os.path.join(update_dir, "intermediate")
    #     os.makedirs(interm_dir, exist_ok=True)

    #     dist_list = dist_list[NUM_PRINCIPLE:]
    #     elev_list = elev_list[NUM_PRINCIPLE:]
    #     azim_list = azim_list[NUM_PRINCIPLE:]
    #     sector_list = sector_list[NUM_PRINCIPLE:]
    #     view_punishments = view_punishments[NUM_PRINCIPLE:]

    #     similarity_texture_cache = build_similarity_texture_cache_for_all_views(mesh, faces, new_verts_uvs,
    #         dist_list, elev_list, azim_list,
    #         args.image_size, args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
    #         DEVICE
    #     )
    #     selected_view_ids = []

    #     print("=> start updating...")
    #     start_time = time.time()
    #     for view_idx in range(args.update_steps):
    #         print("=> processing view {}...".format(view_idx))
            
    #         # 2.1. render and build masks

    #         # heuristically select the viewpoints
    #         dist, elev, azim, sector, selected_view_ids, view_punishments = select_viewpoint(
    #             selected_view_ids, view_punishments,
    #             args.update_mode, dist_list, elev_list, azim_list, sector_list, view_idx,
    #             similarity_texture_cache, exist_texture,
    #             mesh, faces, new_verts_uvs,
    #             args.image_size, args.fragment_k,
    #             init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
    #             DEVICE, False
    #         )

    #         (
    #             view_score,
    #             renderer, cameras, fragments,
    #             init_image, normal_map, depth_map, 
    #             init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
    #             old_mask_image, update_mask_image, generate_mask_image, 
    #             old_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor,
    #         ) = render_one_view_and_build_masks(dist, elev, azim, 
    #             selected_view_ids[-1], view_idx, view_punishments, # => actual view idx and the sequence idx 
    #             similarity_texture_cache, exist_texture,
    #             mesh, faces, new_verts_uvs,
    #             args.image_size, args.fragment_k,
    #             init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
    #             DEVICE, save_intermediate=True, smooth_mask=args.smooth_mask, view_threshold=args.view_threshold
    #         )

    #         # # -------------------- OPTION ZONE ------------------------
    #         # # still generate for missing regions during refinement
    #         # # NOTE this could take significantly more time to complete.
    #         # if args.use_patch:
    #         #     # 2.2.1 generate missing region
    #         #     prompt = " the {} view of {}".format(sector, args.prompt) if args.add_view_to_prompt else args.prompt
    #         #     print("=> generating image for prompt: {}...".format(prompt))

    #         #     if args.no_repaint:
    #         #         generate_mask_image = Image.fromarray((np.ones_like(np.array(generate_mask_image)) * 255.).astype(np.uint8))

    #         #     print("=> generate {} pixels for view {}".format(generate_mask_tensor.sum().int(), view_idx))
    #         #     generate_image, generate_image_before, generate_image_after = apply_controlnet_depth(controlnet, ddim_sampler, 
    #         #         init_image.convert("RGBA"), prompt, args.new_strength, args.ddim_steps,
    #         #         generate_mask_image, keep_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
    #         #         args.a_prompt, args.n_prompt, args.guidance_scale, args.seed, args.eta, 1, DEVICE, args.blend)

    #         #     generate_image.save(os.path.join(inpainted_image_dir, "{}_new.png".format(view_idx)))
    #         #     generate_image_before.save(os.path.join(inpainted_image_dir, "{}_new_before.png".format(view_idx)))
    #         #     generate_image_after.save(os.path.join(inpainted_image_dir, "{}_new_after.png".format(view_idx)))

    #         #     # 2.2.2. back-project and create texture
    #         #     # NOTE projection mask = generate mask
    #         #     init_texture, project_mask_image, exist_texture = backproject_from_image(
    #         #         mesh, faces, new_verts_uvs, cameras, 
    #         #         generate_image, generate_mask_image, generate_mask_image, init_texture, exist_texture, 
    #         #         args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
    #         #         DEVICE
    #         #     )

    #         #     project_mask_image.save(os.path.join(mask_image_dir, "{}_new_project.png".format(view_idx)))

    #         #     # update the mesh
    #         #     mesh.textures = TexturesUV(
    #         #         maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
    #         #         faces_uvs=faces.textures_idx[None, ...],
    #         #         verts_uvs=new_verts_uvs[None, ...]
    #         #     )

    #         #     # 2.2.4. save generated assets
    #         #     # save backprojected OBJ file
    #         #     save_backproject_obj(
    #         #         mesh_dir, "{}_new.obj".format(view_idx),
    #         #         mesh.verts_packed(), faces.verts_idx, new_verts_uvs, faces.textures_idx, init_texture, 
    #         #         DEVICE
    #         #     )

    #         # # -------------------- OPTION ZONE ------------------------


    #         # 2.2. update existing region
    #         prompt = " the {} view of {}".format(sector, args.prompt) if args.add_view_to_prompt else args.prompt
    #         print("=> updating image for prompt: {}...".format(prompt))

    #         if not args.no_update and update_mask_tensor.sum() > 0 and update_mask_tensor.sum() / (all_mask_tensor.sum()) > 0.05:
    #             print("=> update {} pixels for view {}".format(update_mask_tensor.sum().int(), view_idx))
    #             update_image, update_image_before, update_image_after = apply_controlnet_depth(controlnet, ddim_sampler, 
    #                 init_image.convert("RGBA"), prompt, args.update_strength, args.ddim_steps,
    #                 update_mask_image, old_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
    #                 args.a_prompt, args.n_prompt, args.guidance_scale, args.seed, args.eta, 1, DEVICE, args.blend)

    #             update_image.save(os.path.join(inpainted_image_dir, "{}.png".format(view_idx)))
    #             update_image_before.save(os.path.join(inpainted_image_dir, "{}_before.png".format(view_idx)))
    #             update_image_after.save(os.path.join(inpainted_image_dir, "{}_after.png".format(view_idx)))
    #         else:
    #             print("=> nothing to update for view {}".format(view_idx))
    #             update_image = init_image

    #             old_mask_tensor += update_mask_tensor
    #             update_mask_tensor[update_mask_tensor == 1] = 0 # HACK nothing to update

    #             old_mask_image = transforms.ToPILImage()(old_mask_tensor)
    #             update_mask_image = transforms.ToPILImage()(update_mask_tensor)


    #         # 2.3. back-project and create texture
    #         # NOTE projection mask = update mask
    #         init_texture, project_mask_image, exist_texture = backproject_from_image(
    #             mesh, faces, new_verts_uvs, cameras, 
    #             update_image, update_mask_image, update_mask_image, init_texture, exist_texture, 
    #             args.image_size * args.render_simple_factor, args.uv_size, args.fragment_k,
    #             DEVICE
    #         )

    #         project_mask_image.save(os.path.join(mask_image_dir, "{}_project.png".format(view_idx)))

    #         # update the mesh
    #         mesh.textures = TexturesUV(
    #             maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
    #             faces_uvs=faces.textures_idx[None, ...],
    #             verts_uvs=new_verts_uvs[None, ...]
    #         )

    #         # 2.4. save generated assets
    #         # save backprojected OBJ file            
    #         save_backproject_obj(
    #             mesh_dir, "{}.obj".format(view_idx),
    #             mesh_scale * mesh.verts_packed() + mesh_center if args.use_unnormalized else mesh.verts_packed(),
    #             faces.verts_idx, new_verts_uvs, faces.textures_idx, init_texture, 
    #             DEVICE
    #         )

    #         # save the intermediate view
    #         inter_images_tensor, *_ = render(mesh, renderer)
    #         inter_image = inter_images_tensor[0].cpu()
    #         inter_image = inter_image.permute(2, 0, 1)
    #         inter_image = transforms.ToPILImage()(inter_image).convert("RGB")
    #         inter_image.save(os.path.join(interm_dir, "{}.png".format(view_idx)))

    #         # save texture mask
    #         exist_texture_image = exist_texture * 255. 
    #         exist_texture_image = Image.fromarray(exist_texture_image.cpu().numpy().astype(np.uint8)).convert("L")
    #         exist_texture_image.save(os.path.join(mesh_dir, "{}_texture_mask.png".format(view_idx)))

    #     print("=> total update time: {} s".format(time.time() - start_time))

    #     # post-process
    #     if args.post_process:
    #         del controlnet
    #         del ddim_sampler

    #         inpainting = get_inpainting(DEVICE)
    #         post_texture = apply_inpainting_postprocess(inpainting, 
    #             init_texture, 1-exist_texture[None, :, :, None], "", args.uv_size, args.uv_size, DEVICE)

    #         save_backproject_obj(
    #             mesh_dir, "{}_post.obj".format(view_idx),
    #             mesh_scale * mesh.verts_packed() + mesh_center if args.use_unnormalized else mesh.verts_packed(),
    #             faces.verts_idx, new_verts_uvs, faces.textures_idx, post_texture, 
    #             DEVICE
    #         )
    
    #     # save viewpoints
    #     save_viewpoints(args, output_dir, dist_list, elev_list, azim_list, selected_view_ids)

    #     # visualize viewpoints
    #     visualize_refinement_viewpoints(output_dir, selected_view_ids, dist_list, elev_list, azim_list)
