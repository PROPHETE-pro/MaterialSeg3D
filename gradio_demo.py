import gradio as gr
import argparse, sys, os, math, re

import numpy as np
import json

import sys
sys.path.append("./mmsegmentation")

from mmseg.apis import init_model, inference_model, show_result_pyplot
import cv2
import pdb
import torch

import os
import argparse
import time
import imageio
import pdb
from tqdm import tqdm
import cv2
import pickle
import json
from scipy import stats


sys.path.append("./Text2Tex")

# pytorch3d
from pytorch3d.renderer import TexturesUV
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    AmbientLights,
    SoftPhongShader
)
from pytorch3d.ops import interpolate_face_attributes

from lib.render_helper import init_renderer, render
from lib.projection_helper import (build_backproject_mask, build_diffusion_mask, compose_quad_mask, compute_view_heat)
from lib.shading_helper import init_soft_phong_shader

from lib.camera_helper import init_camera
from lib.render_helper import init_renderer, render
from lib.shading_helper import (
    BlendParams,
    init_soft_phong_shader,
    init_flat_texel_shader,
)
from lib.vis_helper import visualize_outputs, visualize_quad_mask
from lib.constants import *

import torch
from torchvision import transforms

from PIL import Image

from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform,
    FoVPerspectiveCameras
)

from lib.mesh_helper import (
    init_mesh_2,
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
    get_all_4_locations,
    select_viewpoint,
)
from lib.camera_helper import init_viewpoints


mapping = {3:2, 4:2, 5:3, 6:4, 7:5, 8:6, 9:6, 10:7, 11:7, 12:8, 13:9, 14:10, 15:11, 16:12, 17:13, 18:14, 19:15}
palette=[[0,0,0], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [100, 100, 0]]

def get_rendering(sample_folder):
    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]

    file_list = os.listdir(sample_folder)
    for file in file_list:
        if file.endswith('.obj'):

            cmd = f'./blender-2.90.0-linux64/blender -b -P ./GET3D/render_shapenet_data/render_shapenet.py -- --output ./output {os.path.join(sample_folder, file)} --scale 1 --views 41 --resolution 1024 >> tmp.out'
            os.system(cmd)

            img_list = []
            # pdb.set_trace()
            for i in range(5):
                out_dir = os.path.join('./output/Image', sample, f'{sample}_{i}.png')
                img = cv2.imread(out_dir)
                img_rgb = img[:,:,[2,1,0]]
                img_list.append(img_rgb)

            return img_list[0],img_list[1],img_list[2],img_list[3],img_list[4]


def get_segmentation(sample_folder, category):
    def getFileList(dir, Filelist, ext=None, skip=None, spec=None):
        newDir = dir
        if os.path.isfile(dir):
            if ext is None:
                Filelist.append(dir)
            else:
                if ext in dir[-3:]:
                    Filelist.append(dir)

        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                if os.path.isdir(os.path.join(dir, s)):
                    newDir = os.path.join(dir, s)
                    getFileList(newDir, Filelist, ext, skip, spec)

                else:
                    acpt = True
                    if skip is not None:
                        for skipi in skip:
                            if skipi in s:
                                acpt = False
                                break

                    if acpt == False:
                        continue
                    else:
                        sp = False
                        if spec is not None:
                            for speci in spec:
                                if speci in s:
                                    sp = True
                                    break

                        else:
                            sp = True

                        if sp == False:
                            continue
                        else:
                            newDir = os.path.join(dir, s)
                            getFileList(newDir, Filelist, ext, skip, spec)

        return Filelist

    def to_rgb(label, palette):
        h, w = label.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                value = label[i, j]
                rgb_value = palette[value]
                rgb[i, j] = rgb_value

        return rgb

    def transfer2(image, mapping):
        image_base = image.copy()

        for i in mapping.keys():
            image[image_base == i] = mapping[i]

        return image

    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]
    img_list = getFileList(os.path.join('./output/Image/', sample), [], ext='png')
    # pdb.set_trace()
    for img in img_list:
        image = cv2.imread(img)
        img_name = img.split('/')[-1]

        back1 = np.array([0, 0, 0])
        back2 = np.array([1, 1, 1])
        target_color = np.array([255, 255, 255])
        image[np.all(image==back1, axis=2)] = target_color
        image[np.all(image==back2, axis=2)] = target_color

        save_file = img.replace('Image', 'Image_white')
        save_dir = save_file.rstrip(img_name)
        os.makedirs(save_dir, exist_ok=True)

        try:
            cv2.imwrite(save_file, image)
        except:
            pdb.set_trace()

    seg_list = getFileList(save_dir, [], ext='png')
    config_path = os.path.join('./mmsegmentation/work_dir', category, f'3D_texture_{category}.py')
    checkpoint_path = os.path.join('./mmsegmentation/work_dir', category, 'ckpt.pth')
    model = init_model(config_path, checkpoint_path)

    print(len(seg_list))
    i = 0
    # pdb.set_trace()
    for img in seg_list:
        print('i = ', i)
        img_name = img.split('/')[-1]

        save_dir = img.rstrip(img_name).replace('Image_white', 'predict')
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, img_name)

        visual_dir = save_dir.replace('predict', 'vis')
        visual_path = save_path.replace('predict', 'vis')
        os.makedirs(visual_dir, exist_ok=True)

        result = inference_model(model, img)
        predict = result.pred_sem_seg.data
        save_pred = np.squeeze(predict.cpu().numpy())
        save_mapping = transfer2(save_pred, mapping)
        cv2.imwrite(os.path.join(save_path), save_mapping)

        vis_img = to_rgb(save_pred, palette)
        cv2.imwrite(os.path.join(visual_path), vis_img)

        i += 1

    vis_list = []
    for i in range(5):
        seg_result = os.path.join('./output/vis', sample, f'{sample}_{i}.png')
        seg = cv2.imread(seg_result)
        seg_rgb = seg[:, :, [2, 1, 0]]
        vis_list.append(seg_rgb)

    return vis_list[0],vis_list[1],vis_list[2],vis_list[3],vis_list[4]

def render_to_uv(sample_folder, category):
    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]

    os.chdir('./Text2Tex')
    cmd = f'python ./scripts/view_2_UV.py --cuda 2 --work_dir ../output/predict --sample_dir {sample_folder} --sample {sample} --img_size 512 --category {category}'
    os.system(cmd)

    os.chdir('../')

    ORM_dir = os.path.join('./output/ORM/', sample, 'ORM.png')

    os.system('cp ' + ORM_dir + ' ' + sample_folder)

    ORM = cv2.imread(ORM_dir)
    ORM_rgb = ORM[:,:,[2,1,0]]


    cmd2 = f'/path-to-MaterialSeg3D/blender-2.90.0-linux64/blender -b -P material_glb.py -- --obj_file {sample_folder} --orm_path {ORM_dir}'
    os.system(cmd2)

    glb_path = os.path.join(sample_folder, sample+'.glb')

    return ORM_rgb, glb_path


def display(sample_folder):
    sample_folder = sample_folder.rstrip('/')
    sample = sample_folder.split('/')[-1]

    for file in os.listdir(sample_folder):
        if file.endswith('png'):
            uv = Image.open(os.path.join(sample_folder,file))

    cmd = f'/path-to-MaterialSeg3D/blender-2.90.0-linux64/blender -b -P trans_glb.py -- --obj_file {sample_folder}'

    os.system(cmd)
    mesh_path = os.path.join(sample_folder, f'{sample}_raw.glb')

    return uv, mesh_path

def example(sample_folder):
    return './figure/material_ue.png', './figure/material_car.png', './figure/raw_ue.png', './figure/raw_car.png'

with gr.Blocks(title="MaterialSeg3D") as interface:
    gr.Markdown(
        """
    # MaterialSeg3D Demo
    
    **Tips:**
    1. Please input the directory of the folder containing the .obj file and .png Albedo UV (do not contain any other file ends with .png or .obj).
    2. Do not input the quotation mark of the directory (i.e. /path/to/asset).
    3. It requires ~5 minutes to run the segmentation step, and ~15 minutes for the material generation. Please refer to the local storage for the process.
    """
    )
    with gr.Column():
        with gr.Row():
            with gr.Column():
                input_dir = gr.Textbox(lines=5, placeholder="Input directory path", label="Path to the folder includes object")
                select_cat = gr.Dropdown(['car', 'furniture', 'building', 'instrument', 'plant'],
                                           label="Category", info="Choose the category of the asset")
            albedo_uv = gr.Image(interactive=False, show_label=False)
            input_mesh = gr.Model3D(interactive=False, label='Input mesh')
        with gr.Row():
            view_1 = gr.Image(interactive=False, height=240, show_label=False)
            view_2 = gr.Image(interactive=False, height=240, show_label=False)
            view_3 = gr.Image(interactive=False, height=240, show_label=False)
            view_4 = gr.Image(interactive=False, height=240, show_label=False)
            view_5 = gr.Image(interactive=False, height=240, show_label=False)

        with gr.Row():
            seg1 = gr.Image(interactive=False, height=240, show_label=False)
            seg2 = gr.Image(interactive=False, height=240, show_label=False)
            seg3 = gr.Image(interactive=False, height=240, show_label=False)
            seg4 = gr.Image(interactive=False, height=240, show_label=False)
            seg5 = gr.Image(interactive=False, height=240, show_label=False)

        with gr.Row():
            ORM = gr.Image(interactive=False, label='ORM UV map with Opacity, Roughness, and Metallic')
            material_glb = gr.Model3D(interactive=False, label='Materialized object')

        seg_btn = gr.Button('Rendering', variant='primary', interactive=True)
        uv_btn = gr.Button('Materializing', variant='primary', interactive=True)

        gr.Markdown(
            """
        # Render your own asset
        
        **Note:**
        * The display quality of gr.Model3D cannot show the visual effect of the applied material. We highly recommend you to render the object within the render engine.

        **Steps:**
        1. Download the provided ORM UV map from the displayed result and place it in the input directory (already saved to the given folder).
        2. Open the render engine for 3D model (Blender, Unity Engine, etc).
        3. Map the Metallic and Roughness channel with the R and G channel of the ORM UV, respectively.
        4. We provide render settings in UE5 as an example.
        """
        )

        with gr.Row():
            raw_ue = gr.Image(interactive=False,label='Default UE setting without ORM UV')
            raw_car = gr.Image(interactive=False, label='Default display of the object')

        with gr.Row():
            mat_ue = gr.Image(interactive=False, label='UE setting applying ORM UV')
            mat_car = gr.Image(interactive=False, label='object rendering with applying ORM UV')


    seg_btn.click(fn=display, inputs=[input_dir],outputs=[albedo_uv,input_mesh]).success(fn=get_rendering, inputs=[input_dir], outputs=[view_1,view_2,view_3,view_4,view_5]).success(
        fn=get_segmentation, inputs=[input_dir, select_cat], outputs=[seg1,seg2,seg3,seg4,seg5])
    uv_btn.click(fn=render_to_uv, inputs=[input_dir, select_cat], outputs=[ORM, material_glb]).success(fn=example, inputs=[input_dir], outputs=[mat_ue, mat_car, raw_ue, raw_car])

interface.launch(share=True)
