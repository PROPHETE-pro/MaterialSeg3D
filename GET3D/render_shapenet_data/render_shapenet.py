# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

#第243行是指定相机坐标的，可以替换

import argparse, sys, os, math, re
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json 
import mathutils




parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--views', type=int, default=24,
    help='number of views to be rendered')
parser.add_argument(
    'obj', type=str,
    help='Path to the obj file to be rendered.')
parser.add_argument(
    '--output_folder', type=str, default='/tmp',
    help='The path the output will be dumped to.')
parser.add_argument(
    '--scale', type=float, default=1,
    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument(
    '--format', type=str, default='PNG',
    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument(
    '--resolution', type=int, default=1024,
    help='Resolution of the images.')
parser.add_argument(
    '--engine', type=str, default='CYCLES',
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
bpy.context.scene.cycles.filter_width = 0.01
bpy.context.scene.render.film_transparent = True

bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.diffuse_bounces = 1
bpy.context.scene.cycles.glossy_bounces = 1
bpy.context.scene.cycles.transparent_max_bounces = 3
bpy.context.scene.cycles.transmission_bounces = 3
bpy.context.scene.cycles.samples = 32
bpy.context.scene.cycles.use_denoising = True


def enable_cuda_devices():
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print("Compute device selected: {0}".format(compute_device_type))
            break
        except TypeError:
            pass

    # Any CUDA/OPENCL devices?
    acceleratedTypes = ['CUDA', 'OPENCL']
    accelerated = any(device.type in acceleratedTypes for device in cprefs.devices)
    print('Accelerated render = {0}'.format(accelerated))

    # If we have CUDA/OPENCL devices, enable only them, otherwise enable
    # all devices (assumed to be CPU)
    print(cprefs.devices)
    for device in cprefs.devices:
        device.use = not accelerated or device.type in acceleratedTypes
        print('Device enabled ({type}) = {enabled}'.format(type=device.type, enabled=device.use))

    return accelerated


enable_cuda_devices()
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')


def bounds(obj, local=False):
    local_coords = obj.bound_box[:]
    om = obj.matrix_world

    if not local:
        worldify = lambda p: om @ Vector(p[:])
        coords = [worldify(p).to_tuple() for p in local_coords]
    else:
        coords = [p[:] for p in local_coords]

    rotated = zip(*coords[::-1])

    push_axis = []
    for (axis, _list) in zip('xyz', rotated):
        info = lambda: None
        info.max = max(_list)
        info.min = min(_list)
        info.distance = info.max - info.min
        push_axis.append(info)

    import collections

    originals = dict(zip(['x', 'y', 'z'], push_axis))

    o_details = collections.namedtuple('object_details', 'x y z')
    return o_details(**originals)

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1*R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT
# import pdb;pdb.set_trace
imported_object = bpy.ops.import_scene.obj(filepath=args.obj, use_edges=False, use_smooth_groups=False, split_mode='OFF')

for this_obj in bpy.data.objects:
    if this_obj.type == "MESH":
        this_obj.select_set(True)
        bpy.context.view_layer.objects.active = this_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.split_normals()

bpy.ops.object.mode_set(mode='OBJECT')
obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

mesh_obj = obj
scale = args.scale
# # 以下注释代码为二次normal，调整模型大小的代码

# factor = max(mesh_obj.dimensions[0], mesh_obj.dimensions[1], mesh_obj.dimensions[2]) / scale
# print('size of object:')
# print(mesh_obj.dimensions)
# print(factor)
# object_details = bounds(mesh_obj)
# print(
#     object_details.x.min, object_details.x.max,
#     object_details.y.min, object_details.y.max,
#     object_details.z.min, object_details.z.max,
# )
# print(bounds(mesh_obj))
# mesh_obj.scale[0] /= factor
# mesh_obj.scale[1] /= factor
# mesh_obj.scale[2] /= factor
# bpy.ops.object.transform_apply(scale=True)

bpy.ops.object.light_add(type='AREA')
light2 = bpy.data.lights['Area']

light2.energy = 40000
bpy.data.objects['Area'].location[2] = 1
bpy.data.objects['Area'].scale[0] = 100
bpy.data.objects['Area'].scale[1] = 100
bpy.data.objects['Area'].scale[2] = 100

# Place camera
cam = scene.objects['Camera']
cam.location = (0, 1.2, 0)  # radius equals to 1


cam.data.lens = 16
cam.data.sensor_width = 32

# bpy.data.cameras[0].angle_x = 1.8


cam_constraint = cam.constraints.new(type='TRACK_TO')
# cam_constraint.track_axis = 'TRACK_Z'
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty


rotation_mode = 'XYZ'

model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
synset_idx = args.obj.split('/')[-3]

img_follder = os.path.join(os.path.abspath(args.output_folder), 'Image', model_identifier)
# camera_follder = os.path.join(os.path.abspath(args.output_folder), 'camera',  model_identifier)

os.makedirs(img_follder, exist_ok=True)
# os.makedirs(camera_follder, exist_ok=True)

# index = 0
# cam.matrix_world = P2B(R_data[index], T_data[index])
# B_all = np.load('/data/zeyu_li/data/test/blender/data/all_B_array.npy')

# #随机指定camera视角的信息
# rotation_angle_list = np.random.rand(args.views)
# elevation_angle_list = np.random.rand(args.views)
# rotation_angle_list = rotation_angle_list * 360
# elevation_angle_list = elevation_angle_list * 80 - 40

# 可以替换为指定视角npy的信息
rotation_angle_list = np.load("/path-to-MaterialSeg3D/GET3D/rotation.npy")
elevation_angle_list = np.load("/path-to-MaterialSeg3D/GET3D/elevation.npy")

# np.save(os.path.join(camera_follder, 'rotation'), rotation_angle_list)
# np.save(os.path.join(camera_follder, 'elevation'), elevation_angle_list)
to_export = {
    'camera_angle_x': bpy.data.cameras[0].angle_x,
    # "aabb": [[-scale/2,-scale/2,-scale/2],
    #          [scale/2,scale/2,scale/2]]
}
frames = [] 

# 循环每个相机位置
for i in range(0, args.views):
    
    # # 将欧拉角应用于cam_empty的旋转
    # cam_empty.rotation_euler = rotation_euler
    cam_empty.rotation_euler[2] = math.radians(rotation_angle_list[i])
    cam_empty.rotation_euler[0] = math.radians(elevation_angle_list[i])
    # 其他渲染相关的操作
    render_file_path = os.path.join(img_follder, model_identifier+'_%01d.png' % (i))
    scene.render.filepath = render_file_path
    bpy.ops.render.render(write_still=True)
    bpy.context.view_layer.update()
    

# for i in range(0, args.views):
#     # Extract rotation and translation from B_all[i]
#     rotation_matrix = B_all[i][:3, :3]  # Extract the 3x3 rotation matrix
#     translation = B_all[i][:3, 3]  # Extract the translation vector

#     # Set the rotation of the cam_empty
#     rotation_matrix_blender = mathutils.Matrix(rotation_matrix)
#     rotation_quaternion = rotation_matrix_blender.to_quaternion()
#     cam_empty.rotation_mode = 'QUATERNION'
#     cam_empty.rotation_quaternion = rotation_quaternion

#     # Set the translation of the cam_empty
#     cam_empty.location = mathutils.Vector(translation)

#     print(f"Rendering view {i}")
#     render_file_path = os.path.join(img_follder, f'{str(i).zfill(3)}.png')
#     scene.render.filepath = render_file_path
#     bpy.ops.render.render(write_still=True)
#     bpy.context.view_layer.update()



#     # cam_empty.rotation_euler[2] = math.radians(rotation_angle_list[i] )
#     # cam_empty.rotation_euler[0] = math.radians(elevation_angle_list[i])

#     # # 设置相机位置为 R 和 T 数组中的对应值
#     cam.matrix_world = B_all[i]
    
#     render_file_path = os.path.join(img_follder, '%03d.png' % (i))
#     scene.render.filepath = render_file_path
#     bpy.ops.render.render(write_still=True)
#     # might not need it, but just in case cam is not updated correctly
#     bpy.context.view_layer.update()


    rt = get_3x4_RT_matrix_from_blender(cam)
    pos, rt, scale = cam.matrix_world.decompose()
    rt = rt.to_matrix()


    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0,0,0,1])
    print(matrix)

    to_add = {\
        "file_path":f'{str(i).zfill(3)}.png',
        "transform_matrix":matrix
    }
    frames.append(to_add)

to_export['frames'] = frames
'''
with open(f'{img_follder}/transforms.json', 'w') as f:
    json.dump(to_export, f,indent=4)    
'''