import bpy
import argparse
import pdb
import os
import sys


parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--obj_file', type=str)

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

for file in os.listdir(args.obj_file):
    if file.endswith('.obj'):
        obj = os.path.join(args.obj_file, file)
    if file.endswith('.png'):
        uv = os.path.join(args.obj_file, file)

obj_name = args.obj_file.split('/')[-1]

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()


bpy.ops.import_scene.obj(filepath=obj)


bpy.ops.image.open(filepath=uv)


for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        material = bpy.data.materials.new(name='Texture Material')
        material.use_nodes = True
        tree = material.node_tree
        nodes = tree.nodes
        


bpy.ops.export_scene.gltf(filepath=os.path.join(args.obj_file, f'{obj_name}_raw.glb'))


bpy.ops.wm.read_factory_settings()






