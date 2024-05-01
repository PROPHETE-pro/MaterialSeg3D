import bpy
import argparse
import pdb
import os
import sys


parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--obj_file', type=str)
parser.add_argument(
    '--orm_path', type=str)

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


for file in os.listdir(args.obj_file):
    if file.endswith('.obj'):
        obj_file_path = os.path.join(args.obj_file, file)
    if file.endswith('.png'):
        rgb_uv_path = os.path.join(args.obj_file, file)

output_filepath = obj_file_path.replace('obj', 'glb')

'''
output_folder = "example/car2"
output_filename = "car2.glb"
output_filepath = os.path.join(output_folder, output_filename)
'''

bpy.ops.wm.read_factory_settings(use_empty=True)


#obj_file_path = "example/car2/car2.obj"
bpy.ops.import_scene.obj(filepath=obj_file_path)


selected_objects = bpy.context.selected_editable_objects


for obj in selected_objects:
    
    material_slots = obj.material_slots
    
    
    for mat_slot in material_slots:
        
        material = mat_slot.material
        
        
        material.use_nodes = True
        
        
        bsdf_node = material.node_tree.nodes.get('Principled BSDF')
        uv_image_node1 = material.node_tree.nodes.new('ShaderNodeTexImage')
        uv_image_node2 = material.node_tree.nodes.new('ShaderNodeTexImage')
        output_node = material.node_tree.nodes.get('Material Output')
        
        
        uv_image1_path = rgb_uv_path
        uv_image1 = bpy.data.images.load(uv_image1_path)
        uv_image_node1.image = uv_image1
        material.node_tree.links.new(uv_image_node1.outputs['Color'], bsdf_node.inputs['Base Color'])
        
        
        uv_image2_path = args.orm_path
        uv_image2 = bpy.data.images.load(uv_image2_path)
        uv_image_node2.image = uv_image2
        material.node_tree.links.new(uv_image_node2.outputs['Color'], bsdf_node.inputs['Roughness'])
        
        
        material.node_tree.links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])


bpy.ops.export_scene.gltf(filepath=output_filepath)


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()