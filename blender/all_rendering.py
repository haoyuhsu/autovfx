import pickle
import numpy as np
import os
import sys
import bpy
import math
import shutil
import json
import time
from mathutils import Vector, Matrix
import argparse
import glob
import colorsys
import bmesh
from mathutils.bvhtree import BVHTree


"""
Blender python script for rendering all visual effects.
"""


context = bpy.context
scene = context.scene
render = scene.render


#########################################################
# Ensure all collections and objects are visible        
#########################################################
def ensure_collection_visibility(collection_name):
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        collection.hide_viewport = False  # Ensure collection is visible in the viewport
        collection.hide_render = False    # Ensure collection is enabled for rendering
    else:
        print(f"Collection '{collection_name}' not found.")

def enable_render_for_all_objects():
    for obj in bpy.data.objects:
        obj.hide_viewport = False # Ensure the object is visible in the viewport
        obj.hide_render = False  # Ensure the object is visible in the render

ensure_collection_visibility("Collection") # Ensure default collection is visible and renderable
enable_render_for_all_objects() # Ensure all objects are visible in the render


#########################################################
# Handle duplicate objects (not used)                   
#########################################################
# def duplicate_hierarchy(obj, parent=None):
#     """Recursively duplicate an object and all its children."""
#     # Duplicate the object (without the data)
#     new_obj = obj.copy()
#     # Link the object data if it exists (for meshes, curves, etc.)
#     if new_obj.data:
#         new_obj.data = obj.data.copy()
#     # If a parent is specified, set the duplicated object's parent
#     if parent:
#         new_obj.parent = parent
#     # Link the new object to the collection
#     bpy.context.collection.objects.link(new_obj)
#     # Recursively duplicate children
#     for child in obj.children:
#         duplicate_hierarchy(child, new_obj)
#     return new_obj


# def create_linked_duplicate(object_name: str) -> None:
#     """Creates n linked duplicate of the given object."""
#     original_obj = bpy.data.objects.get(object_name)
#     if original_obj:
#         new_obj = duplicate_hierarchy(original_obj)
#     else:
#         new_obj = None
#         print(f"Object '{object_name}' not found.")
#     return new_obj


#########################################################
# Argument parser for blender: https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
#########################################################
class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())
    

#########################################################
# Blender scene setup
#########################################################
def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def setup_blender_env(img_width, img_height):

    reset_scene()

    # Set render engine and parameters
    render.engine = 'CYCLES'
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = img_width
    render.resolution_y = img_height
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.preview_samples = 64
    scene.cycles.samples = 64  # 32 for testing, 256 or higher 512 for final
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.cycles.film_exposure = 2.0

    # Set the device_type (from Zhihao's code, not sure why specify this)
    preferences = context.preferences
    preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # get_devices() to let Blender detects GPU device
    preferences.addons["cycles"].preferences.get_devices()
    print(preferences.addons["cycles"].preferences.compute_device_type)
    for d in preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])


#########################################################
# Blender camera setup
#########################################################
def create_camera_list(c2w, K):
    """
    Create a list of camera parameters

    Args:
        c2w: (N, 4, 4) camera to world transform
        K: (3, 3) or (N, 3, 3) camera intrinsic matrix
    """
    cam_list = []
    for i in range(len(c2w)):
        pose = c2w[i].reshape(-1, 4)
        if len(K.shape) == 3:
            cam_list.append({'c2w': pose, 'K': K[i]})
        else:
            cam_list.append({'c2w': pose, 'K': K})
    return cam_list


def setup_camera():
    # Find a camera in the scene
    cam = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            cam = obj
            print("found camera")
            break
    # If no camera is found, create a new one
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
    # Set the camera as the active camera for the scene
    bpy.context.scene.camera = cam
    return cam


class Camera():
    def __init__(self, im_height, im_width, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.w = im_width
        self.h = im_height
        self.camera = setup_camera()
        
    def set_camera(self, K, c2w):
        self.K = K       # (3, 3)
        self.c2w = c2w   # (3 or 4, 4), camera to world transform
        # original camera model: x: right, y: down, z: forward (OpenCV, COLMAP format)
        # Blender camera model:  x: right, y: up  , z: backward (OpenGL, NeRF format)
        
        self.camera.data.type = 'PERSP'
        self.camera.data.lens_unit = 'FOV'
        f = K[0, 0]
        rad = 2 * np.arctan(self.w/(2 * f))
        self.camera.data.angle = rad
        self.camera.data.sensor_fit = 'HORIZONTAL'  # 'HORIZONTAL' keeps horizontal right (more recommended)

        # f = K[1, 1]
        # rad = 2 * np.arctan(self.h/(2 * f))
        # self.camera.data.angle = rad
        # self.camera.data.sensor_fit = 'VERTICAL'  # 'VERTICAL' keeps vertical right
        
        self.pose = self.transform_pose(c2w)
        self.camera.matrix_world = Matrix(self.pose)
        
    def transform_pose(self, pose):
        '''
        Transform camera-to-world matrix
        Input:  (3 or 4, 4) x: right, y: down, z: forward
        Output: (4, 4)      x: right, y: up  , z: backward
        '''
        pose_bl = np.zeros((4, 4))
        pose_bl[3, 3] = 1
        # camera position remain the same
        pose_bl[:3, 3] = pose[:3, 3] 
        
        R_c2w = pose[:3, :3]
        transform = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ]) 
        R_c2w_bl = R_c2w @ transform
        pose_bl[:3, :3] = R_c2w_bl
        
        return pose_bl

    def initialize_depth_extractor(self):
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        bpy.context.view_layer.cycles.use_denoising = True
        bpy.context.view_layer.cycles.denoising_store_passes = True
        bpy.context.scene.use_nodes = True

        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        render_layers = nodes['Render Layers']
        depth_file_output = nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.name = 'File Output Depth'
        depth_file_output.format.file_format = 'OPEN_EXR'
        links.new(render_layers.outputs[2], depth_file_output.inputs[0])

    def render_single_timestep_rgb_and_depth(self, cam_info, FRAME_INDEX, dir_name_rgb='rgb', dir_name_depth='depth'):

        dir_path_rgb = os.path.join(self.out_dir, dir_name_rgb)
        dir_path_depth = os.path.join(self.out_dir, dir_name_depth)
        os.makedirs(dir_path_rgb, exist_ok=True)
        os.makedirs(dir_path_depth, exist_ok=True)

        self.set_camera(cam_info['K'], cam_info['c2w'])

        # Set paths for both RGB and depth outputs
        depth_output_path = os.path.join(dir_path_depth, '{:0>3d}'.format(FRAME_INDEX))
        rgb_output_path = os.path.join(dir_path_rgb, '{:0>3d}.png'.format(FRAME_INDEX))

        # Assuming your Blender setup has nodes named accordingly
        bpy.context.scene.render.filepath = rgb_output_path
        bpy.data.scenes["Scene"].node_tree.nodes["File Output Depth"].base_path = depth_output_path

        bpy.ops.render.render(use_viewport=True, write_still=True)


#########################################################
# Blender lighting setup
#########################################################
def add_env_lighting(env_map_path: str, strength: float = 1.0):
    """
    Add environment lighting to the scene with controllable strength.

    Args:
        env_map_path (str): Path to the environment map.
        strength (float): Strength of the environment map.
    """
    # Ensure that we are using nodes for the world's material
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    # Create an environment texture node and load the image
    env = nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(env_map_path)

    # Create a Background node and set its strength
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Strength'].default_value = strength

    # Create an Output node
    out = nodes.new('ShaderNodeOutputWorld')

    # Link nodes together
    links = world.node_tree.links
    links.new(env.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], out.inputs['Surface'])


def add_emitter_lighting(obj: bpy.types.Object, strength: float = 100.0, color=(1, 1, 1)):
    """
    Add an emitter light to the object with controllable strength and color.
    """
    # Create a new material for the object
    mat = bpy.data.materials.new(name='EmitterMaterial')
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Set the material to use nodes
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create an Emission node and set its strength and color
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = strength
    emission.inputs['Color'].default_value = (*color, 1.0)

    # Create an Output node
    out = nodes.new('ShaderNodeOutputMaterial')

    # Link nodes together
    links = mat.node_tree.links
    links.new(emission.outputs['Emission'], out.inputs['Surface'])


def add_sun_lighting(strength: float = 1.0, direction=(0, 0, 1)):
    """
    Add a sun light to the scene with controllable strength and direction.

    Args:
        strength (float): Strength of the sun light.
        direction (tuple): Direction of the sun light.
    """
    sun_name = 'Sun'
    sun = bpy.data.objects.get(sun_name)
    if sun is None:
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 0))
        sun = bpy.context.object
        sun.name = sun_name

    direction = Vector(direction)
    direction.normalize()
    rotation = direction.to_track_quat('Z', 'Y').to_euler()
    sun.rotation_euler = rotation
    sun.data.energy = strength


#########################################################
# Object manipulation
#########################################################
def object_meshes(single_obj):
    for obj in [single_obj] + single_obj.children_recursive:
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_bbox(single_obj=None, ignore_matrix=False):
    bpy.ops.object.select_all(action="DESELECT")
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else object_meshes(single_obj):
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene(single_obj):
    bbox_min, bbox_max = scene_bbox(single_obj)
    scale = 1 / max(bbox_max - bbox_min)
    single_obj.scale = single_obj.scale * scale
    bpy.context.view_layer.update()             # Ensure the scene is fully updated
    bbox_min, bbox_max = scene_bbox(single_obj)
    offset = -(bbox_min + bbox_max) / 2
    single_obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def load_object(object_path: str) -> bpy.types.Object:
    """Loads an object asset into the scene."""
    # import the object
    if object_path.endswith(".glb") or object_path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path, axis_forward='Y', axis_up='Z')
    elif object_path.endswith(".ply"):
        # bpy.ops.import_mesh.ply(filepath=object_path)                             # only used for snap blender
        bpy.ops.wm.ply_import(filepath=object_path, forward_axis='Y', up_axis='Z')  # used for blender 4.0 & snap blender
    elif object_path.endswith(".obj"):
        # bpy.ops.import_scene.obj(filepath=object_path, use_split_objects=False, forward_axis='Y', up_axis='Z')  # only used for snap blender
        bpy.ops.wm.obj_import(filepath=object_path, use_split_objects=False, forward_axis='Y', up_axis='Z')       # used for blender 4.0 & snap blender
    ##### This part is used for ChatSim assets #####
    elif object_path.endswith(".blend"):
        blend_path = object_path
        new_obj_name = 'chatsim_' + blend_path.split('/')[-1].split('.')[0]
        model_obj_name = 'Car'                                                  # general names used for all assets in ChatSim
        with bpy.data.libraries.load(blend_path) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:                                             # actually part that import the object
            if obj.name == model_obj_name:
                bpy.context.collection.objects.link(obj)
        if model_obj_name in bpy.data.objects:                                  # rename the object to avoid conflict
            imported_object = bpy.data.objects[model_obj_name]
            imported_object.name = new_obj_name
            print(f"rename {model_obj_name} to {new_obj_name}")
        for slot in imported_object.material_slots:                             # rename the material to avoid conflict
            material = slot.material
            if material:
                material.name = new_obj_name + "_" + material.name
        return imported_object
    else:
        raise ValueError(f"Unsupported file type: {object_path}")
    new_obj = bpy.context.object
    return new_obj


def merge_meshes(obj):
    """
    Merge all meshes within the object into a single mesh

    Args:
        obj: blender object
    """
    all_object_nodes = [obj] + obj.children_recursive
    mesh_objects = [obj for obj in all_object_nodes if obj.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')     # Deselect all objects first
    for obj in mesh_objects:
        obj.select_set(True)                         # Select each mesh object
        bpy.context.view_layer.objects.active = obj  # Set as active object
    bpy.ops.object.join()
    bpy.ops.object.select_all(action="DESELECT")


def remove_empty_nodes_v1(obj):
    """
    Remove empty nodes in the object (original version)

    Args:
        obj: blender object
    
    Returns:
        obj: blender object with empty nodes removed, only keep the mesh nodes
    """
    all_object_nodes = [obj] + obj.children_recursive
    
    # find the mesh node first
    current_obj = None
    world_matrix = None
    for obj_node in all_object_nodes:
        if obj_node.type == 'MESH':
            current_obj = obj_node      # after merging, only one mesh node left
            world_matrix = obj_node.matrix_world.copy()
            break
    
    # perform removal    
    for obj_node in all_object_nodes:
        if obj_node != current_obj:
            bpy.data.objects.remove(obj_node, do_unlink=True)
            
    # apply world transform back
    current_obj.matrix_world = world_matrix
            
    bpy.ops.object.select_all(action="DESELECT")
    return current_obj


def remove_empty_nodes_v2(obj):
    """
    Remove empty nodes in the object while preserving mesh transformations.
    
    Args:
        obj: Blender object, typically the root of a hierarchy.
    
    Returns:
        Blender object with empty nodes removed, only keeping the mesh nodes.
    """
    all_object_nodes = [obj] + obj.children_recursive
    mesh_node = None

    # Find the first mesh node
    for obj_node in all_object_nodes:
        if obj_node.type == 'MESH':
            mesh_node = obj_node
            break

    if mesh_node:
        # Clear parent to apply transformation, if any
        mesh_node.matrix_world = mesh_node.matrix_local if mesh_node.parent is None else mesh_node.matrix_world
        mesh_node.parent = None

        # Perform removal of other nodes
        for obj_node in all_object_nodes:
            if obj_node != mesh_node:
                bpy.data.objects.remove(obj_node, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    mesh_node.select_set(True)
    bpy.context.view_layer.objects.active = mesh_node

    return mesh_node


def transform_object_origin(obj, set_origin_to_bottom=True):
    """
    Transform object to align with the scene, make the bottom point or center point of the object to be the origin

    Args:
        obj: blender object
    """
    bbox_min, bbox_max = scene_bbox(obj)

    new_origin = np.zeros(3)
    new_origin[0] = (bbox_max[0] + bbox_min[0]) / 2.
    new_origin[1] = (bbox_max[1] + bbox_min[1]) / 2.
    if set_origin_to_bottom:
        new_origin[2] = bbox_min[2]
    else:
        new_origin[2] = (bbox_max[2] + bbox_min[2]) / 2.

    all_object_nodes = [obj] + obj.children_recursive

    ## move the asset origin to the new origin
    for obj_node in all_object_nodes:
        if obj_node.data:
            me = obj_node.data
            mw = obj_node.matrix_world
            matrix = obj_node.matrix_world
            o = Vector(new_origin)
            o = matrix.inverted() @ o
            me.transform(Matrix.Translation(-o))
            mw.translation = mw @ o

    ## move all transform to origin (no need to since Empty objects have all been removed)
    # for obj_node in all_object_nodes:
    #    obj_node.matrix_world.translation = [0, 0, 0]
    #    obj_node.rotation_quaternion = [1, 0, 0, 0]

    bpy.ops.object.select_all(action="DESELECT")


def rotate_obj(obj, R):
    """
    Apply rotation matrix to blender object

    Args:
        obj: blender object
        R: (3, 3) rotation matrix
    """
    R = Matrix(R)
    obj.rotation_mode = 'QUATERNION'
    
    # Combine the rotations by matrix multiplication
    current_rotation = obj.rotation_quaternion.to_matrix().to_3x3()
    new_rotation_matrix = R @ current_rotation
    
    # Convert back to a quaternion and apply to the object
    obj.rotation_quaternion = new_rotation_matrix.to_quaternion()


def get_object_center_to_bottom_offset(obj):
    """
    Get the offset from the center to the bottom of the object

    Args:
        obj: blender object

    Returns:
        offset: (3,) offset
    """
    bbox_min, bbox_max = scene_bbox(obj)
    bottom_pos = np.zeros(3)
    bottom_pos[0] = (bbox_max[0] + bbox_min[0]) / 2.
    bottom_pos[1] = (bbox_max[1] + bbox_min[1]) / 2.
    bottom_pos[2] = bbox_min[2]
    offset = np.array(obj.location) - bottom_pos
    return offset


def insert_object(obj_path, pos, rot, scale=1.0, from_3DGS=False):
    """
    Insert object into the scene

    Args:
        obj_path: path to the object
        pos: (3,) position
        rot: (3, 3) rotation matrix
        scale: scale of the object

    Returns:
        inserted_obj: blender object
    """
    inserted_obj = load_object(obj_path)
    merge_meshes(inserted_obj)
    inserted_obj = remove_empty_nodes_v1(inserted_obj)
    # inserted_obj = remove_empty_nodes_v2(inserted_obj)
    if not from_3DGS and not inserted_obj.name.startswith('chatsim'):
        normalize_scene(inserted_obj)
    transform_object_origin(inserted_obj, set_origin_to_bottom=False)  # set origin to the center for simulation
    inserted_obj.scale *= scale
    rotate_obj(inserted_obj, rot)
    bpy.context.view_layer.update()
    ## object origin is at center and pos represents the contact point, so we need to adjust the object position
    ## NOTE: we might have problem when rotation on either x or y axis is not 0
    if True:
        offset = get_object_center_to_bottom_offset(inserted_obj)
        inserted_obj.location = pos + offset
    else:
        inserted_obj.location = pos   # TODO: 3DGS might also adopt bottom point as position
    inserted_obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # apply transform to allow simulation
    if from_3DGS:
        bpy.ops.object.shade_smooth()  # smooth shading for 3DGS objects
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    return inserted_obj


def insert_animated_object(obj_path, pos, rot, scale=1.0):
    """
    Insert animated object into the scene

    Args:
        obj_path: path to the object
        pos: (3,) position
        rot: (3, 3) rotation matrix
        scale: scale of the object

    Returns:
        inserted_obj: blender object
    """
    inserted_obj = load_object(obj_path)
    merge_meshes(inserted_obj)
    if not inserted_obj.name.startswith('chatsim'):
        normalize_scene(inserted_obj)
    inserted_obj.scale *= scale
    rotate_obj(inserted_obj, rot)
    bpy.context.view_layer.update()
    ## object origin is at center and pos represents the contact point, so we need to adjust the object position
    ## NOTE: we might have problem when rotation on either x or y axis is not 0
    offset = get_object_center_to_bottom_offset(inserted_obj) if not inserted_obj.name.startswith('chatsim') else np.zeros(3)
    inserted_obj.location = pos + offset
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    return inserted_obj


def get_geometry_proxy(obj, voxel_size=0.01):
    """
    Remesh the object with voxel size

    Args:
        obj: blender object
        voxel_size: voxel size
    """
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # duplicate
    proxy_obj = obj.copy()
    proxy_obj.data = obj.data.copy()
    proxy_obj.name = obj.name + "_proxy"
    bpy.context.collection.objects.link(proxy_obj)
    # apply remesh modifier
    remesh_mod = proxy_obj.modifiers.new(name="Remesh", type='REMESH')
    remesh_mod.mode = 'VOXEL'
    remesh_mod.voxel_size = voxel_size
    bpy.ops.object.modifier_apply(modifier=remesh_mod.name)
    # Remove any existing rigid body physics from the proxy object
    if proxy_obj.rigid_body:
        bpy.context.view_layer.objects.active = proxy_obj
        bpy.ops.rigidbody.object_remove()
    # Option1: add a Copy Transforms constraint
    copy_transforms = proxy_obj.constraints.new(type='COPY_TRANSFORMS')
    copy_transforms.target = obj
    # Option2: parent the proxy object to the original object
    # proxy_obj.parent = obj
    # proxy_obj.matrix_parent_inverse = obj.matrix_world.inverted()
    proxy_obj.hide_render = True   # avoid rendering the proxy object
    bpy.ops.object.select_all(action="DESELECT")
    return proxy_obj


#########################################################
# Shadow catcher setup
#########################################################
def add_meshes_shadow_catcher(mesh_path=None, is_uv_mesh=False):
    """
    Add entire scene meshes as shadow catcher to the scene

    Args:
        mesh_path: path to the mesh file
        is_uv_mesh: whether the mesh is a UV textured mesh
    """
    # add meshes extracted from NeRF/3DGS as shadow catcher
    if mesh_path is None or not os.path.exists(mesh_path):
        AssertionError('meshes file does not exist')
    mesh = load_object(mesh_path)
    # mesh.is_shadow_catcher = True   # set True for transparent shadow catcher
    mesh.visible_diffuse = False      # prevent meshes light up the scene

    if not is_uv_mesh:
        mesh.visible_glossy = False   # prevent white material from reflecting light
        white_mat = create_white_material()
        if mesh.data.materials:
            mesh.data.materials[0] = white_mat
        else:
            mesh.data.materials.append(white_mat)

    bpy.ops.object.select_all(action="DESELECT")
    return mesh


def add_planar_shadow_catcher(size=10):
    """
    Add a large planar surface as shadow catcher to the scene

    Args:
        size: size of the planar surface
    """
    bpy.ops.mesh.primitive_plane_add(size=1)
    mesh = bpy.context.object

    mesh.scale = (size, size, 1)
    mesh.name =  "floor_plane"

    mesh.visible_glossy = False   # prevent white material from reflecting light
    white_mat = create_white_material()
    if mesh.data.materials:
        mesh.data.materials[0] = white_mat
    else:
        mesh.data.materials.append(white_mat)

    bpy.ops.object.select_all(action="DESELECT")
    return mesh


#########################################################
# Rigid body simulation
#########################################################
def add_rigid_body(obj, rb_type='ACTIVE', collision_shape='MESH', mass=1.0, restitution=0.6, collision_margin=0.001):
    """
    Add rigid body to the object

    Args:
        obj: blender object
        mass: mass of the object
        collision_shape: collision shape of the object
    """
    all_obj_nodes = [obj] + obj.children_recursive
    for obj_node in all_obj_nodes:
        if obj_node.type == 'MESH':
            bpy.context.view_layer.objects.active = obj_node
            bpy.ops.rigidbody.object_add()
            if rb_type == 'KINEMATIC':
                obj_node.rigid_body.type = 'PASSIVE'
                obj_node.rigid_body.kinematic = True
            else:
                obj_node.rigid_body.type = rb_type
            obj_node.rigid_body.collision_shape = collision_shape
            obj_node.rigid_body.restitution = restitution
            obj_node.rigid_body.mass = mass
            obj_node.rigid_body.collision_margin = collision_margin
            # obj_node.rigid_body.friction = 0.8   # TODO: lead to blender crash (not sure why)
            bpy.ops.object.select_all(action="DESELECT")


#########################################################
# Animation
#########################################################
# def set_linear_trajectory(blender_obj, points, t1=1, t2=-1, f_axis='TRACK_NEGATIVE_Y'):
#     """
#     Set a trajectory of an object from multiple points (use Bezier curve)
#     """

#     curve_data = bpy.data.curves.new('myCurve', type='CURVE')
#     curve_data.dimensions = '3D'
#     curve_data.path_duration = max(t2 - t1, 1)
#     curve_data.use_path = True
#     curve_data.eval_time = 0
#     curve_data.keyframe_insert(data_path="eval_time", frame=t1)
#     curve_data.eval_time = max(t2 - t1, 1)
#     curve_data.keyframe_insert(data_path="eval_time", frame=t2)

#     n_points = len(points)

#     # TODO Test: chatsim objects are centered at the origin, but the points is on the floor, so we need to adjust the offset
#     # offset = get_object_center_to_bottom_offset(blender_obj) if blender_obj.name.startswith('chatsim') else np.zeros(3)
    
#     spline = curve_data.splines.new(type='BEZIER')
#     spline.use_endpoint_u = True
#     spline.use_endpoint_v = True
#     spline.bezier_points.add(n_points - 1)  # already has one point by default
#     for i, point in enumerate(points):
#         spline.bezier_points[i].co = point
#         spline.bezier_points[i].handle_left_type = spline.bezier_points[i].handle_right_type = 'AUTO'
    
#     curve_obj = bpy.data.objects.new('MyCurveObject', curve_data)
#     scene.collection.objects.link(curve_obj)

#     follow_path_constraint = blender_obj.constraints.new(type='FOLLOW_PATH')
#     follow_path_constraint.target = curve_obj
#     follow_path_constraint.use_curve_follow = True
#     follow_path_constraint.forward_axis = f_axis
#     follow_path_constraint.up_axis = 'UP_Z'

#     # set blender obj location and rotation to zero
#     blender_obj.location = (0, 0, 0)
#     blender_obj.rotation_euler = (0, 0, 0)

#     bpy.context.view_layer.update()
            

def set_linear_trajectory(blender_obj, points, t1=1, t2=-1, f_axis='TRACK_NEGATIVE_Y'):
    """
    Set a trajectory of an object from multiple points (use Poly curve)
    """

    curve_data = bpy.data.curves.new('myCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.path_duration = max(t2 - t1, 1)
    curve_data.use_path = True
    curve_data.eval_time = 0
    curve_data.keyframe_insert(data_path="eval_time", frame=t1)
    curve_data.eval_time = max(t2 - t1, 1)
    curve_data.keyframe_insert(data_path="eval_time", frame=t2)

    n_points = len(points)

    # TODO Test: chatsim objects are centered at the origin, but the points is on the floor, so we need to adjust the offset
    # offset = get_object_center_to_bottom_offset(blender_obj) if blender_obj.name.startswith('chatsim') else np.zeros(3)

    spline = curve_data.splines.new(type='POLY')
    spline.points.add(n_points - 1)  # already has one point by default
    for i, point in enumerate(points):
        spline.points[i].co = (point[0], point[1], point[2], 1)
        # spline.points[i].co = (point[0]+offset[0], point[1]+offset[1], point[2]+offset[2], 1)
    
    curve_obj = bpy.data.objects.new('MyCurveObject', curve_data)
    scene.collection.objects.link(curve_obj)

    follow_path_constraint = blender_obj.constraints.new(type='FOLLOW_PATH')
    follow_path_constraint.target = curve_obj
    follow_path_constraint.use_curve_follow = True
    follow_path_constraint.forward_axis = f_axis
    follow_path_constraint.up_axis = 'UP_Z'

    # set blender obj location and rotation to zero
    blender_obj.location = (0, 0, 0)
    blender_obj.rotation_euler = (0, 0, 0)

    bpy.context.view_layer.update()


def extend_cyclic_animation_command_line(obj, n_repetitions_after=0, n_repetitions_before=0):
    """
    Extend the cyclic animation of the object
    """
    if obj.animation_data is not None and obj.animation_data.action is not None:
        for fcurves_f in obj.animation_data.action.fcurves:
            new_modifier = fcurves_f.modifiers.new(type='CYCLES')
            new_modifier.cycles_after = n_repetitions_after  # 0 means infinite repetitions after the end of the fcurves_f
            new_modifier.cycles_before = n_repetitions_before # 0 means infinite repetitions before the start of the fcurves_f
            # new_modifier.frame_start = fcurves_f.range()[0]
            # new_modifier.frame_end = fcurves_f.range()[1]


def add_cyclic_animation(obj, n_repetitions_after=0, n_repetitions_before=0):
    """
    Add cyclic animation to the object
    """
    all_obj_nodes = [obj] + obj.children_recursive
    for obj_node in all_obj_nodes:
        extend_cyclic_animation_command_line(obj_node, n_repetitions_after, n_repetitions_before)


#########################################################
# Materials
#########################################################
def create_white_material():
    mat = bpy.data.materials.new(name="WhiteMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Metallic"].default_value = 0.0
    # bsdf.inputs["Specular"].default_value = 0.0  # issue: https://github.com/ross-g/io_pdx_mesh/issues/86
    bsdf.inputs[7].default_value = 0.0   # Specular
    bsdf.inputs["Roughness"].default_value = 1.0
    return mat


def get_mesh_nodes(obj):
    """
    Get all mesh nodes in the object

    Args:
        obj: blender object
    
    Returns:
        mesh_nodes: list of blender mesh objects
    """
    all_object_nodes = [obj] + obj.children_recursive
    mesh_nodes = []
    for obj_node in all_object_nodes:
        if obj_node.type == 'MESH':
            mesh_nodes.append(obj_node)
    return mesh_nodes


def get_material_nodes(mesh_nodes):
    """
    Get the material nodes (PRINCIPLE_BSDF & TEXTURE) of the object
    """
    principled_nodes = []
    base_color_nodes = []
    material_nodes = []
    for mesh in mesh_nodes:
        for material_slot in mesh.material_slots:
            material = material_slot.material
            if material and material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        principled_nodes.append(node)
                    elif node.type == 'TEX_IMAGE':
                        base_color_nodes.append(node)
                material_nodes.append(material)
    return principled_nodes, base_color_nodes, material_nodes


def adjust_principled_bsdf(principled_nodes, specular=0.5, metallic=0.0, roughness=0.5):
    """
    Adjust the specular, metallic and roughness value of the material
    """
    for node in principled_nodes:
        if node.inputs['Specular'].is_linked:
            node.id_data.links.remove(node.inputs['Specular'].links[0])
        if node.inputs['Metallic'].is_linked:
            node.id_data.links.remove(node.inputs['Metallic'].links[0])
        if node.inputs['Roughness'].is_linked:
            node.id_data.links.remove(node.inputs['Roughness'].links[0])
        node.inputs['Specular'].default_value = specular
        node.inputs['Metallic'].default_value = metallic
        node.inputs['Roughness'].default_value = roughness


def make_mirror_material(principled_nodes):
    """
    Make the material to be mirror material
    """
    for node in principled_nodes:
        if node.inputs['Base Color'].is_linked:
            node.id_data.links.remove(node.inputs['Base Color'].links[0])
        node.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)   # set the base color to white
    adjust_principled_bsdf(principled_nodes, specular=1.0, metallic=1.0, roughness=0.0)


def change_texture_image(base_color_nodes, texture_image_path):
    """
    Change the texture image of the material
    """
    texture_image = bpy.data.images.load(texture_image_path)
    for node in base_color_nodes:
        node.image = texture_image


def change_materials(obj, material_folder):
    """
    Change the material nodes of the object
    """
    folder_name = os.path.basename(material_folder)
    texture_folder = os.path.join(material_folder, folder_name + '_1k', 'textures')
    base_color_path = glob.glob(os.path.join(texture_folder, '*diff*'))[0]
    normal_path = glob.glob(os.path.join(texture_folder, '*nor_gl*'))[0]
    displacement_path = glob.glob(os.path.join(texture_folder, '*disp*'))[0]
    roughness_path = glob.glob(os.path.join(texture_folder, '*rough*'))[0]

    mat = bpy.data.materials.new(name='PolyHavenMaterial')
    mat.use_nodes = True
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create nodes
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    texture_base_color = nodes.new('ShaderNodeTexImage')
    texture_normal = nodes.new('ShaderNodeTexImage')
    texture_displacement = nodes.new('ShaderNodeTexImage')
    texture_roughness = nodes.new('ShaderNodeTexImage')
    normal_map = nodes.new('ShaderNodeNormalMap')
    displacement = nodes.new('ShaderNodeDisplacement')
    material_output = nodes.new('ShaderNodeOutputMaterial')

    # Set the image
    texture_base_color.image = bpy.data.images.load(base_color_path)
    texture_normal.image = bpy.data.images.load(normal_path)
    texture_displacement.image = bpy.data.images.load(displacement_path)
    texture_roughness.image = bpy.data.images.load(roughness_path)

    # Link nodes
    links = mat.node_tree.links
    links.new(texture_base_color.outputs['Color'], principled.inputs['Base Color'])
    links.new(texture_normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    links.new(texture_displacement.outputs['Color'], displacement.inputs['Height'])
    links.new(displacement.outputs['Displacement'], material_output.inputs['Displacement'])
    links.new(texture_roughness.outputs['Color'], principled.inputs['Roughness'])
    links.new(principled.outputs['BSDF'], material_output.inputs['Surface'])


def rgb_to_hsv(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    return h, s, v


def hsv_to_rgb(h, s, v):
    h /= 360.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    r *= 255
    g *= 255
    b *= 255
    return r, g, b


def change_texture_color(base_color_nodes, target_color_rgb, obj=None, move_ratio=0.8):
    """
    Modify the color of texture image
    """
    target_color_hue = rgb_to_hsv(target_color_rgb[0], target_color_rgb[1], target_color_rgb[2])[0]

    # compute mean hue of the whole asset
    hue_values = []
    node_info = []
    for node in base_color_nodes:
        rgb_pixels = np.asarray(node.image.pixels).reshape((-1, 4))
        hsv_pixels = np.zeros((rgb_pixels.shape[0], 3))
        # convert rgb to hsv
        for i in range(rgb_pixels.shape[0]):
            r, g, b, a = rgb_pixels[i]
            if a == 0:
                continue
            # h, s, v = rgb_to_hsv(r, g, b)
            h, s, v = rgb_to_hsv(r * 255, g * 255, b * 255)
            hsv_pixels[i] = [h, s, v]
        hue_values.append(hsv_pixels[:, 0])
        node_info.append({
            'name': node.name,
            'rgb_pixels': rgb_pixels,
            'hsv_pixels': hsv_pixels
        })
    hue_values = np.concatenate(hue_values)
    mean_hue = np.mean(hue_values)

    if obj:
        mean_rgb = compute_average_color_per_vertex(obj) * 255
        mean_hue = rgb_to_hsv(mean_rgb[0], mean_rgb[1], mean_rgb[2])[0]

    hue_offset = move_ratio * (target_color_hue - mean_hue)

    # overwrite the hue offset to whole asset
    for idx, node in enumerate(base_color_nodes):
        rgb_pixels = node_info[idx]['rgb_pixels']
        hsv_pixels = node_info[idx]['hsv_pixels']
        hsv_pixels[:, 0] += hue_offset
        # convert hsv to rgb
        for i in range(rgb_pixels.shape[0]):
            h, s, v = hsv_pixels[i]
            h = h % 360
            r, g, b = hsv_to_rgb(h, s, v)
            # rgb_pixels[i] = [r, g, b, rgb_pixels[i, 3]]
            rgb_pixels[i] = [r / 255, g / 255, b / 255, rgb_pixels[i, 3]]
        node.image.pixels = rgb_pixels.reshape(-1)
        # ensure the image is updated
        node.image.update()
        if node.image.filepath:
            node.image.save()


def apply_material_to_object(obj, material_info):
    """
    Apply material to the object
    """
    mesh_nodes = get_mesh_nodes(obj)
    principled_nodes, base_color_nodes, mat_nodes = get_material_nodes(mesh_nodes)
    if material_info['is_mirror']:
        make_mirror_material(principled_nodes)
        return
    if material_info['material_path'] is not None:
        change_materials(obj, material_info['material_path'])
        return
    if material_info['rgb'] is not None:
        print("Apply color {} to object".format(material_info['rgb']))
        if len(base_color_nodes) > 0:
            change_texture_color(base_color_nodes, material_info['rgb'], obj)
        else:
            for node in principled_nodes:
                node.inputs['Base Color'].default_value = (*material_info['rgb'], 1)
        return
    adjust_principled_bsdf(principled_nodes, material_info['specular'], material_info['metallic'], material_info['roughness'])


def get_mean_color(base_color_nodes):
    """
    Get the mean color of texture images from an object
    """
    rgb_pixels_list = []
    for node in base_color_nodes:
        rgb_pixels = np.asarray(node.image.pixels).reshape((-1, 4))
        rgb_pixels = rgb_pixels[:, :3]
        rgb_pixels_list.append(rgb_pixels)
    rgb_pixels = np.concatenate(rgb_pixels_list, axis=0)
    mean_rgb = np.mean(rgb_pixels, axis=0)
    return mean_rgb


def compute_average_color_per_vertex(obj):
    """
    Compute the average color of the vertices of an mesh object
    """
    mesh_nodes = get_mesh_nodes(obj)
    color_accum = np.zeros(4)
    n_vertices = 0

    for mesh in mesh_nodes:
        for material_slot in mesh.material_slots:
            material = material_slot.material
            if material and material.use_nodes:
                uv_layer = mesh.data.uv_layers.active.data

                # principle_bsdf = material.node_tree.nodes['Principled BSDF']
                principle_bsdf = None
                for node in material.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        principle_bsdf = node
                        break

                if principle_bsdf:
                    # get which text_image node is connected to the base color
                    img = None
                    for link in principle_bsdf.inputs['Base Color'].links:
                        if link.from_node.type == 'TEX_IMAGE':
                            img = link.from_node.image
                            break
                    
                    if img:      
                        img_width, img_height = img.size
                        num_channels = img.channels
                        pixels = np.array(img.pixels)

                        for poly in mesh.data.polygons:
                            for loop_index in poly.loop_indices:
                                loop = mesh.data.loops[loop_index]
                                uv = uv_layer[loop.index].uv
                                x = int(uv.x * (img_width - 1))
                                y = int(uv.y * (img_height - 1))
                                pixel_position = (y * img_width + x) * num_channels
                                # Check for pixel data length to avoid broadcasting errors
                                if len(pixels[pixel_position:pixel_position + num_channels]) == num_channels:
                                    color_accum += pixels[pixel_position:pixel_position + num_channels]
                                    n_vertices += 1

    mean_color = color_accum / n_vertices
    return mean_color


#########################################################
# Smoke & Fire simulation
#########################################################
def add_smoke(blender_obj, with_fire=False, domain_res=128, domain_height=None, cache_dir=None, start_frame=1, end_frame=100, remesh_voxel_size=0.01):
    # Deselect (might not be needed)
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select object and make it active (active might not be needed)
    if blender_obj.type != 'MESH':
        all_object_nodes = [blender_obj] + blender_obj.children_recursive
        for obj_node in all_object_nodes:
            if obj_node.type == 'MESH':
                blender_obj = obj_node
                break
    blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = blender_obj

    # If the object has too much faces, it will decrease the performance of the simulation
    N_FACES_THRESHOLD = 20000
    if len(blender_obj.data.polygons) > N_FACES_THRESHOLD:
        print("The object has too much faces, it will decrease the performance of the simulation")
        proxy_obj = get_geometry_proxy(blender_obj, remesh_voxel_size)
        smoke_proxy_obj_dict[blender_obj.name] = proxy_obj
    else:
        proxy_obj = blender_obj
    bpy.ops.object.select_all(action='DESELECT')
    proxy_obj.select_set(True)
    bpy.context.view_layer.objects.active = proxy_obj

    # Add smoke, the new smoke domain will become selected
    bpy.ops.object.quick_smoke()
    smoke_domain = bpy.context.object

    # if needed, make the domain taller (the scale[2] is half of the height)
    if domain_height is not None:
        s = domain_height / 2 / smoke_domain.scale[2]
        bpy.ops.transform.resize(value=(1, 1, s))  
        bpy.ops.transform.translate(value=(0, 0, smoke_domain.scale[2] * (s - 1)/2))

    ##### Setup smoke domain settings #####
    settings = smoke_domain.modifiers["Fluid"].domain_settings

    settings.resolution_max = domain_res
    if cache_dir is not None:
        settings.cache_type = 'ALL'
        settings.cache_directory = cache_dir

    # adaptive domain + dissolve smoke + fluid noise
    settings.use_adaptive_domain = True
    settings.additional_res = domain_res # additional resolution
    settings.adapt_margin = 4            # avoid boundary issue
    settings.adapt_threshold = 0.01      # remove empty space
    settings.use_dissolve_smoke = True
    settings.dissolve_speed = 30         # number of frames to dissolve smoke
    settings.use_noise = True                           # enable fluid noise
    settings.noise_scale = 2                            # noise scale (from Infinigen)
    settings.noise_strength = np.random.uniform(0.5, 1) # noise strength (from Infinigen)
    settings.vorticity = np.random.uniform(0, 0.1)      # vorticity (from Infinigen)

    # absorb (from Infinigen)
    settings.use_collision_border_back = False
    settings.use_collision_border_bottom = False
    settings.use_collision_border_front = False
    settings.use_collision_border_left = False
    settings.use_collision_border_right = False
    settings.use_collision_border_top = False

    if with_fire:
        settings.flame_vorticity = np.random.uniform(0.45, 0.55) # flame vorticity (from Infinigen)
        settings.burning_rate = np.random.uniform(0.5, 0.8)      # burning rate (from Infinigen)

    ##### Setup smoke domain material #####
    # Create a unique material name for this smoke domain
    unique_material_name = f"{proxy_obj.name}_SmokeDomainMaterial"
    bpy.data.materials["Smoke Domain Material"].name = unique_material_name

    mat_volume_node = bpy.data.materials[unique_material_name].node_tree.nodes["Principled Volume"]
    mat_volume_node.inputs[0].default_value = (0.1, 0.1, 0.1, 1)  # smoke color
    mat_volume_node.inputs[2].default_value = 30.  # smoke density (original 70.)

    if with_fire:
        mat_volume_node.inputs[2].default_value = 8.0  # smoke density (original 8.0)
        mat_volume_node.inputs[8].default_value = 1.5  # blackbody intensity (original 1.5)
        mat_volume_node.inputs[9].default_value = (1, 0.388643, 0.00941088, 1)  # blackbody tint
        mat_volume_node.inputs[10].default_value = 1500  # temperature

        # Additional fire modifier from Infinigen
        obj_smoke_node_tree = bpy.data.materials[unique_material_name].node_tree
        fire_modifier_infinigen(obj_smoke_node_tree)

        # Additional fire modifier from YouTube tutorial
        # obj_smoke_node_tree = bpy.data.materials[unique_material_name].node_tree
        # fire_modifier_youtube(obj_smoke_node_tree)

    smoke_domain.data.materials[0] = bpy.data.materials.get(unique_material_name)

    ##### Setup smoke inflow settings #####
    smoke_inflow = proxy_obj.modifiers["Fluid"].flow_settings
    smoke_inflow.flow_type = 'BOTH'  # fire makes smoke expand faster
    smoke_inflow.flow_behavior = 'INFLOW'
    smoke_inflow.fuel_amount = 1.5
    # smoke_inflow.subframes = 3  # avoid discontinuity of fire in movement (careful in use, might have fire/smoke overshoot)
    smoke_inflow.use_initial_velocity = True

    # Insert keyframes for the smoke inflow (control the smoke intensity over time)
    smoke_inflow.fuel_amount = 0.0
    smoke_inflow.keyframe_insert(data_path='fuel_amount', frame=0)
    smoke_inflow.fuel_amount = 1.5
    smoke_inflow.keyframe_insert(data_path='fuel_amount', frame=start_frame)
    smoke_inflow.fuel_amount = 0.0
    smoke_inflow.keyframe_insert(data_path='fuel_amount', frame=end_frame)
    # Apply exponential interpolation to the keyframes of fuel_amount to slow down the decline of smoke
    for fcurve in proxy_obj.animation_data.action.fcurves:
        if fcurve.data_path == 'modifiers["Fluid"].flow_settings.fuel_amount':
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'EXPO'
                # Verbose for debugging
                # print("===== Set interpolation to EXPO =====")
                # print(keyframe.co[0], keyframe.co[1])

    # Insert keyframes for use_inflow (control the start and end of the smoke)
    smoke_inflow.use_inflow = False
    smoke_inflow.keyframe_insert(data_path='use_inflow', frame=0)
    smoke_inflow.use_inflow = True
    smoke_inflow.keyframe_insert(data_path='use_inflow', frame=start_frame)
    smoke_inflow.use_inflow = False
    smoke_inflow.keyframe_insert(data_path='use_inflow', frame=end_frame)

    # gradually change the color of the blender_obj to black while the fire is burning
    mat_obj = blender_obj.active_material
    if mat_obj is None:
        fire_burn_material_name = f"{blender_obj.name}_FireBurnMaterial"
        mat_obj = bpy.data.materials.new(name=fire_burn_material_name)
        blender_obj.data.materials.append(mat_obj)
    if not mat_obj.use_nodes:
        mat_obj.use_nodes = True
    nodes = mat_obj.node_tree.nodes
    links = mat_obj.node_tree.links
    principled_bsdf = nodes.get('Principled BSDF')
    if not principled_bsdf:
        principled_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        links.new(principled_bsdf.outputs['BSDF'], nodes.get('Material Output').inputs['Surface'])
    texture_nodes = [node for node in nodes if node.type == 'TEX_IMAGE']
    if texture_nodes:
        for texture_node in texture_nodes:
            mix_rgb_node = nodes.new('ShaderNodeMixRGB')
            mix_rgb_node.blend_type = 'MIX'
            mix_rgb_node.inputs['Color2'].default_value = (0.1, 0.1, 0.1, 1)  # black color
            links.new(texture_node.outputs['Color'], mix_rgb_node.inputs['Color1'])
            links.new(mix_rgb_node.outputs['Color'], principled_bsdf.inputs['Base Color'])
            # Insert keyframes for the mix factor
            mix_rgb_node.inputs['Fac'].default_value = 0.0
            mix_rgb_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=0)
            mix_rgb_node.inputs['Fac'].default_value = 0.0
            mix_rgb_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=start_frame)
            mix_rgb_node.inputs['Fac'].default_value = 1.0
            mix_rgb_node.inputs['Fac'].keyframe_insert(data_path='default_value', frame=end_frame)
    else:
        principled_bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
        principled_bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=0)
        principled_bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
        principled_bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=start_frame)
        principled_bsdf.inputs['Base Color'].default_value = (0.1, 0.1, 0.1, 1)
        principled_bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=end_frame)

    return smoke_domain


def fire_modifier_infinigen(node_tree):
    """
    Add fire modifier to the node tree (references: https://github.com/princeton-vl/infinigen/blob/f5567d2/infinigen/assets/materials/blackbody_shader.py)
    Note: this will take plenty of time to render
    """
    nodes = node_tree.nodes
    links = node_tree.links

    # Clear all default nodes
    for node in nodes:
        nodes.remove(node)

    # Add Volume Info Node
    volume_info = nodes.new(type="ShaderNodeVolumeInfo")

    # Add ColorRamp Node (Flame)
    from numpy.random import normal as N
    colorramp = nodes.new(type="ShaderNodeValToRGB")
    colorramp.color_ramp.interpolation = 'B_SPLINE'
    colorramp.color_ramp.elements.new(0)
    colorramp.color_ramp.elements[0].position = 0.2455 + 0.01 * N()
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.2818 + 0.01 * N()
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    colorramp.color_ramp.elements[2].position = 0.5864 + 0.01 * N()
    colorramp.color_ramp.elements[2].color = (0.0, 0.0, 0.0, 1.0)

    # Add ColorRamp Node (Density)
    colorramp_1 = nodes.new(type="ShaderNodeValToRGB")
    colorramp_1.color_ramp.interpolation = 'B_SPLINE'
    colorramp_1.color_ramp.elements[0].position = 0.3636 + 0.01 * N()
    colorramp_1.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.6409 + 0.01 * N()
    colorramp_1.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)

    # Add Math Node (Multiply)
    multiply = nodes.new(type="ShaderNodeMath")
    multiply.operation = 'MULTIPLY'
    multiply_1 = nodes.new(type="ShaderNodeMath")
    multiply_1.operation = 'MULTIPLY'
    # multiply_1.inputs[1].default_value = 8626.6650 + 20 * N()  # this may be too bright
    # multiply_1.inputs[1].default_value = 1000.0 + 20 * N()
    multiply_1.inputs[1].default_value = 150.0 + 20 * N()

    # Add Principled Volume Node
    principled_volume = nodes.new(type="ShaderNodeVolumePrincipled")
    # principled_volume.inputs['Color'].default_value = np.random.uniform(0.2568, 0.4568, 4)
    # principled_volume.inputs['Density'].default_value = 15.0000 + N()

    # our default principled volume settings
    principled_volume.inputs[0].default_value = (0.1, 0.1, 0.1, 1)
    principled_volume.inputs[2].default_value = 8.0  # smoke density (original 8.0)
    principled_volume.inputs[9].default_value = (1, 0.388643, 0.00941088, 1)  # blackbody tint
    principled_volume.inputs[10].default_value = 1500  # temperature

    # Add Material Output Node
    material_output = nodes.new(type="ShaderNodeOutputMaterial")

    # Link Nodes
    links.new(volume_info.outputs['Flame'], colorramp.inputs['Fac'])
    links.new(volume_info.outputs['Density'], colorramp_1.inputs['Fac'])
    links.new(colorramp.outputs['Color'], multiply.inputs[0])
    links.new(colorramp_1.outputs['Color'], multiply.inputs[1])
    links.new(multiply.outputs[0], multiply_1.inputs[0])
    links.new(multiply_1.outputs[0], principled_volume.inputs['Blackbody Intensity'])
    links.new(principled_volume.outputs['Volume'], material_output.inputs['Volume'])


# def fire_modifier_youtube(node_tree):
#     """
#     Add fire modifier to the node tree (references: https://www.youtube.com/watch?v=zyIJQHlFQs0)
#     """
#     # Create nodes
#     mat_volume_node = node_tree.nodes['Principled Volume']
#     volume_info_node = node_tree.nodes.new(type='ShaderNodeVolumeInfo')
#     color_ramp_node_A = node_tree.nodes.new(type='ShaderNodeValToRGB')
#     color_ramp_node_B = node_tree.nodes.new(type='ShaderNodeValToRGB')
#     multiply_node_1 = node_tree.nodes.new(type='ShaderNodeMath')
#     multiply_node_1.operation = 'MULTIPLY'
#     multiply_node_2 = node_tree.nodes.new(type='ShaderNodeMath')
#     multiply_node_2.operation = 'MULTIPLY'
#     multiply_node_2.inputs[1].default_value = 10
#     # Configure Color Ramp node A
#     color_ramp_node_A.color_ramp.interpolation = 'B_SPLINE'
#     color_ramp_node_A.color_ramp.elements[0].position = 0.727
#     color_ramp_node_A.color_ramp.elements[0].color = (1, 1, 1, 1)
#     color_ramp_node_A.color_ramp.elements[1].position = 0.903
#     color_ramp_node_A.color_ramp.elements[1].color = (0, 0, 0, 1)
#     # Configure Color Ramp node B
#     color_ramp_node_B.color_ramp.interpolation = 'B_SPLINE'
#     color_ramp_node_B.color_ramp.elements.new(0.250)
#     color_ramp_node_B.color_ramp.elements.new(0.542)
#     color_ramp_node_B.color_ramp.elements[0].position = 0.218
#     color_ramp_node_B.color_ramp.elements[0].color = (0, 0, 0, 1)
#     color_ramp_node_B.color_ramp.elements[1].position = 0.250
#     color_ramp_node_B.color_ramp.elements[1].color = (1, 1, 1, 1)
#     color_ramp_node_B.color_ramp.elements[2].position = 0.542
#     color_ramp_node_B.color_ramp.elements[2].color = (0, 0, 0, 1)
#     # Connect nodes
#     node_tree.links.new(volume_info_node.outputs['Density'], color_ramp_node_A.inputs['Fac'])
#     node_tree.links.new(volume_info_node.outputs['Flame'], color_ramp_node_B.inputs['Fac'])
#     node_tree.links.new(color_ramp_node_B.outputs['Color'], multiply_node_1.inputs[0])
#     node_tree.links.new(color_ramp_node_A.outputs['Color'], multiply_node_1.inputs[1])
#     node_tree.links.new(multiply_node_1.outputs['Value'], multiply_node_2.inputs[0])
#     node_tree.links.new(multiply_node_2.outputs['Value'], mat_volume_node.inputs['Blackbody Intensity'])


#########################################################
# Fracture simulation
#########################################################
def create_bvh_tree_from_object(obj, margin=0.005):
    """
    Create BVH tree from the object (increase margin if collision is not detected)
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # Apply scale transformation to account for margin
    if margin > 0.0:
        for v in bm.verts:
            v.co += v.normal * margin
    bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh


def create_fracture_object(obj_mesh, N_PORTIONS=100, COLLISION_MARGIN=0.001):
    """
    Create a fractured object from the input object
    """
    # mean_color = get_mean_color(get_material_nodes(get_mesh_nodes(obj_mesh))[1])
    mean_color = compute_average_color_per_vertex(obj_mesh)

    # create a new material for the fractured object
    new_mat = bpy.data.materials.new(name="MyMaterial")
    new_mat.diffuse_color = (mean_color[0], mean_color[1], mean_color[2], 1)
    obj_mesh.data.materials.append(new_mat)
    
    # create a fractured object
    bpy.ops.object.select_all(action='DESELECT')
    obj_mesh.select_set(True)
    bpy.context.view_layer.objects.active = obj_mesh
    bpy.ops.preferences.addon_enable(module='object_fracture_cell')
    bpy.ops.object.add_fracture_cell_objects(
        source={'PARTICLE_OWN'}, 
        source_limit=N_PORTIONS,
        source_noise=1.0, 
        cell_scale=(1, 1, 1), 
        recursion=0, 
        recursion_source_limit=8, 
        recursion_clamp=250, 
        recursion_chance=0.25, 
        recursion_chance_select='SIZE_MIN', 
        use_smooth_faces=False, 
        use_sharp_edges=True, 
        use_sharp_edges_apply=True, 
        use_data_match=True, 
        use_island_split=True, 
        margin=0.001, 
        material_index=1, 
        use_interior_vgroup=False, 
        mass_mode='VOLUME', 
        mass=1, 
        use_recenter=True, 
        use_remove_original=True, 
        collection_name="", 
        use_debug_points=False, 
        use_debug_redraw=True, 
        use_debug_bool=False,
    )

    bpy.ops.object.select_pattern(pattern=obj_mesh.name + '_cell*')
    objects_to_move = bpy.context.selected_objects
    n_fractured_objects = len(objects_to_move)

    obj_mass = obj_mesh.rigid_body.mass
    obj_restitution = obj_mesh.rigid_body.restitution
    for obj in objects_to_move:
        add_rigid_body(obj, 'ACTIVE', 'CONVEX_HULL', obj_mass / n_fractured_objects, obj_restitution, COLLISION_MARGIN)  # make the debris to have small mass

    # (Optional) remove outlier pieces from fractured 3dgs objects via mesh intersection
    ##### 1. For each fractured pieces, make a copy then add a boolean modifier and set it to difference with the original object
    ##### 2. Use Exact solver and toggle both Self-intersection and Hole-Enabled to True 
    ##### 3. Apply the remesh modifier to the result from the boolean modifier 
    ##### 4. Compute the area of the result mesh compared to the original fractured mesh 
    ##### 5. If the area is above a threshold, remove the object
    if obj_mesh.name in all_3dgs_object_names:

        remove_indices = []

        for i, fractured_obj in enumerate(objects_to_move):

            bpy.ops.object.select_all(action='DESELECT')
            fractured_obj.select_set(True)
            bpy.context.view_layer.objects.active = fractured_obj
            bpy.ops.object.duplicate()
            duplicated_obj = bpy.context.view_layer.objects.active

            bool_mod = duplicated_obj.modifiers.new(name="Boolean", type='BOOLEAN')
            bool_mod.operation = 'DIFFERENCE'
            bool_mod.object = obj_mesh
            bool_mod.solver = 'EXACT'
            bool_mod.use_self = True            # Self-intersection enabled
            bool_mod.use_hole_tolerant = True   # Hole-tolerant enabled
            bpy.context.view_layer.objects.active = duplicated_obj
            bpy.ops.object.modifier_apply(modifier=bool_mod.name)

            remesh_mod = duplicated_obj.modifiers.new(name="Remesh", type='REMESH')
            remesh_mod.mode = 'SMOOTH'
            remesh_mod.scale = 0.9
            remesh_mod.threshold = 1
            bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

            remeshed_area = compute_area(duplicated_obj)
            original_area = compute_area(fractured_obj)

            outlier_ratio = 0.4
            if original_area > 0 and (remeshed_area / original_area) > outlier_ratio:
                remove_indices.append(i)
                delete_object_recursive(fractured_obj)

            delete_object_recursive(duplicated_obj)

        objects_to_move = [obj for i, obj in enumerate(objects_to_move) if i not in remove_indices]

    # Deselect all objects after moving them
    bpy.ops.object.select_all(action='DESELECT')

    return objects_to_move


def check_collision(obj1, obj2):
    """
    Check collision between two objects
    """
    if obj1.type != 'MESH':
        obj1 = get_mesh_nodes(obj1)[0]
    if obj2.type != 'MESH':
        obj2 = get_mesh_nodes(obj2)[0]
    bvh1 = create_bvh_tree_from_object(obj1)
    bvh2 = create_bvh_tree_from_object(obj2)
    return bvh1.overlap(bvh2)


def compute_area(obj):
    # Calculate the surface area of the given object
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    area = sum(face.calc_area() for face in bm.faces)
    bm.free()
    return area


#########################################################
# Others (Utility functions)
#########################################################
def get_object_matrix_world(obj_path):
    """
    Get the world matrix of the object (after re-centering)

    Args:
        obj_path: path to the object

    Returns:
        matrix_world: (4, 4) matrix
    """
    inserted_obj = load_object(obj_path)
    merge_meshes(inserted_obj)
    inserted_obj = remove_empty_nodes_v1(inserted_obj)
    transform_object_origin(inserted_obj, set_origin_to_bottom=False)
    matrix_world = inserted_obj.matrix_world.copy()
    bpy.data.objects.remove(inserted_obj, do_unlink=True)
    return matrix_world


def decompose_matrix_world(matrix_world):
    """
    Decompose the world matrix into position, rotation, and scale

    Args:
        matrix_world: (4, 4) matrix

    Returns:
        pos: (3,) position
        rot: (3, 3) rotation matrix
        scale: scale
    """
    pos = np.array(matrix_world.to_translation())
    rot = np.array(matrix_world.to_3x3())
    # rot = np.array(matrix_world.to_quaternion().to_matrix())
    scale = matrix_world.to_scale()[0]
    return pos, rot, scale


def set_hide_render_recursive(obj, hide_render):
    if obj.name.endswith('_proxy'):   # prevent proxy objects being rendered
        return
    obj.hide_render = hide_render
    for child in obj.children:
        set_hide_render_recursive(child, hide_render)


def set_visible_camera_recursive(obj, visible_camera):
    if obj.name.endswith('_proxy'):   # prevent proxy objects being rendered
        return
    obj.visible_camera = visible_camera
    for child in obj.children:
        set_visible_camera_recursive(child, visible_camera)


def delete_object_recursive(obj):
    for child in obj.children:
        delete_object_recursive(child)
    bpy.data.objects.remove(obj, do_unlink=True)


#########################################################
# Event handler
#########################################################
def event_parser(event):
    '''
    Parse the event from the config file.

    Returns:
        Two events in the format of (object_id, action, frame_num)
    '''
    object_id = event['object_id']
    event_type = event['event_type']
    start_frame = event['start_frame']
    end_frame = event['end_frame']

    # action for the event
    event_to_action = {
        'static': ('insert', 'remove'),
        'animation': ('start_animation', 'stop_animation'),
        'physics': ('start_physics', 'stop_physics'),
        'fire': ('start_fire', 'stop_fire', 'remove_fire'),
        'smoke': ('start_smoke', 'stop_smoke', 'remove_smoke'),
        'break': ('start_break', None),
        'incinerate': ('start_incinerate', None),
    }

    event_list = []
    if event_type in ['static', 'animation', 'physics']:
        event_list.append((object_id, event_to_action[event_type][0], start_frame))
        if end_frame is not None:
            event_list.append((object_id, event_to_action[event_type][1], end_frame))
    elif event_type in ['fire', 'smoke']:
        event_list.append((object_id, event_to_action[event_type][0], start_frame))
        if end_frame is not None:
            END_FIRE_SMOKE_BEFORE_DELETION = 0  # 20 by default
            event_list.append((object_id, event_to_action[event_type][1], end_frame - END_FIRE_SMOKE_BEFORE_DELETION))  # stop the fire/smoke 5 frames before deletion
            event_list.append((object_id, event_to_action[event_type][2], end_frame))
    elif event_type in ['break', 'incinerate']:
        event_list.append((object_id, event_to_action[event_type][0], start_frame))

    return event_list


def run_event_handler(frame_index, **kwargs):
    '''
    Run the event handler to check if there is any event happening at the current frame.
    '''
    print("======================= Run event handler at frame {} ============================".format(frame_index))
    for obj_id, event_list in all_events_dict.items():
        for event in event_list:
            obj_id, action, frame_num = event
            if frame_num == frame_index:
                execute_event(obj_id, action, **kwargs)


def execute_event(obj_id, action, **kwargs):
    '''
    Execute the event for the given object id and action.
    '''
    COLLISION_MARGIN = kwargs.get('collision_margin', 0.001)
    DOMAIN_HEIGHT = kwargs.get('domain_height', 2.0)
    CACHE_DIR = kwargs.get('cache_dir', None)

    obj = all_object_dict[obj_id]
    if action == 'insert':                  # TODO: currently not handle material change, 3DGS objects, fire, smoke, fracture
        pass
        # if obj_id in all_object_dict:
        #     return
        # obj_info = all_object_info[obj_id]
        # rb_info = obj_info['rigid_body']
        # obj_mesh = insert_object(obj_info['object_path'], np.array(obj_info['pos']), np.array(obj_info['rot']), obj_info['scale'])
        # add_rigid_body(obj_mesh, rb_info['rb_type'], 'CONVEX_HULL', rb_info['mass'], rb_info['restitution'], COLLISION_MARGIN)
        # object_list.append(obj_mesh)
        # all_object_dict[obj_info['object_id']] = obj_mesh
    if action == 'remove':
        pass
        # obj_mesh = all_object_dict[obj_id]
        # object_list.remove(obj_mesh)
        # bpy.data.objects.remove(obj_mesh, do_unlink=True)
    if action == 'start_animation':
        pass
    if action == 'stop_animation':
        pass
    if action == 'start_physics':
        all_obj_nodes = [obj] + obj.children_recursive
        for obj_node in all_obj_nodes:
            if obj_node.type == 'MESH' and obj_node.rigid_body is not None:
                obj_node.rigid_body.type = 'ACTIVE'
                obj_node.rigid_body.kinematic = False
    if action == 'stop_physics':
        all_obj_nodes = [obj] + obj.children_recursive
        for obj_node in all_obj_nodes:
            if obj_node.type == 'MESH' and obj_node.rigid_body is not None:
                obj_node.rigid_body.type = 'PASSIVE'
                obj_node.rigid_body.kinematic = True
    if action == 'start_fire':
        # smoke_domain = add_smoke(obj, with_fire=True, domain_height=DOMAIN_HEIGHT, cache_dir=CACHE_DIR)
        # smoke_domain.name = obj_id + '_smoke'
        # smoke_domain_dict[obj_id] = smoke_domain
        fire_object_id_list.append(obj_id)
    if action == 'remove_fire':
        smoke_domain = smoke_domain_dict[obj_id]
        bpy.data.objects.remove(smoke_domain, do_unlink=True)
        del smoke_domain_dict[obj_id]
        fire_object_id_list.remove(obj_id)
    if action == 'start_smoke':
        # smoke_domain = add_smoke(obj, with_fire=False, domain_height=DOMAIN_HEIGHT, cache_dir=CACHE_DIR)
        # smoke_domain.name = obj_id + '_smoke'
        # smoke_domain_dict[obj_id] = smoke_domain
        smoke_object_id_list.append(obj_id)
    if action == 'remove_smoke':
        smoke_domain = smoke_domain_dict[obj_id]
        bpy.data.objects.remove(smoke_domain, do_unlink=True)
        del smoke_domain_dict[obj_id]
        smoke_object_id_list.remove(obj_id)
    if action == 'stop_fire' or action == 'stop_smoke':
        smoke_domain = smoke_domain_dict[obj_id]
        blender_obj = obj
        if blender_obj.type != 'MESH':
            all_object_nodes = [blender_obj] + blender_obj.children_recursive
            if blender_obj.name in smoke_proxy_obj_dict:            # TODO: use for turn off smoke emission for proxy objects
                proxy_obj = smoke_proxy_obj_dict[blender_obj.name]
                all_object_nodes += [proxy_obj] + proxy_obj.children_recursive
            for obj_node in all_object_nodes and blender_obj.modifiers['Fluid'] is not None:
                if obj_node.type == 'MESH' and 'Fluid' in obj_node.modifiers:
                    blender_obj = obj_node
                    break
        blender_obj.modifiers["Fluid"].flow_settings.fuel_amount = 0
        # Future TODO: set key frame to gradually change the fuel amount to 0
        # mat_volume_node = bpy.data.materials["Smoke Domain Material"].node_tree.nodes["Principled Volume"]
        # mat_volume_node.inputs[2].default_value = 0.0
    if action == 'start_break':
        if obj.type != 'MESH':
            obj_mesh = get_mesh_nodes(obj)[0]
        else:
            obj_mesh = obj
        fracture_object = create_fracture_object(obj_mesh, COLLISION_MARGIN=COLLISION_MARGIN)
        # remove the original object from the list (object_list, object_3dgs_list and all_object_dict)
        if obj in object_list:
            object_list.remove(obj)
        if obj in object_3dgs_list:
            object_3dgs_list.remove(obj)
        del all_object_dict[obj_id]
        # set to invisible (avoid rendering mask)
        set_hide_render_recursive(obj, True)
        # delete the original object
        delete_object_recursive(obj)
        # add the fractured objects to the list
        debris_object_list.extend(fracture_object)
    if action == 'start_incinerate':
        # TODO: not implemented yet
        pass

    if action in ['start_fire', 'start_smoke', 'start_break', 'start_incinerate']:
        # if in 3dgs object list, remove it from the list
        if obj in object_3dgs_list:
            object_3dgs_list.remove(obj)

    bpy.context.view_layer.update()



#########################################################
# Global variables
#########################################################
all_object_info = {}           # object_id -> object_info from insert_object_info
all_object_dict = {}           # object_id -> blender object
object_list = []               # list of foreground objects (ex: blender assets or 3dgs objects with modified materials)
object_3dgs_list = []          # list of 3dgs objects (ex: 3dgs objects with original materials)
object_3dgs_scale_dict = {}    # scale of 3dgs objects (keep the scale here since apply_transform will change the scale to 1.0 for the sake of rigid body simulation)
smoke_domain_dict = {}         # object_id -> smoke domain
fire_object_id_list = []       # list of object_id that has fire
smoke_object_id_list = []      # list of object_id that has smoke
fracture_object_list = []      # list of fracturable objects (keep track if collision happens)
all_events_dict = {}           # object_id -> list of events
debris_object_list = []        # store the debris generated from fracture

smoke_proxy_obj_dict = {}      # object_id -> smoke domain for proxy object
all_3dgs_object_names = []     # list of names of all 3dgs objects (used for custom post-filtering after creating fractures)

# COLLISION_MARGIN = 0.001
# DOMAIN_HEIGHT = 8.0
# CACHE_DIR = None 
# # CACHE_DIR = '/tmp/smoke_cache'  # default cache directory for smoke simulation


#########################################################
# Main function (currently from rb_sim_rendering.py)
#########################################################
def run_blender_render(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    results_dir = config['results_dir']
    traj_results_dir = config['traj_results_dir']
    h, w = config['im_height'], config['im_width']
    K = np.array(config['K'])
    c2w = np.array(config['c2w'])
    insert_object_info = config['insert_object_info']
    scene_mesh_path = config['scene_mesh_path']
    global_env_map_path = config['global_env_map_path']
    is_uv_mesh = config['is_uv_mesh']
    fire_object_info = config['fire_objects']
    smoke_object_info = config['smoke_objects']
    emitter_mesh_path = config['emitter_mesh_path']

    is_indoor_scene = False
    if 'is_indoor_scene' in config:
        is_indoor_scene = config['is_indoor_scene']

    is_waymo_scene = False
    if 'waymo_scene' in config:
        is_waymo_scene = config['waymo_scene']

    if config["render_type"] == 'SINGLE_VIEW' and config["anchor_frame_idx"] is not None:
        anchor_frame_idx = config["anchor_frame_idx"]
        c2w = c2w[anchor_frame_idx:anchor_frame_idx+1]
        if len(K.shape) == 3:
            K = K[anchor_frame_idx:anchor_frame_idx+1]

    if "output_dir_name" in config:
        output_dir = os.path.join(traj_results_dir, config["output_dir_name"])
    else:
        output_dir = os.path.join(traj_results_dir, 'blend_results')
    os.makedirs(output_dir, exist_ok=True)

    # anti-aliasing rendering
    upscale = 2.0
    w = int(w * upscale)
    h = int(h * upscale)
    if len(K.shape) == 2:
        K[0, 0] *= upscale
        K[1, 1] *= upscale
        K[0, 2] *= upscale
        K[1, 2] *= upscale
    else:
        for i in range(len(K)):
            K[i][0, 0] *= upscale
            K[i][1, 1] *= upscale
            K[i][0, 2] *= upscale
            K[i][1, 2] *= upscale

    setup_blender_env(w, h)

    scene_mesh = add_meshes_shadow_catcher(scene_mesh_path, is_uv_mesh)

    # Add a large plane for waymo scene (if required)
    # planar_mesh = None
    # if 'waymo_scene' in config and config['waymo_scene']:
    #     planar_mesh = add_planar_shadow_catcher(size=50)

    emitter_mesh = None
    if emitter_mesh_path is not None:
        emitter_mesh = load_object(emitter_mesh_path)               # use emitter mesh for illumination if provided
        add_emitter_lighting(emitter_mesh)                          # default strength: 100, color: white
        # emitter_mesh.location[2] -= 0.001                           # TODO: DEBUGGING on for gray stripe artifact (move the emitter mesh down to avoid this)
        bpy.context.view_layer.update()
    else:
        if is_indoor_scene: # or is_waymo_scene:
            add_env_lighting(global_env_map_path, strength=2.0)     # indoor scene or waymo scene
        else:
            add_env_lighting(global_env_map_path, strength=0.6)     # outdoor scene

    COLLISION_MARGIN = 0.001  # default collision margin: 0.005 or 0.001
    DOMAIN_HEIGHT = 2.0
    CACHE_DIR = None
    REMESH_VOXEL_SIZE = 0.01

    if is_waymo_scene:
        print("=====> Add sun lighting for waymo scene")
        sun_dir = np.array(config['sun_dir'])
        add_sun_lighting(1.0, sun_dir)
        COLLISION_MARGIN = 0.01
        DOMAIN_HEIGHT = 6.0
        REMESH_VOXEL_SIZE = 0.1

    print(COLLISION_MARGIN)

    cam = Camera(h, w, output_dir)
    cam_list = create_camera_list(c2w, K)

    scene.frame_start = 1
    if config["render_type"] == 'SINGLE_VIEW':
        scene.frame_end = config['num_frames']
    else:
        scene.frame_end = len(c2w)
    print("frame start: {}, frame end: {}".format(scene.frame_start, scene.frame_end))

    for obj_info in insert_object_info:
        all_object_info[obj_info['object_id']] = obj_info

    # setup the events
    if 'events' in config:
        for event in config['events']:
            event_list = event_parser(event)
            for obj_id, action, frame_num in event_list:
                if obj_id not in all_events_dict:
                    all_events_dict[obj_id] = []
                all_events_dict[obj_id].append((obj_id, action, frame_num))

    ##### initially add all objects to the scene (including the fire and smoke) #####
    for obj_id, obj_info in all_object_info.items():
        print("=====", obj_info['object_id'], "=====")
        if obj_info['animation'] is not None:
            object_mesh = insert_animated_object(obj_info['object_path'], np.array(obj_info['pos']), np.array(obj_info['rot']), obj_info['scale'])
            rb_info = obj_info['rigid_body']
            # add_rigid_body(object_mesh, rb_info['rb_type'], rb_info['collision_shape'], rb_info['mass'], rb_info['restitution'], COLLISION_MARGIN)
            add_rigid_body(object_mesh, rb_info['rb_type'], 'CONVEX_HULL', rb_info['mass'], rb_info['restitution'], COLLISION_MARGIN)  # use convex hull by default
            add_cyclic_animation(object_mesh)
            if obj_info['animation']['type'] == 'trajectory':
                t1, t2 = 1, scene.frame_end
                if obj_info['object_id'] in all_events_dict:    # get the start and end frame of the animation
                    for event in all_events_dict[obj_info['object_id']]:
                        if event[1] == 'start_animation':
                            t1 = event[2]
                        if event[1] == 'stop_animation':
                            t2 = event[2]
                set_linear_trajectory(object_mesh, obj_info['animation']['points'], t1=t1, t2=t2, f_axis=obj_info['forward_axis'])
            # obj_info['object_id'] = 'chatsim_' + obj_info['object_id'] if object_mesh.name.startswith('chatsim') else obj_info['object_id']
            object_mesh.name = obj_info['object_id']
            if obj_info['from_3DGS']:
                all_3dgs_object_names.append(object_mesh.name)
            object_list.append(object_mesh)
        else:
            object_mesh = insert_object(obj_info['object_path'], np.array(obj_info['pos']), np.array(obj_info['rot']), obj_info['scale'], obj_info['from_3DGS'])
            rb_info = obj_info['rigid_body']
            # add_rigid_body(object_mesh, rb_info['rb_type'], rb_info['collision_shape'], rb_info['mass'], rb_info['restitution'], COLLISION_MARGIN)
            add_rigid_body(object_mesh, rb_info['rb_type'], 'CONVEX_HULL', rb_info['mass'], rb_info['restitution'], COLLISION_MARGIN)  # use convex hull by default
            if obj_info['material'] is not None:
                apply_material_to_object(object_mesh, obj_info['material'])
            # obj_info['object_id'] = 'chatsim_' + obj_info['object_id'] if object_mesh.name.startswith('chatsim') else obj_info['object_id']
            object_mesh.name = obj_info['object_id']
            if obj_info['from_3DGS'] and obj_info['material'] is None:   # if material is modified, treat as normal object
                object_3dgs_scale_dict[object_mesh.name] = obj_info['scale']
                object_3dgs_list.append(object_mesh)
            else:
                object_list.append(object_mesh)
            if obj_info['from_3DGS']:
                all_3dgs_object_names.append(object_mesh.name)
        if obj_info['fracture']:
            fracture_object_list.append(object_mesh)
        all_object_dict[obj_info['object_id']] = object_mesh

    add_rigid_body(scene_mesh, 'PASSIVE', 'MESH', 1.0, 0.5, COLLISION_MARGIN)

    # if planar_mesh is not None:
    #     add_rigid_body(planar_mesh, 'PASSIVE', 'MESH', 1.0, 0.5, COLLISION_MARGIN)

    # check fire_object_info and smoke_object_info
    if fire_object_info is not None:
        for object_id in fire_object_info:
            object_mesh = all_object_dict[object_id]
            smoke_domain = add_smoke(object_mesh, with_fire=True, domain_height=DOMAIN_HEIGHT, cache_dir=CACHE_DIR, start_frame=1, end_frame=scene.frame_end, remesh_voxel_size=REMESH_VOXEL_SIZE)
            smoke_domain.name = object_id + '_smoke'
            smoke_domain_dict[object_id] = smoke_domain
            fire_object_id_list.append(object_id)

    if smoke_object_info is not None:
        for object_id in smoke_object_info:
            object_mesh = all_object_dict[object_id]
            smoke_domain = add_smoke(object_mesh, with_fire=False, domain_height=DOMAIN_HEIGHT, cache_dir=CACHE_DIR, start_frame=1, end_frame=scene.frame_end, remesh_voxel_size=REMESH_VOXEL_SIZE)
            smoke_domain.name = object_id + '_smoke'
            smoke_domain_dict[object_id] = smoke_domain
            smoke_object_id_list.append(object_id)

    # check if fire and smoke related events exist
    for event in config['events']:
        if event['event_type'] in ['fire', 'smoke']:
            obj_id = event['object_id']
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            object_mesh = all_object_dict[obj_id]
            if event['event_type'] == 'fire':
                smoke_domain = add_smoke(object_mesh, with_fire=True, domain_height=DOMAIN_HEIGHT, cache_dir=CACHE_DIR, start_frame=start_frame, end_frame=end_frame, remesh_voxel_size=REMESH_VOXEL_SIZE)
                smoke_domain.name = obj_id + '_smoke'
                smoke_domain_dict[obj_id] = smoke_domain
                fire_object_id_list.append(obj_id)
            if event['event_type'] == 'smoke':
                smoke_domain = add_smoke(object_mesh, with_fire=False, domain_height=DOMAIN_HEIGHT, cache_dir=CACHE_DIR, start_frame=start_frame, end_frame=end_frame, remesh_voxel_size=REMESH_VOXEL_SIZE)
                smoke_domain.name = obj_id + '_smoke'
                smoke_domain_dict[obj_id] = smoke_domain
                smoke_object_id_list.append(obj_id)
            

    bpy.context.view_layer.update()     # Update the scene

    ##### setup properties for rigid body world #####
    scene.rigidbody_world.time_scale = 1.0
    scene.rigidbody_world.point_cache.frame_start = scene.frame_start
    scene.rigidbody_world.point_cache.frame_end = scene.frame_end

    # DO NOT bake if simulate with particles
    # if len(smoke_object_info) == 0 and len(fire_object_info) == 0:
    #     bpy.ops.ptcache.bake_all(bake=True)

    cam.initialize_depth_extractor()  # initialize once

    rb_transform = {}

    for FRAME_INDEX in range(scene.frame_start, scene.frame_end + 1):

        # print("Frame index: ", FRAME_INDEX)
        # print("===== All object names: ", [obj.name for obj in all_object_dict.values()])
        # print("===== Object names: ", [obj.name for obj in object_list])
        # print("===== 3DGS object names: ", [obj.name for obj in object_3dgs_list])
        # print("===== Fracture objects: ", [obj.name for obj in fracture_object_list])
        # print("===== Debris objects: ", [obj.name for obj in debris_object_list])

        scene.frame_set(FRAME_INDEX)
        bpy.context.view_layer.update()     # Ensure the scene is fully updated

        run_event_handler(FRAME_INDEX, collision_margin=COLLISION_MARGIN, domain_height=DOMAIN_HEIGHT, cache_dir=CACHE_DIR)

        # check if fracture objects collide with other objects
        fracture_name_list = []
        for obj in fracture_object_list:
            for _, obj2 in all_object_dict.items():
                if obj.name != obj2.name and check_collision(obj, obj2):
                    print("Collision detected between {} and {}".format(obj.name, obj2.name))
                    fracture_name_list.append(obj.name)
                    break
        
        # fracture the object
        for obj_name in fracture_name_list:
            obj = all_object_dict[obj_name]
            if obj.type != 'MESH':
                obj_mesh = get_mesh_nodes(obj)[0]
            else:
                obj_mesh = obj
            fracture_object = create_fracture_object(obj_mesh, COLLISION_MARGIN=COLLISION_MARGIN)
            # remove the original object from the list (fracutre_object_list, object_list, object_3dgs_list and all_object_dict)
            fracture_object_list.remove(obj)
            if obj in object_list:
                object_list.remove(obj)
            if obj in object_3dgs_list:
                object_3dgs_list.remove(obj)
            del all_object_dict[obj_name]
            # set to invisible (avoid rendering mask)
            set_hide_render_recursive(obj, True)
            # delete the original object
            delete_object_recursive(obj)
            # add the fractured objects to the list
            debris_object_list.extend(fracture_object)
        
        bpy.context.view_layer.update()     # Ensure the scene is fully updated

        has_fire = len(fire_object_id_list) > 0

        # Step 1: render only inserted objects
        for object_mesh in object_list + debris_object_list:
            set_visible_camera_recursive(object_mesh, True)
        for object_mesh in object_3dgs_list:
            set_visible_camera_recursive(object_mesh, False)
        for smoke_domain in smoke_domain_dict.values():
            set_hide_render_recursive(smoke_domain, True)
        scene_mesh.visible_camera = False
        if emitter_mesh is not None:
            emitter_mesh.visible_camera = False
        scene.cycles.samples = 64
        
        if config["render_type"] == 'SINGLE_VIEW':
            cam.render_single_timestep_rgb_and_depth(cam_list[0], FRAME_INDEX, dir_name_rgb='rgb_obj', dir_name_depth='depth_obj')
        else:
            cam.render_single_timestep_rgb_and_depth(cam_list[FRAME_INDEX-1], FRAME_INDEX, dir_name_rgb='rgb_obj', dir_name_depth='depth_obj')

        # Step 2: render only 3DGS objects
        for object_mesh in object_list + debris_object_list:
            set_visible_camera_recursive(object_mesh, False)
        for object_mesh in object_3dgs_list:
            set_visible_camera_recursive(object_mesh, True)
        for smoke_domain in smoke_domain_dict.values():
            set_hide_render_recursive(smoke_domain, True)
        scene_mesh.visible_camera = False
        if emitter_mesh is not None:
            emitter_mesh.visible_camera = False
        scene.cycles.samples = 64

        if len(object_3dgs_list) > 0:
            if config["render_type"] == 'SINGLE_VIEW':
                cam.render_single_timestep_rgb_and_depth(cam_list[0], FRAME_INDEX, dir_name_rgb='rgb_obj_3dgs', dir_name_depth='depth_obj_3dgs')
            else:
                cam.render_single_timestep_rgb_and_depth(cam_list[FRAME_INDEX-1], FRAME_INDEX, dir_name_rgb='rgb_obj_3dgs', dir_name_depth='depth_obj_3dgs')

        # Step 3: render only smoke and fire with inserted objects
        # for object_mesh in object_list + debris_object_list:
        #     set_visible_camera_recursive(object_mesh, True)
        for object_id in fire_object_id_list + smoke_object_id_list:
            object_mesh = all_object_dict[object_id]
            set_visible_camera_recursive(object_mesh, True)
        for object_mesh in object_3dgs_list:
            set_visible_camera_recursive(object_mesh, False)
        for smoke_domain in smoke_domain_dict.values():
            set_hide_render_recursive(smoke_domain, False)
        scene_mesh.visible_camera = False
        if emitter_mesh is not None:
            emitter_mesh.visible_camera = False
        scene.cycles.samples = 512   # increase the samples for smoke and fire

        if len(smoke_domain_dict) > 0:
            if config["render_type"] == 'SINGLE_VIEW':
                cam.render_single_timestep_rgb_and_depth(cam_list[0], FRAME_INDEX, dir_name_rgb='rgb_smoke_fire', dir_name_depth='depth_smoke_fire')
                if has_fire:
                    render.image_settings.color_mode = "RGB"
                    cam.render_single_timestep_rgb_and_depth(cam_list[0], FRAME_INDEX, dir_name_rgb='rgb_smoke_fire_pre', dir_name_depth='depth_smoke_fire_pre')
                    render.image_settings.color_mode = "RGBA"
            else:
                cam.render_single_timestep_rgb_and_depth(cam_list[FRAME_INDEX-1], FRAME_INDEX, dir_name_rgb='rgb_smoke_fire', dir_name_depth='depth_smoke_fire')
                if has_fire:
                    render.image_settings.color_mode = "RGB"
                    cam.render_single_timestep_rgb_and_depth(cam_list[FRAME_INDEX-1], FRAME_INDEX, dir_name_rgb='rgb_smoke_fire_pre', dir_name_depth='depth_smoke_fire_pre')
                    render.image_settings.color_mode = "RGBA"

        # Step 4: render only shadow catcher
        for object_mesh in object_list + debris_object_list:
            set_hide_render_recursive(object_mesh, True)
        for object_mesh in object_3dgs_list:
            set_hide_render_recursive(object_mesh, True)
        for smoke_domain in smoke_domain_dict.values():
            set_hide_render_recursive(smoke_domain, True)
        scene_mesh.visible_camera = True
        if emitter_mesh is not None:
            emitter_mesh.visible_camera = True
        scene.cycles.samples = 64

        if config["render_type"] == 'SINGLE_VIEW':
            cam.render_single_timestep_rgb_and_depth(cam_list[0], FRAME_INDEX, dir_name_rgb='rgb_shadow', dir_name_depth='depth_shadow')
        else:
            cam.render_single_timestep_rgb_and_depth(cam_list[FRAME_INDEX-1], FRAME_INDEX, dir_name_rgb='rgb_shadow', dir_name_depth='depth_shadow')

        # Step 5: render all effects
        for object_mesh in object_list + debris_object_list:
            set_hide_render_recursive(object_mesh, False)
            set_visible_camera_recursive(object_mesh, True)
        for object_mesh in object_3dgs_list:
            set_hide_render_recursive(object_mesh, False)
            set_visible_camera_recursive(object_mesh, False)
        for smoke_domain in smoke_domain_dict.values():
            set_hide_render_recursive(smoke_domain, False)
        scene_mesh.visible_camera = True
        if emitter_mesh is not None:
            emitter_mesh.visible_camera = True
        if len(smoke_domain_dict) > 0:
            scene.cycles.samples = 512   # TODO: might drop to 64 if no smoke and fire
        else:
            scene.cycles.samples = 64

        if config["render_type"] == 'SINGLE_VIEW':
            cam.render_single_timestep_rgb_and_depth(cam_list[0], FRAME_INDEX, dir_name_rgb='rgb_all', dir_name_depth='depth_all')
        else:
            cam.render_single_timestep_rgb_and_depth(cam_list[FRAME_INDEX-1], FRAME_INDEX, dir_name_rgb='rgb_all', dir_name_depth='depth_all')

        # Step 6: save the rigid body transformation of 3dgs objects
        for object_mesh in object_3dgs_list:
            if object_mesh.name not in rb_transform:
                rb_transform[object_mesh.name] = {}
            transform = {}
            pos, rot, scale = decompose_matrix_world(object_mesh.matrix_world)
            transform['pos'] = pos.tolist()
            transform['rot'] = rot.tolist()
            transform['scale'] = object_3dgs_scale_dict[object_mesh.name]
            rb_transform[object_mesh.name]['{0:03d}'.format(FRAME_INDEX)] = transform
            
    # add rigid body transformation to the original config file
    if rb_transform:
        config['rb_transform'] = rb_transform
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument('--input_config_path', type=str, default='')
    args = parser.parse_args()
    run_blender_render(args.input_config_path)