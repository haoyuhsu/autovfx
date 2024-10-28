import os
import open3d as o3d
import numpy as np
import math
import random
import glob
from PIL import Image
import cv2
import torch
import trimesh
import copy
from scipy.spatial.transform import Rotation as R
from opt import ROOT_DIR, BLENDER_PATH
from sugar.gaussian_splatting.utils.general_utils import safe_state
from tracking.demo_with_text import run_deva
from extract.extract_object import extract_object_from_scene, extract_object_from_single_view, get_largest_object, inpaint_object
from gaussians_utils import get_scaling_of_mesh, get_center_of_mesh_2, get_bottom_center_of_mesh
from retrieval.wrapper_objaverse import retrieve_asset_from_objaverse
from retrieval.wrapper_polyhaven import retrieve_materials_from_polyhaven
from gpt.gpt4v_utils import estimate_object_scale, estimate_object_forward_axis
# from blender.asset_rendering import run_object_render as render_object


'''
Wrapper of the modular functions for GPT model to call
- (o) detect_object
- (?) sample_point_on_object
- (o) sample_point_above_object
- (o) retrieve_asset
- (o) insert_object
- (o) remove_object
- (o) update_object
- (o) allow_physics
- (o) add_fire
- (o) add_smoke
- (o) set_static_animation
- (o) set_moving_animation
- (o) retrieve_material
- (o) init_material
- (o) apply_material
- (o) allow_fracture

- (o) get_object_bottom_position
- (?) get_object_center_position
- (o) translate_object
- (o) rotate_object
- (o) scale_object
- (o) get_random_2D_rotation
- (o) get_random_3D_rotation
- (o) make_copy

##### Additional functions for time-varying scene editing #####
- (o) make_break
- (o) incinerate
- (o) add_event
- (o) get_camera_position

##### Additional functions for autonomous driving scene editing #####
- (o) get_vehicle_position
- (o) get_direction
- (o) retrieve_chatsim_asset

'''


def get_default_object_info():
    '''
    Get default object information
    '''
    return {
        'object_name': 'object',
        'object_id': 'object_id',
        'object_path': 'path/to/object.obj',
        'pos': np.array([0, 0, 0]).astype(np.float32),
        'rot': np.eye(3).astype(np.float32),
        'scale': 1.0,
        'from_3DGS': False,
        'forward_axis': "TRACK_NEGATIVE_Y",  # "FORWARD_X", "FORWARD_Y", "TRACK_NEGATIVE_X", "TRACK_NEGATIVE_Y"
        'animation': None,
        'rigid_body': {
            'rb_type': 'PASSIVE',       # 'ACTIVE', 'PASSIVE', 'KINEMATIC'
            'collision_shape': 'MESH',  # 'BOX', 'SPHERE', 'CYLINDER', 'CONE', 'CAPSULE', 'MESH'
            'mass': 1.0,
            'restitution': 0.5,
        },
        'material': None,
        'fracture': False,
        'break': False,
        'incinerate': False,
    }


def get_default_event_info():
    '''
    Get default event information
    '''
    return {
        'object_id': 'dummy',
        'event_type': 'dummy',
        'start_frame': 1,
        'end_frame': None
    }


class Material:
    def __init__(self, roughness=0.5, metallic=0.0, specular=0.5, material_path=None, is_mirror=False, rgb=None):
        self.roughness = roughness
        self.metallic = metallic
        self.specular = specular
        self.material_path = material_path
        self.is_mirror = is_mirror
        self.rgb = rgb


def detect_object(scene_representation, object_name):
    '''
    Detect and extract instance level meshes from the scene
    '''
    print("Detecting object: {}".format(object_name))

    object_tracking_results_dir = os.path.join(scene_representation.tracking_results_dir, '_'.join(object_name.split(' ')))
    if not os.path.exists(object_tracking_results_dir):
        print("Tracking object {}......".format(object_name))
        run_deva(os.path.join(scene_representation.traj_results_dir, 'images'), scene_representation.tracking_results_dir, object_name, scene_representation.DINO_THRESHOLD)

    obj_ids = sorted([int(x) for x in os.listdir(object_tracking_results_dir) if x.isdigit()])
    if len(obj_ids) == 0:
        raise ValueError("No instance of object {} found in the tracking results.".format(object_name))
    
    # handle multiple objects (pick the largest one across all frames)
    obj_ids = [get_largest_object(scene_representation, object_name, obj_ids)]
    
    obj_list = []
    for obj_id in obj_ids:
        obj_mesh_path = extract_object_from_scene(scene_representation, object_name, obj_id)
        new_obj = get_default_object_info()
        new_obj['object_name'] = object_name
        new_obj['object_id'] = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=16))
        new_obj['object_path'] = obj_mesh_path
        new_obj['pos'] = get_bottom_center_of_mesh(obj_mesh_path)
        new_obj['from_3DGS'] = True
        obj_list.append(new_obj)

    return obj_list[0]  # return the first detected object


def sample_point_on_object(scene_representation, obj):
    '''
    Sample a point on the object.
    1. Get vertices of the object mesh that are facing upwards
    2. Cast rays from the vertices + offset to the negative z-axis
    3. Get the intersection point with the object mesh
    '''
    obj_mesh = trimesh.load_mesh(obj['object_path'])

    # Get triangles that are facing upwards
    COSINE_THRESHOLD = np.cos(np.radians(10))
    normals = obj_mesh.face_normals
    mask_pos_z = np.dot(normals, np.array([0, 0, 1])) > COSINE_THRESHOLD
    mask_neg_z = np.dot(normals, np.array([0, 0, -1])) > COSINE_THRESHOLD
    mask = np.logical_or(mask_pos_z, mask_neg_z)
    z_facing_triangles = np.where(mask)[0]

    # Cast rays from the vertices + offset to the negative z-axis (get the top surface)
    vertices = obj_mesh.triangles_center[z_facing_triangles]
    z_max = np.max(vertices[:, 2]) + 0.5
    rays_o = np.concatenate([vertices[:, :2], np.ones((vertices.shape[0], 1)) * z_max], axis=1)
    rays_d = np.tile(np.array([0, 0, -1]), (vertices.shape[0], 1))
    locations, index_ray, index_tri = obj_mesh.ray.intersects_location(
        ray_origins=rays_o,
        ray_directions=rays_d,
        multiple_hits=False
    )

    # Check if all neighbors are also close to facing upwards
    selected_triangles = []
    for triangle_index in index_tri:
        # Get neighboring triangle indices
        neighbors = obj_mesh.face_adjacency[obj_mesh.face_adjacency[:, 0] == triangle_index, 1]
        neighbors = np.concatenate([neighbors, obj_mesh.face_adjacency[obj_mesh.face_adjacency[:, 1] == triangle_index, 0]])
        neighbors = np.unique(neighbors)
        if all(n in z_facing_triangles for n in neighbors):
            selected_triangles.append(triangle_index)
    selected_triangles = np.array(selected_triangles)
    locations = obj_mesh.triangles_center[selected_triangles]

    if len(locations) == 0:
        raise ValueError("No intersection point found on the object.")
    
    # TODO: rejection sampling required
    selected_location = locations[random.randint(0, len(locations)-1)]
    print("Sampling point on object: {} {} at location {}".format(obj['object_name'], obj['object_id'], selected_location))
    return selected_location


def sample_point_above_object(scene_representation, obj, VERTICAL_OFFSET=0.6):
    '''
    Sample a point above the object. (e.g., 10cm above the object)
    '''
    print("Sampling point above object: {} {}".format(obj['object_name'], obj['object_id']))
    selected_location = sample_point_on_object(scene_representation, obj)
    selected_location[2] += VERTICAL_OFFSET / scene_representation.scene_scale
    return selected_location


def retrieve_asset(scene_representation, object_name, is_animated=False):
    '''
    Retrieve 3D asset by object name from objaverse.  # TODO: retrieve animated asset
    '''
    obj_info = retrieve_asset_from_objaverse(object_name, is_animated=is_animated)
    new_obj = get_default_object_info()
    new_obj['object_name'] = object_name
    new_obj['object_id'] = obj_info['object_id']
    new_obj['object_path'] = obj_info['object_path']
    new_obj['from_3DGS'] = False

    multi_view_render_dir = os.path.join(scene_representation.cache_dir, 'assets_rendering_multi_views')

    # Quick rendering of the object
    os.system('{} --background --python ./blender/asset_rendering.py -- --object_file={} --output_dir={} --num_images={}'.format( \
        BLENDER_PATH, \
        obj_info['object_path'], \
        multi_view_render_dir, \
        4
    ))

    # Estimate forward axis of the object by GPT-4V API (only used for animated object)
    forward_axis = 'TRACK_NEGATIVE_Y'  # default forward axis
    img_folder = os.path.join(multi_view_render_dir, obj_info['object_id'])
    if is_animated:
        forward_axis = estimate_object_forward_axis(img_folder, object_name)
        print("Estimated forward axis of {} is {}.".format(object_name, forward_axis))

    # Estimate the scale of the object by GPT-4V API
    FORWARD_AXIS_TO_INDEX = {'TRACK_NEGATIVE_Y': 0, 'FORWARD_X': 1, 'FORWARD_Y': 2, 'TRACK_NEGATIVE_X': 3}
    img_path = sorted(glob.glob(os.path.join(img_folder, '*.png')))[FORWARD_AXIS_TO_INDEX[forward_axis]]
    object_scale = estimate_object_scale(img_path, obj_info['object_name'])  # use both rendered image and object name
    # object_scale = estimate_object_scale(None, obj_info['object_name'])    # use only object name
    # object_scale = estimate_object_scale(img_path, None)                   # use only rendered image
    print("Estimated scale of {} is {} meters.".format(obj_info['object_name'], object_scale))

    new_obj['forward_axis'] = forward_axis
    new_obj['scale'] = object_scale / scene_representation.scene_scale

    return new_obj


def insert_object(scene_representation, obj):
    '''
    Insert object into the scene. (add the object into list)
    '''
    scene_representation.insert_object(obj)
    print("Inserting object: {} {}".format(obj['object_name'], obj['object_id']))


def remove_object(scene_representation, obj, remove_gaussians=True):
    '''
    Remove object from the scene.
    '''
    # TODO: change the path of scene mesh, also change the path of gaussians
    # TODO: both geometry & gaussians in-painting are required
    
    obj_path = obj['object_path']
    base_folder = '/'.join(obj_path.split('/')[:-2])
    obj_name, obj_id = base_folder.split('/')[-2], base_folder.split('/')[-1]

    new_scene_mesh_path = os.path.join(base_folder, 'inpaint_removal_mesh/inpaint_removal_mesh.obj')
    if not os.path.exists(new_scene_mesh_path):
        inpaint_object(scene_representation, obj_name, obj_id)
    scene_representation.scene_mesh_path_for_blender = new_scene_mesh_path

    if remove_gaussians:
        new_gaussians_path = os.path.join(base_folder, 'inpaint_gaussians.ply')
        if not os.path.exists(new_gaussians_path):
            scene_representation.training_3DGS_for_inpainting(
                os.path.join(base_folder, 'removal_gaussians.ply'),
                os.path.join(base_folder, 'render_inpaint_lama'),
                os.path.join(base_folder, 'render_inpaint_mask'),
                base_folder,
                os.path.join(base_folder, 'inpaint_camera_poses.json')
            )
        scene_representation.hparams.gaussians_ckpt_path = new_gaussians_path

    print("Removing object: {} {}".format(obj['object_name'], obj['object_id']))


def update_object(scene_representation, obj):
    '''
    Update object in the scene. (update the object info)
    '''
    # if 3DGS object is on smoke or fire, do NOT remove the object
    has_fire_smoke_event = False
    for event in scene_representation.events:
        if event['object_id'] == obj['object_id'] and event['event_type'] in ['fire', 'smoke']:
            has_fire_smoke_event = True
            break
    if (obj['object_id'] in scene_representation.fire_objects or \
        obj['object_id'] in scene_representation.smoke_objects or \
        has_fire_smoke_event):
        remove_object(scene_representation, obj, remove_gaussians=False)
    else:
        remove_object(scene_representation, obj, remove_gaussians=True)
    insert_object(scene_representation, obj)
    print("Updating object: {} {}".format(obj['object_name'], obj['object_id']))


def allow_physics(obj):
    '''
    Allow rigid body simulation for the object. (update the object info)
    '''
    obj['rigid_body']['rb_type'] = 'ACTIVE'
    print("Allowing physics for object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def add_fire(scene_representation, obj):
    '''
    Add fire to the object. (add object id to the fire list)
    '''
    scene_representation.fire_objects.append(obj['object_id'])
    print("Adding fire to object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def add_smoke(scene_representation, obj):
    '''
    Add smoke to the object. (add object id to the smoke list)
    '''
    scene_representation.smoke_objects.append(obj['object_id'])
    print("Adding smoke to object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def set_static_animation(obj):
    '''
    Allow animation for the object. (update the object info)
    '''
    obj['animation'] = {
        'type': 'static',
        'points': None
    }
    obj['rigid_body']['rb_type'] = 'KINEMATIC'
    print("Allowing animation for object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def set_moving_animation(obj, points):
    '''
    Set object trajectory given a list of 3D points.
    '''
    obj['animation'] = {
        'type': 'trajectory',
        'points': points
    }
    obj['rigid_body']['rb_type'] = 'KINEMATIC'
    print("Setting trajectory for object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def retrieve_material(scene_representation, material_name):
    '''
    Retrieve material by material name from PolyHaven.
    '''
    material_folder = retrieve_materials_from_polyhaven(material_name)
    # material = Material(material_path=material_folder)
    return material_folder


def init_material():
    '''
    Initialize material by material folder.
    '''
    return Material()


def apply_material(obj, material):
    '''
    Apply material to the object. (convert from class to dict)
    '''
    new_material = {}
    new_material['roughness'] = material.roughness
    new_material['metallic'] = material.metallic
    new_material['specular'] = material.specular
    new_material['material_path'] = material.material_path
    new_material['is_mirror'] = material.is_mirror
    new_material['rgb'] = material.rgb
    obj['material'] = new_material
    print("Applying material to object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def allow_fracture(obj):
    '''
    Fracture object into multiple pieces.
    '''
    obj['fracture'] = True
    print("Fracturing object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def get_object_bottom_position(obj):
    '''
    Get object position at the bottom.
    '''
    # TODO: handle both animated object and static object, also 3DGS object
    return obj['pos']


def get_object_center_position(obj):
    '''
    Get object position at the center.
    '''
    if obj['from_3DGS']:
        center = get_center_of_mesh_2(obj['object_path'])
        bottom_center = get_bottom_center_of_mesh(obj['object_path'])
        z_offset = center[2] - bottom_center[2]
        return obj['pos'] + np.array([0, 0, z_offset])
    else:
        scale = get_scaling_of_mesh(obj['object_path'])
        norm_scale = scale / np.max(scale)
        z_offset = 0.5 * norm_scale[2] * obj['scale']          # offset by half of the height
        return obj['pos'] + np.array([0, 0, z_offset])


def translate_object(obj, translation):
    '''
    Translate object by translation vector.
    '''
    obj['pos'] += translation
    print("Translating object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def rotate_object(obj, rotation):
    '''
    Rotate object by rotation matrix.
    '''
    obj['rot'] = rotation @ obj['rot']
    print("Rotating object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def scale_object(obj, scale):
    '''
    Scale object by scale factor.
    '''
    obj['scale'] *= scale
    print("Scaling object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def get_random_2D_rotation():
    '''
    Get random 2D rotation matrix. (rotation around z-axis, 3x3 matrix)
    '''
    angle = random.uniform(0, 2*math.pi)
    return np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])


def get_random_3D_rotation():
    '''
    Get random 3D rotation matrix. (rotation around x, y, z-axis, 3x3 matrix)
    '''
    rotation = R.random()
    return rotation.as_matrix()


def make_copy(obj):
    '''
    Make a deep copy of the object.
    '''
    new_obj = copy.deepcopy(obj)
    new_obj['object_id'] = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=16))
    return new_obj


def make_break(obj):
    '''
    Break object into multiple pieces.
    '''
    obj['break'] = True
    print("Breaking object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def incinerate(obj):
    '''
    Turn object into ashes.
    '''
    obj['incinerate'] = True
    print("Incinerating object: {} {}".format(obj['object_name'], obj['object_id']))
    return obj


def get_camera_position(scene_representation):
    '''
    Get camera position.
    '''
    return scene_representation.camera_position


def add_event(scene_representation, obj, event_type, start_frame=None, end_frame=None):
    '''
    Add event to the scene.
    '''
    new_event = get_default_event_info()
    new_event['object_id'] = obj['object_id']
    new_event['event_type'] = event_type
    if start_frame is not None:
        new_event['start_frame'] = start_frame
    else:
        new_event['start_frame'] = scene_representation.total_frames // 2 if event_type in ['break', 'incinerate'] else 1
    if end_frame is not None:
        new_event['end_frame'] = end_frame
    else:
        new_event['end_frame'] = scene_representation.total_frames + 1
    scene_representation.events.append(new_event)

    # insert the object if not existed in the scene
    # all_object_ids = [_obj['object_id'] for _obj in scene_representation.inserted_objects]
    # if obj['object_id'] not in all_object_ids:
    #     if obj['from_3DGS']:
    #         update_object(scene_representation, obj)
    #     else:
    #         insert_object(scene_representation, obj)


#############################################################################
#  Additional functions used for autonomous driving scene editing (ChatSim) #
#############################################################################
def get_vehicle_position(scene_representation):
    '''
    Get vehicle position. (since the pose is converted into vehicle coordinate system, the z-value is always 0.0)
    '''
    position = scene_representation.camera_position.copy()
    position[2] = 0.0
    return position


def get_direction(scene_representation, direction='front'):
    '''
    Get 6 directions from the camera position. (front, back, left, right, up, down)
    Camera pose is in OpenCV format (x: right, y: down, z: forward)
    '''
    assert direction in ['up', 'down', 'front', 'back', 'left', 'right']
    R = scene_representation.camera_rotation.copy()
    x_axis, y_axis, z_axis = R[:, 0], R[:, 1], R[:, 2]
    directions = {
        'up': np.array([0, 0, 1]),
        'down': np.array([0, 0, -1]),
        'front': np.cross(np.array([0, 0, 1]), x_axis),
        'back': np.cross(np.array([0, 0, -1]), x_axis),
        'left': np.array(-x_axis),
        'right': np.array(x_axis)
    }
    return directions[direction]


def retrieve_chatsim_asset(scene_representation, object_name):
    '''
    Retrieve 3D asset by object name from chatsim assetbank.
    '''
    assetbank_dict = {
        'Audi_Q3_2023': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Audi_Q3_2023.blend'),
        'Benz_G': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Benz_G.blend'),
        'Benz_S': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Benz_S.blend'),
        'BMW_mini': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/BMW_mini.blend'),
        'Cadillac_CT6': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Cadillac_CT6.blend'),
        'Chevrolet': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Chevrolet.blend'),
        'Dodge_SRT_Hellcat': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Dodge_SRT_Hellcat.blend'),
        'Ferriari_f150': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Ferriari_f150.blend'),
        'Lamborghini': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Lamborghini.blend'),
        'Land_Rover_range_rover': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Land_Rover_range_rover.blend'),
        'M1A2_tank': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/M1A2_tank.blend'),
        'Police_car': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Police_car.blend'),
        'Porsche-911-4s-final': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Porsche-911-4s-final.blend'),
        'Tesla_cybertruck': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Tesla_cybertruck.blend'),
        'Tesla_roadster': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/Tesla_roadster.blend'),
        'Bulldozer': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/obstacles/Bulldozer.blend'),
        'Cement_isolation_pier': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/obstacles/Cement_isolation_pier.blend'),
        'Excavator': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/obstacles/Excavator.blend'),
        'Loader_truck': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/obstacles/Loader_truck.blend'),
        'Red_iron_oil_drum': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/obstacles/Red_iron_oil_drum.blend'),
        'Sign_fence': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/obstacles/Sign_fence.blend'),
        'Traffic_cone': os.path.join(scene_representation.cache_dir, 'blender_assets_chatsim/obstacles/Traffic_cone.blend'),
    }
    assert object_name in assetbank_dict.keys()

    new_obj = get_default_object_info()
    new_obj['object_name'] = object_name
    new_obj['object_id'] = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=16))
    new_obj['object_path'] = assetbank_dict[object_name]
    new_obj['from_3DGS'] = False
    new_obj['forward_axis'] = 'FORWARD_X'
    new_obj['scale'] = 1.0

    return new_obj

#############################################################################


if __name__ == '__main__':
    from scene_representation import SceneRepresentation
    from opt import get_opts
    hparams = get_opts()
    print(hparams)
    safe_state(hparams.quiet)
    scene_representation = SceneRepresentation(hparams)
    
    