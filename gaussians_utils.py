import torch
import trimesh
import numpy as np
from sugar.gaussian_splatting.scene.gaussian_model import GaussianModel
from rotation_utils import quaternion_multiply, matrix_to_quaternion, transform_shs
import copy


def check_mesh_path(mesh_path):
    if mesh_path.endswith('.blend'):
        mesh_path = mesh_path.replace('.blend', '.glb')  # TODO: need to convert .blend to .glb
    return mesh_path


# TODO: this function has not been tested
def get_center_of_mesh(mesh_path):
    '''
    Get the center point of the mesh
    '''
    mesh_path = check_mesh_path(mesh_path)
    mesh = trimesh.load_mesh(mesh_path)
    bbox_min, bbox_max = mesh.bounds
    bbox_center = (bbox_min + bbox_max) / 2
    return bbox_center


def get_center_of_mesh_2(mesh_path):
    '''
    Get the center point of the mesh
    '''
    mesh_path = check_mesh_path(mesh_path)
    mesh = trimesh.load_mesh(mesh_path)
    vertices = mesh.vertices
    x_max, x_min, y_max, y_min, z_max, z_min = np.max(vertices[:, 0]), np.min(vertices[:, 0]), np.max(vertices[:, 1]), np.min(vertices[:, 1]), np.max(vertices[:, 2]), np.min(vertices[:, 2])
    bbox_center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2])
    return bbox_center


def get_bottom_center_of_mesh(mesh_path):
    '''
    Get the center point of the bottom of the mesh
    '''
    mesh_path = check_mesh_path(mesh_path)
    mesh = trimesh.load_mesh(mesh_path)
    vertices = mesh.vertices
    x_max, x_min, y_max, y_min, z_max, z_min = np.max(vertices[:, 0]), np.min(vertices[:, 0]), np.max(vertices[:, 1]), np.min(vertices[:, 1]), np.max(vertices[:, 2]), np.min(vertices[:, 2])
    bottom_center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2, z_min])
    return bottom_center


def get_scaling_of_mesh(mesh_path):
    '''
    Get the scaling of the mesh
    '''
    mesh_path = check_mesh_path(mesh_path)
    mesh = trimesh.load_mesh(mesh_path)
    vertices = mesh.vertices
    x_max, x_min, y_max, y_min, z_max, z_min = np.max(vertices[:, 0]), np.min(vertices[:, 0]), np.max(vertices[:, 1]), np.min(vertices[:, 1]), np.max(vertices[:, 2]), np.min(vertices[:, 2])
    scaling = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    return scaling


def load_gaussians(gaussians_path, max_sh_degree=4):
    '''
    Load segmented object gaussians (from .ply file)
    '''
    object_gaussians = GaussianModel(max_sh_degree)  # SuGaR: 4, vanilla 3DGS: 3 (use 4 since we crop from SuGaR)
    object_gaussians.load_ply(gaussians_path)
    return object_gaussians


def merge_two_gaussians(gaussians1, gaussians2, max_sh_degree=4):
    '''
    Merge two gaussians into one
    '''
    new_gaussians = GaussianModel(max_sh_degree)
    new_gaussians._xyz = torch.cat([gaussians1._xyz, gaussians2._xyz], dim=0)
    new_gaussians._features_dc = torch.cat([gaussians1._features_dc, gaussians2._features_dc], dim=0)
    new_gaussians._features_rest = torch.cat([gaussians1._features_rest, gaussians2._features_rest], dim=0)
    new_gaussians._opacity = torch.cat([gaussians1._opacity, gaussians2._opacity], dim=0)
    new_gaussians._scaling = torch.cat([gaussians1._scaling, gaussians2._scaling], dim=0)
    new_gaussians._rotation = torch.cat([gaussians1._rotation, gaussians2._rotation], dim=0)
    return new_gaussians


def transform_gaussians(gaussians, center, rotation, scaling, initial_center):
    '''
    Apply rigid transformation to gaussians (scaling -> rotation -> translation)
    '''
    new_xyz = gaussians._xyz.clone()
    new_rotation = gaussians._rotation.clone()
    new_scales = gaussians._scaling.clone()
    # new_features_rest = gaussians._features_rest.clone()

    # Scale gaussians
    new_xyz -= initial_center.unsqueeze(0)
    new_xyz *= scaling
    new_xyz += initial_center.unsqueeze(0)
    new_scales += np.log(scaling)

    # Rotate gaussians
    new_xyz -= initial_center.unsqueeze(0)
    new_xyz = torch.matmul(new_xyz, rotation.T)
    new_xyz += initial_center.unsqueeze(0)
    new_rotation = quaternion_multiply(matrix_to_quaternion(rotation), new_rotation)
    # new_features_rest = transform_shs(new_features_rest, rotation)

    # Translate gaussians
    translation = center - initial_center
    new_xyz += translation.unsqueeze(0)

    # Update gaussians
    new_gaussians = copy.deepcopy(gaussians)
    new_gaussians._xyz = new_xyz
    new_gaussians._rotation = new_rotation
    new_gaussians._scaling = new_scales
    # new_gaussians._features_rest = new_features_rest

    return new_gaussians
