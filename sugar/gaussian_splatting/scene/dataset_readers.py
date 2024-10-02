#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
import trimesh
from tqdm import tqdm
from utils.graphics_utils import get_ray_directions

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array
    normal: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depth_folder=None, normal_folder=None, image_resolution=1, max_img_size=1920):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        depth_path = os.path.join(depth_folder, image_name + '_depth.npy')
        # depth = np.load(depth_path) if os.path.exists(depth_path) else None
        depth = None
        # normal_path = os.path.join(normal_folder, image_name + '_normal.npy')
        # normal = np.load(normal_path) if os.path.exists(normal_path) else None
        normal_path = os.path.join(normal_folder, image_name + '_normal.png')
        normal = np.array(Image.open(normal_path)) if os.path.exists(normal_path) else None
        if normal is not None:
            # normal = np.transpose(normal, (1, 2, 0))    # (3, H, W) -> (H, W, 3)
            normal = normal.astype(np.float32) / 255.0  # normalize to [0, 1]
            normal = (normal - 0.5) * 2                 # normalize to [-1, 1]
            W2C = getWorld2View2(R, T)
            C2W = np.linalg.inv(W2C)
            normal = normal @ C2W[:3, :3].T             # transform normal to world space

        ##### resize image here to save memory #####
        orig_w, orig_h = image.size
        downscale_factor = 1
        if image_resolution in [1, 2, 4, 8]:
            downscale_factor = image_resolution
        if max(orig_h, orig_w) > max_img_size:
            additional_downscale_factor = max(orig_h, orig_w) / max_img_size
            downscale_factor = additional_downscale_factor * downscale_factor
        resolution = round(orig_w/(downscale_factor)), round(orig_h/(downscale_factor))
        resized_image = image.resize(resolution)
        image = resized_image
        if depth is not None:
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            depth = torch.nn.functional.interpolate(depth, (resolution[1], resolution[0]), mode='bilinear', align_corners=True)
            depth = depth.squeeze(0).squeeze(0).numpy()          # (H, W)
        if normal is not None:
            normal = torch.from_numpy(normal).permute(2, 0, 1).unsqueeze(0)
            normal = torch.nn.functional.interpolate(normal, (resolution[1], resolution[0]), mode='nearest')
            normal = normal.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)

        width, height = resolution  # update resized image size

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              depth=depth, normal=normal)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, depth_path, normal_path, image_resolution=1, llffhold=8, max_img_size=1920, scene_sdf_mesh_path=None, total_points_init=300_000, init_strategy="ray_mesh"):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    depth_dir = 'depth' if depth_path == None else depth_path
    normal_dir = 'normal' if normal_path == None else normal_path
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, \
                    images_folder=os.path.join(path, reading_dir), depth_folder=os.path.join(path, depth_dir), normal_folder=os.path.join(path, normal_dir), image_resolution=image_resolution, max_img_size=max_img_size)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcd_colmap = None
    if init_strategy in ["colmap", "hybrid"]:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd_colmap = fetchPly(ply_path)
        except:
            pcd_colmap = None
        print("[Init] Getting {} points from colmap...".format(pcd_colmap.points.shape[0]))

    # TODO: adjust total_points_init based on the number of points in the colmap point cloud
    if pcd_colmap is None:
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        xyz, rgb, _ = read_points3D_text(txt_path)
        n_colmap_points = xyz.shape[0]
    else:
        n_colmap_points = pcd_colmap.points.shape[0]
    total_points_init = int(2.0 * n_colmap_points)

    # initialize the point cloud from training views with ray-mesh intersection
    pcd_ray_mesh = None
    if init_strategy in ["ray_mesh", "hybrid"]:
        assert scene_sdf_mesh_path is not None, "Please provide the path to the scene mesh for ray-mesh intersection initialization"
        if init_strategy == "hybrid":
            N_TOTAL_POINTS = total_points_init - pcd_colmap.points.shape[0]
        else:
            N_TOTAL_POINTS = total_points_init
        print("[Init] Getting {} points from ray-mesh intersection...".format(N_TOTAL_POINTS))
        if N_TOTAL_POINTS > 0:
            N_POINTS_PER_CAM = N_TOTAL_POINTS // len(cam_infos)
            scene_mesh = trimesh.load_mesh(scene_sdf_mesh_path)
            positions = []
            normals = []
            colors = []
            for cam_info in tqdm(cam_infos, desc="Creating point cloud from ray-mesh intersection"):
                h, w = cam_info.height, cam_info.width
                fy, fx = fov2focal(cam_info.FovY, h), fov2focal(cam_info.FovX, w)
                cx, cy = w / 2, h / 2
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # (3, 3)
                w2c = getWorld2View2(cam_info.R, cam_info.T)  # (4, 4)
                c2w = np.linalg.inv(w2c)
                c2w = torch.FloatTensor(c2w)
                directions = get_ray_directions(h, w, torch.FloatTensor(K), device="cpu", flatten=False)  # (H, W, 3)
                rays_d = directions @ c2w[:3, :3].T
                rays_o = c2w[:3, 3].expand_as(rays_d)
                rays_o = rays_o.reshape(-1, 3).numpy()
                rays_d = rays_d.reshape(-1, 3).numpy()
                # random sample N_POINTS_PER_CAM rays
                indices = np.random.choice(rays_o.shape[0], N_POINTS_PER_CAM, replace=False)
                sampled_rays_o = rays_o[indices]
                sampled_rays_d = rays_d[indices]
                locations, index_ray, index_tri = scene_mesh.ray.intersects_location(
                    ray_origins=sampled_rays_o,
                    ray_directions=sampled_rays_d,
                    multiple_hits=False
                )
                sampled_positions = locations
                sampled_normals = scene_mesh.face_normals[index_tri]
                rgb_colors = np.array(cam_info.image)
                sampled_colors = rgb_colors.reshape(-1, 3)[indices][index_ray] / 255.0
                positions.append(sampled_positions)
                normals.append(sampled_normals)
                colors.append(sampled_colors)
            positions = np.concatenate(positions, axis=0)
            normals = np.concatenate(normals, axis=0)
            colors = np.concatenate(colors, axis=0)
            pcd_ray_mesh = BasicPointCloud(points=positions, colors=colors, normals=normals)

    if init_strategy == "ray_mesh":
        pcd = pcd_ray_mesh
    elif init_strategy == "colmap":
        pcd = pcd_colmap
    elif init_strategy == "hybrid":
        pcd = BasicPointCloud(points=np.concatenate([pcd_colmap.points, pcd_ray_mesh.points], axis=0),
                              colors=np.concatenate([pcd_colmap.colors, pcd_ray_mesh.colors], axis=0),
                              normals=np.concatenate([pcd_colmap.normals, pcd_ray_mesh.normals], axis=0))
        # save memory
        del pcd_ray_mesh
        del pcd_colmap
        
    ply_path = os.path.join(path, "points3D.ply")
    storePly(ply_path, pcd.points, pcd.colors * 255, pcd.normals)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}