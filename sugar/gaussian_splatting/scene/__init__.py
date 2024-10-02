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
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import torch
import numpy as np
from tqdm import tqdm
from scene.cameras import Camera
from utils.graphics_utils import focal2fov


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, args.depth_path, args.normal_path, args.resolution, 
                max_img_size=args.max_img_size, 
                scene_sdf_mesh_path=args.scene_sdf_mesh_path,
                total_points_init=args.total_points_init,
                init_strategy=args.init_strategy
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def loadCustomCameras(self, hparams):

        # get the info of custom camera trajectory
        custom_traj_folder = os.path.join(hparams.source_path, "custom_camera_path")
        with open(os.path.join(custom_traj_folder, hparams.custom_traj_name + '.json'), 'r') as f:
            custom_traj = json.load(f)

        # get camera poses and intrinsics
        fx, fy, cx, cy = custom_traj["fl_x"], custom_traj["fl_y"], custom_traj["cx"], custom_traj["cy"]
        w, h = custom_traj["w"], custom_traj["h"]
        c2w_dict = {}
        for frame in custom_traj["frames"]:
            c2w_dict[frame["filename"]] = np.array(frame["transform_matrix"])
        c2w_dict = dict(sorted(c2w_dict.items()))
        self.c2w_dict = c2w_dict

        # camera list
        custom_cameras = []
        for idx, (filename, c2w) in enumerate(tqdm(c2w_dict.items(), desc="Rendering progress")):
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            FovY = focal2fov(fy, h)
            FovX = focal2fov(fx, w)
            view = Camera(
                colmap_id=1, R=R, T=T, 
                FoVx=FovX, FoVy=FovY, image=None, gt_alpha_mask=None, 
                image_name='{0:05d}'.format(idx), uid=idx)
            custom_cameras.append(view)

        self.custom_cameras = custom_cameras
        
        # store information for blender rendering
        self.img_wh = (w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.c2w = np.array([c2w_dict[frame] for frame in c2w_dict])

        return custom_cameras