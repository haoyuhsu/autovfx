import os
import torch
import torchvision
import numpy as np
import cv2
import math
from opt import get_opts, BLENDER_PATH, ROOT_DIR
import time
import glob
from tqdm import tqdm
import json

from blender import blend_all
# from lighting.ldr2hdr import convert_ldr2hdr
from lighting.difflight import get_envmap_from_single_view
from inpaint.inpaint_anything import inpaint_img

from sugar.sugar_scene.cameras import load_gs_cameras, GSCamera
from sugar.gaussian_splatting.scene.gaussian_model import GaussianModel
# from sugar.gaussian_splatting.scene.cameras import Camera
from sugar.gaussian_splatting.utils.graphics_utils import focal2fov, fov2focal
# from sugar.gaussian_splatting.arguments import PipelineParams
from sugar.sugar_scene.gs_model import PipelineParams, OptimizationParams
from sugar.gaussian_splatting.gaussian_renderer import render
from sugar.gaussian_splatting.render import generate_video_from_frames, depth2img
from sugar.gaussian_splatting.render_panorama import render_panorama
from rich.console import Console

# from blender.static_rendering import run_blender_render as render_all_from_blender
from gaussians_utils import load_gaussians, merge_two_gaussians, transform_gaussians, get_center_of_mesh, get_center_of_mesh_2

from random import randint
from sugar.gaussian_splatting.utils.loss_utils import l1_loss, ssim
from PIL import Image
from sugar.sugar_utils.general_utils import PILtoTorch

import copy

# import importlib
# module_path = 'tracking.Grounded-Segment-Anything.automatic_label_ram_demo'
# ram_demo_module = importlib.import_module(module_path)
# from sugar.gaussian_splatting.gaussian_renderer import get_ray_directions
# import trimesh
# from gpt.gpt4v_utils import estimate_object_scale

from inpaint.retrain_utils import compute_lpips_loss, init_lpips_model, is_large_mask
from sugar.gaussian_splatting.utils.loss_utils import ssim

import open3d as o3d
import trimesh

CONSOLE = Console(width=120)


class SceneRepresentation():

    def __init__(self, hparams):
        self.hparams = hparams
        self.load_scene()
        self.load_cameras()
        
        self.dataset_dir = hparams.source_path
        self.results_dir = hparams.model_path
        os.makedirs(os.path.join(self.results_dir), exist_ok=True)

        custom_traj_name = hparams.custom_traj_name if hparams.custom_traj_name is not None else 'training_cameras'
        
        self.traj_results_dir = os.path.join(self.results_dir, 'custom_camera_path', custom_traj_name)
        os.makedirs(os.path.join(self.traj_results_dir), exist_ok=True)

        self.tracking_results_dir = os.path.join(self.results_dir, 'track_with_deva', custom_traj_name)
        os.makedirs(self.tracking_results_dir, exist_ok=True)

        self.blender_output_dir = os.path.join(self.traj_results_dir, 'blender_output', hparams.blender_output_dir_name)
        os.makedirs(self.blender_output_dir, exist_ok=True)

        self.cache_dir = os.path.join(ROOT_DIR, '_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cfg_path = os.path.join(self.blender_output_dir, hparams.blender_config_name)

        self.custom_traj_name = custom_traj_name
        self.scene_scale = float(hparams.scene_scale) if not hparams.waymo_scene else 1.0
        self.anchor_frame_idx = hparams.anchor_frame_idx if hparams.anchor_frame_idx is not None else 0

        self.inserted_objects = []
        self.fire_objects = []
        self.smoke_objects = []
        self.events = []

        self.blender_cfg = {}
        self.rb_transform_info = None
        self.blender_cache_dir = os.path.join(
            self.cache_dir, 
            'blender_rendering', 
            self.dataset_dir.rstrip('/').split('/')[-1],  # scene name
            self.custom_traj_name
        )

        bg_color = [1,1,1] if self.hparams.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.pipe = PipelineParams()

        self.DINO_THRESHOLD = hparams.deva_dino_threshold

        self.scene_mesh_path_for_blender = hparams.scene_mesh_path

        self.total_frames = self.cameras['c2w'].shape[0] if hparams.render_type == 'MULTI_VIEW' else self.hparams.num_frames
        self.fps = 15

        self.camera_position = self.cameras['c2w'][self.anchor_frame_idx][:3, 3].copy()
        self.camera_rotation = self.cameras['c2w'][self.anchor_frame_idx][:3, :3].copy()

        self.waymo_scene = hparams.waymo_scene


    def insert_object(self, object_info):
        assert isinstance(object_info, dict)
        self.inserted_objects.append(object_info)


    def load_cameras(self):
        '''
        Refernce: loadCustomCameras() in line 104 of sugar/gaussian_splatting/scene/__init__.py
        '''
        # Option 1: Load cameras from custom camera trajectory
        if self.hparams.custom_traj_name is not None:

            # get the info of custom camera trajectory
            custom_traj_folder = os.path.join(self.hparams.source_path, "custom_camera_path")
            with open(os.path.join(custom_traj_folder, self.hparams.custom_traj_name + '.json'), 'r') as f:
                custom_traj = json.load(f)

            # get camera poses and intrinsics
            fx, fy, cx, cy = custom_traj["fl_x"], custom_traj["fl_y"], custom_traj["cx"], custom_traj["cy"]
            w, h = custom_traj["w"], custom_traj["h"]
            c2w_dict = {}
            for frame in custom_traj["frames"]:
                c2w_dict[frame["filename"]] = np.array(frame["transform_matrix"])
            c2w_dict = dict(sorted(c2w_dict.items()))

            if self.hparams.downscale_factor > 1.0:
                h = round(h / self.hparams.downscale_factor)
                w = round(w / self.hparams.downscale_factor)
                fx = fx / self.hparams.downscale_factor
                fy = fy / self.hparams.downscale_factor
                cx = cx / self.hparams.downscale_factor
                cy = cy / self.hparams.downscale_factor

            # camera list
            custom_cameras = []
            for cam_idx, (filename, c2w) in enumerate(tqdm(c2w_dict.items(), desc="Loading custom cameras")):
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]
                FovY = focal2fov(fy, h)
                FovX = focal2fov(fx, w)
                view = GSCamera(
                    colmap_id=cam_idx, R=R, T=T, 
                    FoVx=FovX, FoVy=FovY, image=None, gt_alpha_mask=None, 
                    image_name='{0:05d}'.format(cam_idx), uid=cam_idx,
                    image_height=h, image_width=w)
                custom_cameras.append(view)

            # store information for blender rendering
            self.cameras = {
                'cameras': custom_cameras,
                'img_wh': (w, h),
                'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                'c2w': np.array([c2w_dict[frame] for frame in c2w_dict]),
                'c2w_dict': c2w_dict,
            }

        # Option 2: Load cameras from dataset
        else:
            camera_list = load_gs_cameras(self.hparams.source_path, self.hparams.model_path, self.hparams.downscale_factor)
            tmp_camera = camera_list[0]
            h, w = tmp_camera.image_height, tmp_camera.image_width
            cx, cy = w / 2, h / 2
            fx, fy = fov2focal(tmp_camera.FoVx, w), fov2focal(tmp_camera.FoVy, h)
            c2w_list = []
            c2w_dict = {}
            for cam in camera_list:
                c2w = np.zeros((4,4))
                c2w[:3, :3] = cam.R.transpose()
                c2w[:3, 3] = cam.T
                c2w[3, 3] = 1.0
                c2w_list.append(c2w)
                c2w_dict[cam.image_name + '.png'] = c2w
            self.cameras = {
                'cameras': camera_list,
                'img_wh': (w, h),
                'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                'c2w': np.array(c2w_list),
                'c2w_dict': c2w_dict,
            }


    def load_scene(self):
        '''
        Reference: convert_refined_sugar_into_gaussians() in line 2617 of sugar/sugar_scene/sugar_model.py
        '''
        if self.hparams.gaussians_ckpt_path.endswith('.pt'):
            # Load gaussians parameters from sugar checkpoint
            CONSOLE.print(f"\nLoading the coarse SuGaR model from path {self.hparams.gaussians_ckpt_path}...")
            gaussians = GaussianModel(self.hparams.max_sh_degree)
            checkpoint = torch.load(self.hparams.gaussians_ckpt_path, map_location=gaussians.get_xyz.device)
            with torch.no_grad():
                xyz = checkpoint['state_dict']['_points'].cpu().numpy()
                opacities = checkpoint['state_dict']['all_densities'].cpu().numpy()
                features_dc = checkpoint['state_dict']['_sh_coordinates_dc'].cpu().numpy()
                features_extra = checkpoint['state_dict']['_sh_coordinates_rest'].cpu().numpy()
                scales = checkpoint['state_dict']['_scales'].cpu().numpy()
                rots = checkpoint['state_dict']['_quaternions'].cpu().numpy()
            _set_require_grad = False
            gaussians._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(_set_require_grad))
            gaussians._features_dc = torch.nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(_set_require_grad))
            gaussians._features_rest = torch.nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(_set_require_grad))
            gaussians._opacity = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(_set_require_grad))
            gaussians._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(_set_require_grad))
            gaussians._rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(_set_require_grad))
            gaussians.active_sh_degree = self.hparams.max_sh_degree
        if self.hparams.gaussians_ckpt_path.endswith('.ply'):
            # Load gaussians parameters from vanilla 3DGS checkpoint
            CONSOLE.print(f"\nLoading the vanilla 3DGS model from path {self.hparams.gaussians_ckpt_path}...")
            gaussians = GaussianModel(self.hparams.max_sh_degree - 1)  # SuGaR: 4, vanilla 3DGS: 3
            gaussians.load_ply(self.hparams.gaussians_ckpt_path)
        self.gaussians = gaussians


    def render_scene(self, skip_render_3DGS=False):
        self.render_from_blender()
        if (
            not skip_render_3DGS or
            self.rb_transform_info is not None or
            os.path.exists(os.path.join(self.blender_output_dir, 'melting_meshes'))
        ):
            self.render_from_3DGS(post_rendering=True)
        blend_all.blend_frames(self.blender_output_dir, self.cfg_path)


    def save_cfg(self, cfg, cfg_path):
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=4)


    def set_basic_blender_cfg(self):
        new_cfg = {}
        new_cfg['edit_text'] = self.hparams.edit_text
        new_cfg['blender_cache_dir'] = self.blender_cache_dir
        new_cfg['im_width'], new_cfg['im_height'] = self.cameras['img_wh']
        new_cfg['K'] = self.cameras['K'].tolist()
        new_cfg['c2w'] = self.cameras['c2w'].tolist()
        new_cfg['scene_mesh_path'] = self.scene_mesh_path_for_blender
        new_cfg['is_uv_mesh'] = self.hparams.is_uv_mesh
        new_cfg['output_dir_name'] = self.hparams.blender_output_dir_name
        new_cfg['render_type'] = self.hparams.render_type
        new_cfg['num_frames'] = self.hparams.num_frames
        new_cfg['anchor_frame_idx'] = self.anchor_frame_idx
        new_cfg['emitter_mesh_path'] = self.hparams.emitter_mesh_path
        new_cfg['is_indoor_scene'] = self.hparams.is_indoor_scene
        new_cfg['waymo_scene'] = self.waymo_scene
        self.blender_cfg.update(new_cfg)


    def render_from_blender(self):
        self.set_basic_blender_cfg()
        hdr_env_map_path, sun_dir = self.render_global_env_map()
        self.blender_cfg['global_env_map_path'] = hdr_env_map_path
        self.blender_cfg['sun_dir'] = sun_dir.tolist() if sun_dir is not None else None
        self.blender_cfg['insert_object_info'] = []
        for obj in self.inserted_objects:
            obj['pos'] = obj['pos'].tolist()
            obj['rot'] = obj['rot'].tolist()
            if obj['material'] is not None and obj['material']['rgb'] is not None:
                obj['material']['rgb'] = obj['material']['rgb'].tolist()
            if obj['animation'] is not None and obj['animation']['type'] == 'trajectory':
                obj['animation']['points'] = [point.tolist() for point in obj['animation']['points']]
            self.blender_cfg['insert_object_info'].append(obj)
        self.blender_cfg['fire_objects'] = self.fire_objects
        self.blender_cfg['smoke_objects'] = self.smoke_objects
        self.blender_cfg['events'] = self.events
        self.save_cfg(self.blender_cfg, self.cfg_path)
        torch.cuda.empty_cache()    # release gpu memory for blender
        os.system('{} --background --python ./blender/all_rendering.py -- --input_config_path={}'.format( \
            BLENDER_PATH, self.cfg_path
        ))

        # check if rigid body transform is added to the blender config
        with open(self.cfg_path, 'r') as f:
            self.blender_cfg = json.load(f)
        if 'rb_transform' in self.blender_cfg:
            self.rb_transform_info = self.blender_cfg['rb_transform']


    # def render_local_env_map(self, origin):
    #     origin = torch.FloatTensor(origin)
    #     env_map_dir = os.path.join(self.results_dir, 'panorama', str(math.floor(time.time())))    # use current timestamp as the name of the env map
    #     ldr_env_map_path = render_panorama(self.gaussians, self.pipe, self.background, origin, env_map_dir)
    #     ldr_env_map_path = inpaint_img(ldr_env_map_path)
    #     hdr_env_map_path = convert_ldr2hdr(ldr_env_map_path)
    #     return hdr_env_map_path
    

    def render_global_env_map(self):
        assert self.anchor_frame_idx is not None   # anchor frame index must be specified
        image_path = os.path.join(self.traj_results_dir, 'images', '{0:05d}.png'.format(self.anchor_frame_idx))
        output_dir = os.path.join(self.results_dir, 'hdr', self.hparams.custom_traj_name)
        c2w = self.cameras['c2w'][self.anchor_frame_idx]
        hdr_env_map_path = os.path.join(output_dir, '{0:05d}_rotate.exr'.format(self.anchor_frame_idx))
        if not os.path.exists(hdr_env_map_path):
            hdr_env_map_path = get_envmap_from_single_view(image_path, output_dir, c2w)
        else:
            print('HDR environment map already exists, skip rendering...')
        # TODO: get the sunlight direction for waymo scenes
        sun_dir = None
        if self.waymo_scene:
            ev_image_path = os.path.join(output_dir, 'envmap', '{0:05d}_ev-50.png'.format(self.anchor_frame_idx))
            sun_dir = self.get_sunlight_direction(ev_image_path, c2w)
            print('Sunlight direction: ', sun_dir)
        return hdr_env_map_path, sun_dir
    

    def get_sunlight_direction(self, img_path, c2w):
        image = Image.open(img_path).convert('L')
        # image = image.filter(ImageFilter.GaussianBlur(3))
        image = np.array(image)
        max_index = np.unravel_index(np.argmax(image), image.shape)   # Find the index of the maximum intensity value
        y, x = max_index                                              # max_index will contain the (y, x) coordinates of the pixel with the highest intensity
        h, w = image.shape
        theta = (x / w) * 2 * np.pi                                   # convert to spherical coordinates
        phi = (y / h) * np.pi
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        dir_vector = np.array([x, y, z])
        dir_vector = dir_vector / np.linalg.norm(dir_vector)
        dir_vector = c2w[:3, :3] @ dir_vector                         # rotate the direction vector to the world coordinate
        dir_vector = dir_vector / np.linalg.norm(dir_vector)
        dir_vector = -dir_vector
        return dir_vector


    def render_from_3DGS(self, render_video=False, post_rendering=False):

        self.load_scene()  # reload the scene to get the latest gaussians

        camera_views = self.cameras['cameras']  # a list of Camera objects

        if post_rendering and self.hparams.render_type == 'SINGLE_VIEW':
            camera_views = [copy.deepcopy(self.cameras['cameras'][self.anchor_frame_idx]) for _ in range(self.total_frames)]
            for cam_idx, view in enumerate(camera_views):
                camera_views[cam_idx].image_name = '{0:05d}'.format(cam_idx)

        render_path = os.path.join(self.traj_results_dir, "images")
        os.makedirs(render_path, exist_ok=True)
        depth_path = os.path.join(self.traj_results_dir, "depth")
        os.makedirs(depth_path, exist_ok=True)
        normal_path = os.path.join(self.traj_results_dir, "normal")
        os.makedirs(normal_path, exist_ok=True)

        with torch.no_grad():
            for idx, view in tqdm(enumerate(camera_views), desc="Rendering progress"):
                if self.rb_transform_info is not None:
                    all_gaussians = copy.deepcopy(self.gaussians)
                    for obj_id, obj_rb_info in self.rb_transform_info.items():
                        if "{0:03d}".format(idx + 1) not in obj_rb_info:
                            continue
                        rb_transform = obj_rb_info["{0:03d}".format(idx + 1)]  # frame index starts from 001
                        obj_info = [obj for obj in self.blender_cfg['insert_object_info'] if obj['object_id'] == obj_id][0]
                        obj_gaussians_path = os.path.join('/'.join(obj_info['object_path'].split('/')[:-2]), 'object_gaussians.ply')
                        center = torch.Tensor(rb_transform['pos']).cuda()
                        rotation = torch.Tensor(rb_transform['rot']).cuda()
                        scaling = rb_transform['scale']
                        initial_center = torch.Tensor(get_center_of_mesh_2(obj_info['object_path'])).cuda()
                        object_gaussians = load_gaussians(obj_gaussians_path, self.hparams.max_sh_degree - 1)
                        transformed_gaussians = transform_gaussians(object_gaussians, center, rotation, scaling, initial_center)
                        all_gaussians = merge_two_gaussians(all_gaussians, transformed_gaussians)
                elif os.path.exists(os.path.join(self.blender_cache_dir, self.hparams.blender_output_dir_name, 'melting_meshes')):
                    all_gaussians = copy.deepcopy(self.gaussians)
                    mesh_output_dir = os.path.join(self.blender_cache_dir, self.hparams.blender_output_dir_name, 'melting_meshes')
                    for obj_id in sorted(os.listdir(mesh_output_dir)):
                        melting_mesh_dir = os.path.join(mesh_output_dir, obj_id)
                        obj_info = [
                            obj for obj in self.blender_cfg['insert_object_info']
                            if obj['object_id'] == obj_id
                        ][0]
                        orig_mesh_path = obj_info['object_path']
                        orig_gaussians_path = os.path.join('/'.join(orig_mesh_path.split('/')[:-2]), 'object_gaussians.ply')
                        orig_mesh = trimesh.load_mesh(orig_mesh_path)
                        orig_gaussians = load_gaussians(orig_gaussians_path, self.hparams.max_sh_degree - 1)
                        # associate closest triangle in the original mesh to each Gaussian center
                        orig_mesh_o3d = o3d.t.geometry.RaycastingScene()
                        orig_mesh_o3d.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(orig_mesh.as_open3d))
                        gaussians_xyz = orig_gaussians._xyz.detach().cpu().numpy()
                        ret_dict = orig_mesh_o3d.compute_closest_points(
                            o3d.core.Tensor.from_numpy(gaussians_xyz.astype(np.float32))
                        )
                        triangle_ids_from_gaussians = ret_dict['primitive_ids'].cpu().numpy()
                        # iterate over the melting meshes
                        melting_mesh_paths = [
                            os.path.join(melting_mesh_dir, '{0:03d}_obj.stl'.format(idx + 1)),
                            os.path.join(melting_mesh_dir, '{0:03d}_obj_dup.stl'.format(idx + 1))
                        ]
                        for melting_mesh_path in melting_mesh_paths:
                            if not os.path.exists(melting_mesh_path):
                                continue
                            melting_mesh = trimesh.load_mesh(melting_mesh_path)  # meet ValueError: PLY is unexpected length!
                            # melting_mesh = o3d.io.read_triangle_mesh(melting_mesh_path)
                            # associate closest triangle in the original mesh to each vertex in the melting mesh
                            ret_dict = orig_mesh_o3d.compute_closest_points(
                                o3d.core.Tensor.from_numpy(np.array(melting_mesh.triangles_center).astype(np.float32))
                            )
                            # ret_dict = orig_mesh_o3d.compute_closest_points(
                            #     o3d.core.Tensor.from_numpy(np.array(melting_mesh.vertices).astype(np.float32))
                            # )
                            triangle_ids_from_melting = ret_dict['primitive_ids'].cpu().numpy()
                            # keep the Gaussians sharing the same closest triangle with the melting mesh
                            matching_gaussians_mask = np.isin(triangle_ids_from_gaussians, triangle_ids_from_melting)
                            # create new Gaussians and merge the new Gaussians with the existing ones
                            new_gaussians = copy.deepcopy(orig_gaussians)
                            new_gaussians._xyz = orig_gaussians._xyz[matching_gaussians_mask]
                            new_gaussians._features_dc = orig_gaussians._features_dc[matching_gaussians_mask]
                            new_gaussians._features_rest = orig_gaussians._features_rest[matching_gaussians_mask]
                            new_gaussians._scaling = orig_gaussians._scaling[matching_gaussians_mask]
                            new_gaussians._rotation = orig_gaussians._rotation[matching_gaussians_mask]
                            new_gaussians._opacity = orig_gaussians._opacity[matching_gaussians_mask]
                            all_gaussians = merge_two_gaussians(all_gaussians, new_gaussians)
                else:
                    all_gaussians = self.gaussians
                result = render(view, all_gaussians, self.pipe, self.background)
                # rgb image
                rgba_img = result["render"]
                torchvision.utils.save_image(rgba_img, os.path.join(render_path, view.image_name + ".png"))
                # depth map
                depth_raw = result["depth"].cpu().numpy()
                depth_raw = depth_raw.squeeze()
                np.save(os.path.join(depth_path, view.image_name + ".npy"), depth_raw.astype(np.float32))
                depth_img = depth2img(depth_raw, scale=3.0)
                cv2.imwrite(os.path.join(depth_path, view.image_name + ".png"), depth_img)
                # normal map
                normal = result["normal"].cpu().numpy()
                normal = (normal + 1) / 2
                normal = (normal * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(normal_path, view.image_name + ".png"), cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))

        # generate video from frames
        if render_video:
            rgb_frames_path = sorted(glob.glob(os.path.join(render_path, '*.png')))
            generate_video_from_frames(rgb_frames_path, os.path.join(self.traj_results_dir, 'render_rgb.mp4'), fps=15)
            depth_frames_path = sorted(glob.glob(os.path.join(depth_path, '*.png')))
            generate_video_from_frames(depth_frames_path, os.path.join(self.traj_results_dir, 'render_depth.mp4'), fps=15)
            normal_frames_path = sorted(glob.glob(os.path.join(normal_path, '*.png')))
            generate_video_from_frames(normal_frames_path, os.path.join(self.traj_results_dir, 'render_normal.mp4'), fps=15)


    # def estimate_scene_scale(self):
    #     """
    #     Estimate the scale of the scene (compared to the real world scale) using the following steps:
    #     (1) Segment the anchor frame using RAM
    #     (2) Get the median depth value of each segmented object
    #     (3) Unproject the mask to 3D space with median depth to get estimated distance
    #     (4) Estimate the scale of the object using GPT-4V
    #     (5) Calculate the scene scale by dividing the estimated scale by the estimated distance
    #     """
    #     # store the estimated scene scale in .txt file
    #     scene_scale_txt_path = os.path.join('scene_scale_logs.txt')
    #     dataset_name = self.hparams.source_path.split('/')[-2]
    #     with open(scene_scale_txt_path, 'a') as f:
    #         f.write(f"===== Scene: {dataset_name}, Trajectory: {self.custom_traj_name} =====\n\n")

    #     anchor_frame_path = os.path.join(self.traj_results_dir, 'images', '{0:05d}.png'.format(self.anchor_frame_idx))
    #     ram_seg_result_dir = os.path.join(self.traj_results_dir, 'ram_segment')
    #     ram_demo_module.run_ram_segmentation(anchor_frame_path, ram_seg_result_dir)
    #     with open(os.path.join(ram_seg_result_dir, 'label.json'), 'r') as f:
    #         seg_info = json.load(f)
    #     print("Number of objects segmented in the scene: ", len(seg_info['mask']))
    #     w, h = scene_representation.cameras['img_wh']
    #     K = scene_representation.cameras['K']
    #     directions = get_ray_directions(h, w, torch.FloatTensor(K), device="cuda", flatten=False)  # (H, W, 3)
    #     scene_mesh = trimesh.load_mesh(scene_representation.hparams.scene_mesh_path)
    #     scene_scale_list = []
    #     for obj in seg_info['mask']:
    #         if 'box' in obj:
    #             obj_index = obj['value']
    #             obj_label = obj['label']
    #             obj_logit = obj['logit']
    #             x_min, y_min, x_max, y_max = obj['box']  # x is width, y is height
    #             mask_img = cv2.imread(os.path.join(ram_seg_result_dir, str(obj_index), 'mask.jpg'), cv2.IMREAD_GRAYSCALE)
    #             mask = mask_img > 0
    #             # reject bounding box that is too small or has it side near the corner
    #             MINIMUM_BBOX_SIZE = 100 * 100
    #             bbox_size = (x_max - x_min) * (y_max - y_min)
    #             if bbox_size < MINIMUM_BBOX_SIZE:
    #                 continue
    #             MINIMUM_CORNER_SIZE = 15
    #             if x_min < MINIMUM_CORNER_SIZE or y_min < MINIMUM_CORNER_SIZE or w - x_max < MINIMUM_CORNER_SIZE or h - y_max < MINIMUM_CORNER_SIZE:
    #                 continue
    #             # get the median depth value of the mask
    #             c2w = scene_representation.cameras['c2w_dict']['{0:05d}.png'.format(self.anchor_frame_idx)]
    #             c2w = torch.FloatTensor(c2w).to("cuda")
    #             rays_d = directions @ c2w[:3, :3].T
    #             rays_o = c2w[:3, 3].expand_as(rays_d)
    #             rays_d = rays_d.cpu().numpy()
    #             rays_o = rays_o.cpu().numpy()
    #             ray_directions = rays_d[mask].reshape(-1, 3)
    #             ray_origins = rays_o[mask].reshape(-1, 3)
    #             locations, index_ray, index_tri = scene_mesh.ray.intersects_location(
    #                 ray_origins=ray_origins,
    #                 ray_directions=ray_directions,
    #                 multiple_hits=False
    #             )
    #             w2c = torch.inverse(c2w).cpu().numpy()
    #             xyz = locations
    #             xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    #             xyz = np.matmul(xyz, w2c.T)
    #             depth = xyz[:, 2]
    #             median_depth = np.median(depth)
    #             ### Option 1: unproject the leftmost and rightmost points of the mask to 3D space ###
    #             # non_zero_coord = np.where(mask)  # 0: x, 1: y
    #             # sort_idx = np.argsort(non_zero_coord[1])  # from small y to large y
    #             # left_points_x, left_points_y = non_zero_coord[0][sort_idx[0]], non_zero_coord[1][sort_idx[0]]
    #             # right_points_x, right_points_y = non_zero_coord[0][sort_idx[-1]], non_zero_coord[1][sort_idx[-1]]
    #             # left_rays_o, left_rays_d = rays_o[left_points_x, left_points_y], rays_d[left_points_x, left_points_y]
    #             # right_rays_o, right_rays_d = rays_o[right_points_x, right_points_y], rays_d[right_points_x, right_points_y]
    #             # left_points = left_rays_o + left_rays_d * median_depth
    #             # right_points = right_rays_o + right_rays_d * median_depth
    #             # distance = np.linalg.norm(left_points - right_points)
    #             ### Option 2: unproject the whole mask to 3D space ###
    #             points3D = rays_o[mask] + rays_d[mask] * median_depth
    #             xyz_max, xyz_min = np.max(points3D, axis=0), np.min(points3D, axis=0)
    #             max_scale = np.max(xyz_max - xyz_min)
    #             distance = max_scale
    #             # get estimated scale from GPT-4V
    #             bbox_img_path = os.path.join(ram_seg_result_dir, str(obj_index), 'bbox.jpg')
    #             estimated_scale = estimate_object_scale(bbox_img_path, obj_label)
    #             with open(scene_scale_txt_path, 'a') as f:
    #                 f.write(f"object name: {obj_label}, estimated scale: {estimated_scale}, estimated distance: {distance}, estimated median depth: {median_depth}, estimated scene scale: {estimated_scale / distance}\n")
    #             # print(f"Object {obj_label} has estimated scale: {estimated_scale} meters")
    #             # print(f"Object {obj_label} has estimated distance: {distance} meters")
    #             # print(f"Object {obj_label} has estimated median depth: {median_depth} meters")
    #             # print(f"Object {obj_label} has estimated scene scale: {estimated_scale / distance} meters")
    #             scene_scale_list.append(estimated_scale / distance)
    #     # print("Estimated scene scale: ", np.median(scene_scale_list))

    #     # store the estimated scene scale in .txt file
    #     scene_scale_txt_path = os.path.join('scene_scale_logs.txt')
    #     dataset_name = os.path.basename(self.hparams.source_path).split('/')[-1]
    #     with open(scene_scale_txt_path, 'a') as f:
    #         f.write(f"*** average estimated scale: {np.mean(scene_scale_list)} ***\n")
    #         f.write(f"*** median estimated scale: {np.median(scene_scale_list)} ***\n\n")



    def training_3DGS_for_inpainting(self, gaussians_path, image_dir, mask_dir, output_dir, transforms_path):
        gaussians = GaussianModel(self.hparams.max_sh_degree - 1)
        gaussians.load_ply(gaussians_path)
        opt = OptimizationParams()
        pipe = PipelineParams()
        gaussians.training_setup(opt)

        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

        # get training cameras
        cameraList = []
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        fx, fy, cx, cy = transforms["fl_x"], transforms["fl_y"], transforms["cx"], transforms["cy"]
        w, h = transforms["w"], transforms["h"]
        for idx, info in tqdm(enumerate(transforms["frames"]), desc="Loading custom cameras"):
            filename = info["filename"]
            c2w = np.array(info["transform_matrix"])
            w2c = np.linalg.inv(np.array(c2w))
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]
            FovY = focal2fov(fy, h)
            FovX = focal2fov(fx, w)
            image = Image.open(os.path.join(image_dir, filename))
            image = PILtoTorch(image, (w, h))
            view = GSCamera(
                colmap_id=idx, R=R, T=T, 
                FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None, 
                image_name=filename, uid=idx,
                image_height=h, image_width=w)
            cameraList.append(view)

        LPIPS = init_lpips_model()

        viewpoint_stack = None
        first_iter = 0
        last_iter = 2000  # make it shorter to prevent overfit, original: 5000
        for iteration in tqdm(range(first_iter, last_iter + 1), desc="Re-training for inpainting progress"):

            gaussians.update_learning_rate(iteration)

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = cameraList.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            result = render(viewpoint_cam, gaussians, pipe, self.background)
            image, viewspace_point_tensor, visibility_filter, radii = \
                result["render"], \
                result["viewspace_points"], \
                result["visibility_filter"], \
                result["radii"]
            
            # get the boolean mask
            mask2d = None
            mask2d_path = os.path.join(mask_dir, viewpoint_cam.image_name)
            if os.path.exists(mask2d_path):
                mask2d = Image.open(mask2d_path)
                mask2d = torch.from_numpy(np.array(mask2d) / 255.0).unsqueeze(0).cuda()
                mask2d = mask2d.repeat(4, 1, 1)
                mask2d = mask2d > 0.0
            
            # RGB Loss and LPIPS Loss (adpted from gaussian grouping)
            gt_image = viewpoint_cam.original_image.cuda()
            if mask2d is None or not is_large_mask(mask2d):   # use L1-loss if no mask provided or the mask is not large enough
                loss_rgb = l1_loss(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * loss_rgb + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            else:
                loss_rgb = l1_loss(image[~mask2d], gt_image[~mask2d])
                loss_lpips = compute_lpips_loss(LPIPS, image[:3, ...], gt_image[:3, ...], mask2d[0, ...])
                loss = (1.0 - opt.lambda_dssim) * loss_rgb + opt.lambda_dssim * loss_lpips
            loss.backward()

            with torch.no_grad():
                # Densification
                if iteration < 5000:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration % 300 == 0:
                        size_threshold = 20
                        min_opacity = 0.1    # 0.005 would create floaters due to multi-view inconsistency in inpainting
                        gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, 1.1, size_threshold)  # 1.1 since we normalize the camera in sdf rendering (1 * 1.1)
                        # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                #     if iteration % opt.opacity_reset_interval == 0:
                #         gaussians.reset_opacity()

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

        # save the gaussians to .ply file
        gaussians.save_ply(os.path.join(output_dir, 'inpaint_gaussians.ply'))

        
if __name__ == '__main__':
    hparams = get_opts()
    scene_representation = SceneRepresentation(hparams)

    ##### Test rendering from Blender #####
    # scene_representation.render_scene()

    # with open(scene_representation.cfg_path, 'r') as f:
    #     scene_representation.blender_cfg = json.load(f)
    #     scene_representation.rb_transform_info = scene_representation.blender_cfg['rb_transform']
    # scene_representation.render_scene()

    ##### Test rendering from 3DGS #####
    scene_representation.load_scene()
    scene_representation.render_from_3DGS(render_video=True)

    ##### Pre-render all environment map #####
    # scene_representation.render_global_env_map()

    ##### Estimate scene scale #####
    # scene_representation.estimate_scene_scale()

    ##### Test mesh extraction #####
    # TEXT_PROMPT = 'bulldozer'
    # # scene_representation.render_from_3DGS()
    # # from tracking.demo_with_text import run_deva
    # # run_deva(os.path.join(scene_representation.traj_results_dir, 'images'), scene_representation.tracking_results_dir, TEXT_PROMPT, scene_representation.DINO_THRESHOLD)
    # from extract.extract_object import extract_object_from_scene, inpaint_object
    # id = str([x for x in os.listdir(os.path.join(scene_representation.tracking_results_dir, '_'.join(TEXT_PROMPT.split(' ')))) if x.isdigit()][0])
    # # extract_object_from_scene(scene_representation, TEXT_PROMPT, id)
    # inpaint_object(scene_representation, TEXT_PROMPT, id)
    
    # save_dir = os.path.join(scene_representation.results_dir, 'object_instance', scene_representation.custom_traj_name, '_'.join(TEXT_PROMPT.split(' ')), id)
    # scene_representation.training_3DGS_for_inpainting(
    #     os.path.join(save_dir, 'removal_gaussians.ply'),
    #     os.path.join(save_dir, 'render_inpaint_lama'),
    #     os.path.join(save_dir, 'render_inpaint_mask'),
    #     save_dir,
    #     os.path.join(save_dir, 'inpaint_camera_poses.json')
    # )

    # gaussians = GaussianModel(scene_representation.hparams.max_sh_degree - 1)
    # gaussians.load_ply(os.path.join(save_dir, 'inpaint_gaussians.ply'))
    # scene_representation.gaussians = gaussians
    # scene_representation.render_from_3DGS(render_video=True)

    ##### Test rigid body simulation of existing objects in the scene #####
    # object_gaussians_path = 'output/garden_norm_aniso_0.1_pseudo_normal_0.01_alpha_0.0/object_instance/vase_with_flowers/18040383/object_gaussians.ply'
    # object_gaussians = load_gaussians(object_gaussians_path, scene_representation.hparams.max_sh_degree - 1)
    # rb_transform_file_path = 'output/garden_norm_aniso_0.1_pseudo_normal_0.01_alpha_0.0/custom_camera_path/transforms_001/blend_results_vase_drop/rb_transform.json'
    # with open(rb_transform_file_path, 'r') as f:
    #     rb_transform = json.load(f)
    # scene_representation.rb_transform_info = rb_transform
    # scene_representation.object_mesh_path = 'output/garden_norm_aniso_0.1_pseudo_normal_0.01_alpha_0.0/object_instance/vase_with_flowers/18040383/object_mesh/object_mesh.obj'
    # scene_representation.object_gaussians = object_gaussians
    # scene_representation.render_from_3DGS()