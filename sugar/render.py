# Note: render.py not completed yet

import argparse
import os
import numpy as np
import open3d as o3d
import torch
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.general_utils import str2bool
from sugar_utils.spherical_harmonics import SH2RGB
from rich.console import Console
from PIL import Image
import cv2
from tqdm import tqdm


def depth2img(depth, scale=10):
    depth = depth/scale
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return depth_img


def main(args):
    CONSOLE = Console(width=120)
    CONSOLE.print("Rendering RGB, Depth and Normal Maps...")

    n_skip_images_for_eval_split = 8
    n_gaussians_per_surface_triangle = 1

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    rgb_output_dir = os.path.join(output_dir, "rgb")
    os.makedirs(rgb_output_dir, exist_ok=True)
    depth_output_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_output_dir, exist_ok=True)
    normal_output_dir = os.path.join(output_dir, "normal")
    os.makedirs(normal_output_dir, exist_ok=True)

    # --- Vanilla 3DGS parameters ---
    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load
    use_train_test_split = True

    # --- Coarse model parameters ---
    sugar_checkpoint_path = args.coarse_model_path

    # --- Coarse mesh parameters ---
    sugar_mesh_path = args.coarse_mesh_path

    # --- Fine model parameters ---
    refined_model_path = args.refined_model_path

    # torch.cuda.set_device(args.gpu)

    # --- Loading initial 3DGS model ---
    CONSOLE.print(f"Loading the initial 3DGS model from path {gs_checkpoint_path}...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=False,
        eval_split=use_train_test_split,
        eval_split_interval=n_skip_images_for_eval_split,
        )
    
    # --- Loading coarse SuGaR model ---
    CONSOLE.print(f"\nLoading the coarse SuGaR model from path {sugar_checkpoint_path}...")
    checkpoint = torch.load(sugar_checkpoint_path, map_location=nerfmodel.device)
    colors = SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :])
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=colors,
        initialize=True,
        sh_levels=nerfmodel.gaussians.active_sh_degree+1,
        keep_track_of_knn=True,
        knn_to_track=16,
        beta_mode='average',  # 'learnable', 'average', 'weighted_average'
        primitive_types='diamond',  # 'diamond', 'square'
        surface_mesh_to_bind=None,  # Open3D mesh
        )
    sugar.load_state_dict(checkpoint['state_dict'])
    sugar.eval()

    # --- Loading coarse mesh ---
    o3d_mesh = o3d.io.read_triangle_mesh(sugar_mesh_path)
    
    # --- Loading refined SuGaR model ---
    checkpoint = torch.load(refined_model_path, map_location=nerfmodel.device)
    refined_sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
        initialize=False,
        sh_levels=nerfmodel.gaussians.active_sh_degree+1,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode='average',
        surface_mesh_to_bind=o3d_mesh,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        )
    refined_sugar.load_state_dict(checkpoint['state_dict'])
    refined_sugar.eval()

    # --- Rendering RGB, Depth and Normal Maps ---
    with torch.no_grad():
        cameras_to_use = nerfmodel.training_cameras

        for cam_idx in tqdm(range(len(nerfmodel.training_cameras)), desc='Rendering with training cameras'):
            
            # Render RGB image with coarse sugar model
            outputs = sugar.render_image_gaussian_rasterizer( 
                camera_indices=cameras_to_use,
                verbose=False,
                bg_color=None,
                compute_color_in_rasterizer=False,
                compute_covariance_in_rasterizer=True,
                return_2d_radii=True
            )

            rgb = outputs['image']                   # (H, W, 4)
            depth = outputs["depth"]                 # (H, W)
            normal = outputs["normal"]               # (H, W, 3) --> scale [-1, 1]

            # save RGB image
            rgb_image = Image.fromarray(rgb.detach().cpu().numpy().astype(np.uint8))
            rgb_image.save(os.path.join(rgb_output_dir, f"{cam_idx:0>4d}.png"))

            # save depth image (clip it to 0~5)
            depth_image = depth2img(depth.detach().cpu().numpy(), scale=5)
            cv2.imwrite(os.path.join(depth_output_dir, f"{cam_idx:0>4d}.png"), cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR))
            
            # save normal image
            normal = (normal + 1) / 2 # scale to [0, 1]
            normal_image = Image.fromarray((normal.detach().cpu().numpy() * 255).astype(np.uint8))
            normal_image.save(os.path.join(normal_output_dir, f"{cam_idx:0>4d}.png"))


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to render rgb, depth and normal maps.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='path to the scene data to use.')
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-o', '--output_dir',
                        type=str, default=None, 
                        help='path to the output directory.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    parser.add_argument('--coarse_model_path', 
                        type=str, 
                        help='Path to the coarse sugar model.')
    parser.add_argument('--coarse_mesh_path', 
                        type=str, 
                        help='Path to the extracted mesh from coarse sugar model.')
    parser.add_argument('--refined_model_path', 
                        type=str, 
                        help='Path to the refined sugar model.')
    args = parser.parse_args()

    main(args)