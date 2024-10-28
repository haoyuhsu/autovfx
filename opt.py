import sys
import os
from argparse import ArgumentParser


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BLENDER_PATH = os.path.join(ROOT_DIR, "third_parties/Blender/blender-3.6.11-linux-x64/blender")


def get_opts():

    # Set up command line argument parser
    parser = ArgumentParser(description="Editing script parameters")

    # General parameters
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--root_dir", type=str, default="/home/shenlong/Documents/maxhsu/3D-Scene-Editing")
    # parser.add_argument("--blender_path", type=str, default="/home/shenlong/Documents/maxhsu/blender-4.0.2-linux-x64/blender")

    # Dataset parameters
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--white_background", action="store_true", default=False,
                        help="Whether to use white background for rendering")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the output directory")
    # parser.add_argument("--coarse_model_path", type=str, default=None,
    #                     help="Path to the coarse sugar model checkpoint")
    # parser.add_argument("--coarse_mesh_path", type=str, default=None,
    #                     help="Path to the meshes extracted from coarse sugar model (.ply or .obj)")
    parser.add_argument("--max_sh_degree", type=int, default=4)
    parser.add_argument("--gaussians_ckpt_path", type=str, default=None,
                        help="Path to the Gaussian model checkpoint (.pt for SuGaR, .ply for vanilla 3DGS)")

    # Scene editing parameters
    parser.add_argument("--custom_traj_name", default=None, type=str)
    parser.add_argument("--anchor_frame_idx", default=None, type=int)
    parser.add_argument("--edit_text", default="Place an apple on the table.", type=str)
    parser.add_argument("--scene_scale", default=1.0, type=float,
                        help="Scale of the scene in meters in real-world")
    parser.add_argument("--downscale_factor", default=1, type=int,
                        help="Downscale factor for the images (i.e., 1, 2, 4, 8)")
    parser.add_argument("--scene_mesh_path", default=None, type=str,
                        help="Path to the scene meshes (.ply or .obj)")
    parser.add_argument("--reference_image_path", default=None, type=str,
                        help="Path to the reference image for the scene (not implemented yet)")
    parser.add_argument("--waymo_scene", default=False, action="store_true",
                        help="Enable this option if the scene is a Waymo scene")
    
    # Editing rendering parameters
    parser.add_argument("--blender_config_name", default="_tmp_blender_cfg.json", type=str)
    parser.add_argument("--blender_output_dir_name", default="_tmp_blend_results", type=str,
                        help="Specifies the directory where Blender will store the rendering or blending results")
    parser.add_argument("--render_type", default="MULTI_VIEW", type=str, choices=["MULTI_VIEW", "SINGLE_VIEW"],
                        help="Choose 'MULTI_VIEW' to render frames along the entire camera trajectory, or 'SINGLE_VIEW' for static rendering from a single camera position (anchor view).")
    parser.add_argument("--num_frames", default=100, type=int,
                        help="Specifies the number of frames to render (used for 'SINGLE_VIEW' rendering during simulation)")
    parser.add_argument("--is_uv_mesh", action="store_true", default=False,
                        help="Enable this option if the scene meshes have UV textures")
    parser.add_argument("--emitter_mesh_path", default=None, type=str,
                        help="Path to the emitter mesh (.obj) (used for indoor scenes)")
    parser.add_argument("--is_indoor_scene", action="store_true", default=False,
                        help="Enable this option if the scene is an indoor scene")

    # Meshes extraction parameters
    # parser.add_argument("--object_name", default=None, type=str)
    parser.add_argument("--deva_dino_threshold", default=0.7, type=float,
                        help="Increase this threshold to reduce excessive object detection. (0.7 is optimal, but lower to 0.45 for harder-to-detect cases)")

    args = parser.parse_args()

    return args