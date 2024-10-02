import numpy as np
import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from scene import Scene
from scene.cameras import Camera
from utils.py360_utils import c2e
from PIL import Image
import math
# from inpaint.utils import dilate_mask


# def dilate_image(img, alpha_threshold=0.7, dilate_kernel_size=50):
#     ALPHA_THRESHOLD = alpha_threshold
#     mask = img[..., 3] < ALPHA_THRESHOLD * 255 # thresholding
#     mask = mask.astype(np.uint8) * 255
#     mask = dilate_mask(mask, dilate_kernel_size)
#     mask = mask.astype(np.float32) / 255
#     img[..., 3] *= 1. - mask
#     return img


def create_camera(view_name, colmap_id, lookat, scene_up, center, FOV, image=None, gt_alpha_mask=None, data_device="cuda", zfar=100.0, znear=0.01):
    right = np.cross(lookat, scene_up)
    down = np.cross(lookat, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([right, down, lookat]).T
    c2w[:3, 3] = center
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    return Camera(colmap_id=colmap_id, R=R, T=T, FoVx=FOV, FoVy=FOV, 
                image=image, gt_alpha_mask=gt_alpha_mask, image_name=view_name, 
                uid=colmap_id, data_device=data_device, zfar=zfar, znear=znear)


def create_surrounding_views(center, IMG_SIZE=1024):
    '''
    Create N views for rendering a panorama
    ### sample views at 5 different phi angles (-90, -45, 0, 45, 90)
    ### for -90 and 90, only sample 1 view
    ### for -45, 0, 45, sample N views at different theta angles
    
    Notes: this method has more artifacts in overlapping regions than create_cube_map_views
    '''
    Z_FAR = 100.0
    Z_NEAR = 0.01
    FOV = math.pi / 4
    ASPECT_RATIO = 1.0
    image = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    
    views = {
        "phi90"  : create_camera("up",   0, np.array([0, 0, 1]),  np.array([-1, 0, 0]), center, FOV, image, z_near=0.01),  # avoid floaters
        "phi-90" : create_camera("down", 1, np.array([0, 0, -1]), np.array([1, 0, 0]),  center, FOV, image),
    }
    phi_angles = [-45, 0, 45]
    theta_angles = [-135, -90, -45, 0, 45, 90, 135, 180]
    view_idx = len(views)
    for phi in phi_angles:
        for theta in theta_angles:
            view_name = "phi{}_theta{}".format(phi, theta)
            phi_rad = math.radians(phi)
            theta_rad = math.radians(theta)
            lookat = np.array([math.cos(phi_rad) * math.cos(theta_rad), math.cos(phi_rad) * math.sin(theta_rad), math.sin(phi_rad)])
            lookat = lookat / np.linalg.norm(lookat)
            views[view_name] = create_camera(view_name, view_idx, lookat, np.array([0, 0, 1]), center, FOV, image)
            view_idx += 1

    return views


def create_cube_map_views(center, IMG_SIZE=1024):
    '''
    Create 6 views for rendering a cubemap
    ### sample views into 6 directions to form a skybox cubemap
    ### front: +X, back: -X, left: +Y, right: -Y, up: +Z, down: -Z
    '''
    Z_FAR = 100.0
    Z_NEAR = 0.01
    FOV = math.pi / 2
    ASPECT_RATIO = 1.0
    image = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    
    views = {
        "front": create_camera("front", 0, np.array([1, 0, 0]),  np.array([0, 0, 1]),  center, FOV, image),
        "back" : create_camera("back",  1, np.array([-1, 0, 0]), np.array([0, 0, 1]),  center, FOV, image),
        "left" : create_camera("left",  2, np.array([0, 1, 0]),  np.array([0, 0, 1]),  center, FOV, image),
        "right": create_camera("right", 3, np.array([0, -1, 0]), np.array([0, 0, 1]),  center, FOV, image),
        "up"   : create_camera("up",    4, np.array([0, 0, 1]),  np.array([-1, 0, 0]), center, FOV, image),
        "down" : create_camera("down",  5, np.array([0, 0, -1]), np.array([1, 0, 0]),  center, FOV, image),
    }

    return views


def render_panorama(gaussians, pipeline, background, center, output_dir, pano_h=1024, pano_w=2048):
    '''
    Render a panorama with a specific center
    Args:
        gaussians: GaussianModel object
        pipeline: Pipeline object
        background: torch.Tensor, background color
        center: np.array, center of the panorama
        output_dir: str, output directory
        pano_h: int, height of the panorama (default: 1024)
        pano_w: int, width of the panorama (default: 2048)
    '''

    views = create_cube_map_views(center)
    # views = create_surrounding_views(center)

    render_path = output_dir
    makedirs(render_path, exist_ok=True)

    # Future TODO: render depth of panorama (could be used for adjusting HDR intensity)
    with torch.no_grad():

        cube_map_dict = {}
        for view_name, view in tqdm(views.items(), desc="Rendering progress"):
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, view_name + ".png"))
            cube_map_dict[view_name] = rendering.permute(1,2,0).cpu().numpy()

        # dilate each image
        # for view_name, img in cube_map_dict.items():
        #     cube_map_dict[view_name] = dilate_image(img)

        # convert cubemap to equirectangular
        equirectangular = c2e(cube_map_dict, pano_h, pano_w, mode='bilinear', cube_format='dict')
        equirectangular = Image.fromarray(np.clip(equirectangular * 255, 0, 255).astype(np.uint8))
        equirectangular.save(os.path.join(render_path, "pano_ldr.png"))

    return os.path.join(render_path, "pano_ldr.png")


if __name__ == "__main__":

    pass
    
    # args = get_opts()
    # print("Rendering " + args.model_path)
    # safe_state(args.quiet)

    # dataset = args.model_params
    # pipeline = args.pipeline_params

    # gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ##### Test rendering panorama with specific center #####
    # center = np.array([-1.1491928100585938,
    #             1.199149489402771,
    #             -2.420379161834717])
    # ldr_env_map_path = render_panorama(gaussians, pipeline, background, center, os.path.join(args.model_path, 'panorama', str(math.floor(time.time())) ) )
    # from inpaint.inpaint_anything import inpaint_img
    # from lighting.ldr2hdr import convert_ldr2hdr
    # ldr_env_map_path = inpaint_img(ldr_env_map_path)
    # hdr_env_map_path = convert_ldr2hdr(ldr_env_map_path)

    ##### Test rendering panorama with different vertical offset #####
    # from inpaint.inpaint_anything import inpaint_img
    # from lighting.ldr2hdr import convert_ldr2hdr
    # STEP = 0.2
    # for i in range(10):
    #     center[2] += STEP
    #     folder_name = '{0:03d}_'.format(i) + str(math.floor(time.time()))
    #     render_panorama(gaussians, pipeline, background, center, os.path.join(args.model_path, 'panorama', folder_name ) )
    #     inpaint_img(os.path.join(args.model_path, 'panorama', folder_name, 'pano_ldr.png'))
    #     hdr_env_map_path = convert_ldr2hdr(os.path.join(args.model_path, 'panorama', folder_name, 'pano_ldr_inpaint.png'))