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
import sys
sys.path.append('./sugar/gaussian_splatting')

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

from kornia import create_meshgrid
from utils.graphics_utils import fov2focal

def depth_pcd2normal(xyz):
    '''
    Convert un-projected 3D point clouds into pseudo normal maps.
    Reference: https://github.com/Asparagus15/GaussianShader/
    '''
    hd, wd, _ = xyz.shape 
    bottom_point = xyz[..., 2:hd,   1:wd-1, :]
    top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
    right_point  = xyz[..., 1:hd-1, 2:wd,   :]
    left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

# @torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(H, W, K, device='cuda', random=False, return_uv=False, flatten=True, anti_aliasing_factor=1.0):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    if anti_aliasing_factor > 1.0:
        H = int(H * anti_aliasing_factor) 
        W = int(W * anti_aliasing_factor) 
        K *= anti_aliasing_factor
        K[2, 2] = 1
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else: # pass by the center
        directions = \
            torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    return directions


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, depth_image, alpha_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # concatenate RGB image with alpha image
    rendered_image = torch.cat((rendered_image, alpha_image), dim=0)

    depth_image = depth_image.squeeze(0)  # (1, H, W) -> (H, W)

    render_extras = {}

    # compute normal image (reference: GaussianShader)
    normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized)
    normal_normed = normal * 0.5 + 0.5          # from [-1, 1] to [0, 1]
    render_extras["normal"] = normal_normed

    out_extras = {}
    for k in render_extras.keys():
        if render_extras[k] is None: continue
        image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = render_extras[k],
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        out_extras[k] = image

    for k in ["normal"]:
        if k in out_extras.keys():
            out_extras[k] = (out_extras[k] - 0.5) * 2. # from [0, 1] to [-1, 1]
    
    # normalize the normal map
    normal_image = out_extras["normal"]
    normal_image = normal_image.permute(1, 2, 0) # (H, W, 3)
    normal_image = torch.nn.functional.normalize(normal_image, p=2, dim=-1)

    # compute pseudo normal image from depth with local differences
    h, w = viewpoint_camera.image_height, viewpoint_camera.image_width
    fx = fov2focal(viewpoint_camera.FoVx, w)
    fy = fov2focal(viewpoint_camera.FoVy, h)
    cx = w / 2
    cy = h / 2
    c2w = viewpoint_camera.world_view_transform.inverse()
    directions = get_ray_directions(h, w, torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), flatten=False)  # (H, W, 3)
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand_as(rays_d)
    depth = depth_image.unsqueeze(-1)   # (H, W, 1)
    points3D = rays_o + rays_d * depth
    pseudo_normal = depth_pcd2normal(points3D)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth_image,
            "normal": normal_image,
            "pseudo_normal": pseudo_normal,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
