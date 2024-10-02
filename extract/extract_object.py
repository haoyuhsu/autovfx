import os
import cv2
import numpy as np
import trimesh
import open3d as o3d
import torch
from tqdm import tqdm
import copy
from PIL import Image
import torchvision
import json
import glob
from sugar.gaussian_splatting.gaussian_renderer import get_ray_directions
from sugar.gaussian_splatting.gaussian_renderer import render
from scipy.spatial import Delaunay
from inpaint.lama_inpaint import inpaint_img_with_lama
from inpaint.utils import save_array_to_img, load_img_to_array, dilate_mask, erode_mask

from sugar.gaussian_splatting.scene.gaussian_model import GaussianModel
from sugar.gaussian_splatting.utils.graphics_utils import focal2fov, fov2focal
from sugar.sugar_scene.cameras import load_gs_cameras, GSCamera


def normalize(vec):
    eps = np.finfo(float).eps
    normalized_vec = vec / (np.linalg.norm(vec)+eps)
    return normalized_vec


def rotm_from_lookat(lookat, up):
    z_axis = normalize(lookat)
    # x_axis = normalize(np.cross(up, z_axis))
    x_axis = normalize(np.cross(z_axis, up))
    y_axis = normalize(np.cross(z_axis, x_axis))
    R = np.array((x_axis, y_axis, z_axis)).T  # cv2world
    return R


def grid_half_sphere(radius=1.5, num_views=30, theta=None, phi_range=(0, 360)):
    if theta is None:
        theta = np.deg2rad(np.array((0, 15, 30, 45, 60)))
    else:
        theta = np.deg2rad(theta)
    phi = np.deg2rad(np.linspace(phi_range[0], phi_range[1], num_views // len(theta)+1)[:-1])
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()
    x = np.cos(theta) * np.cos(phi) * radius
    y = np.cos(theta) * np.sin(phi) * radius
    z = np.sin(theta) * radius
    t = np.stack((x, y, z), axis=-1)
    return t


def extract_object_from_scene(scene_representation, object_name, object_id):
    '''
    (1) Extract object instance (both mesh and 3D gaussians) from the tracking results.
    (2) Render object instance with custom trajectory and save both meshes and 3D gaussians.
    (3) Render scene without object instance and save both meshes and 3D gaussians.
    '''
    print("Extracting object: {} with id {}".format(object_name, object_id))

    save_dir = os.path.join(scene_representation.results_dir, "object_instance", scene_representation.custom_traj_name, '_'.join(object_name.split(' ')), str(object_id))
    os.makedirs(save_dir, exist_ok=True)

    # store intersected points from anchor view
    points_anchor_view = extract_object_from_single_view(scene_representation, object_name, object_id, use_ray_mesh_intersection=True)
    if points_anchor_view is not None:
        np.save(os.path.join(save_dir, "points_anchor_view.npy"), points_anchor_view)

    if os.path.exists(os.path.join(save_dir, "object_mesh", "object_mesh.obj")):
        print("Object instance already extracted.")
        return os.path.join(save_dir, "object_mesh", "object_mesh.obj")

    object_tracking_results_dir = os.path.join(scene_representation.tracking_results_dir, '_'.join(object_name.split(' ')))    
    object_instance_dir = os.path.join(object_tracking_results_dir, str(object_id))

    if not os.path.exists(object_tracking_results_dir):
        raise FileNotFoundError("Tracking results for object {} not found.".format(object_name))
    if not os.path.exists(object_instance_dir):
        raise FileNotFoundError("Object instance {} not found in the tracking results.".format(object_id))

    c2w_dict = scene_representation.cameras['c2w_dict']
    w, h = scene_representation.cameras['img_wh']
    K = scene_representation.cameras['K']

    # get the tracking binary masks for the object
    obj_masks = {}
    for file in sorted(os.listdir(object_instance_dir)):
        if file.endswith(".png"):
            mask = cv2.imread(os.path.join(object_instance_dir, file), cv2.IMREAD_GRAYSCALE)
            # dilate_kernel_size = 25
            # mask = dilate_mask(mask, dilate_kernel_size)  # TODO Test: dilate the mask to cover the object instance
            # mask = erode_mask(mask, erode_kernel_size)
            obj_masks[file] = mask

    # load scene meshes
    scene_mesh_path = scene_representation.hparams.scene_mesh_path
    scene_mesh = trimesh.load_mesh(scene_mesh_path)
    # print("Number of vertices:", len(scene_mesh.vertices))
    # print("Number of faces:", len(scene_mesh.faces))

    gaussians = scene_representation.gaussians

    # compute closest triangle of each gaussians
    gaussians_xyz = gaussians._xyz.detach().cpu().numpy()
    scene_o3d = o3d.t.geometry.RaycastingScene()
    scene_o3d.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene_mesh.as_open3d))
    ret_dict = scene_o3d.compute_closest_points(o3d.core.Tensor.from_numpy(gaussians_xyz.astype(np.float32)))
    triangle_ids = ret_dict['primitive_ids'].cpu().numpy()

    # voting method to get object instance mesh
    TRIANGLES_VIEW_COUNTER = torch.zeros(len(scene_mesh.faces), dtype=torch.int32, device="cuda")
    TRIANGLES_VIEW_COUNTER_NAIVE = torch.zeros(len(scene_mesh.faces), dtype=torch.int32, device="cuda")
    for filename, mask in tqdm(obj_masks.items(), desc="Unprojecting..."):

        with torch.no_grad():
        
            idx = int(filename.split('/')[-1].split('.')[0])
            mask = torch.tensor(mask, dtype=torch.bool, device="cuda")

            c2w = c2w_dict[filename]
            c2w = torch.FloatTensor(c2w).to("cuda")
            directions = get_ray_directions(h, w, torch.FloatTensor(K), device="cuda", flatten=False)  # (H, W, 3)
            rays_d = directions @ c2w[:3, :3].T
            rays_o = c2w[:3, 3].expand_as(rays_d)

            rays_d = rays_d[mask].reshape(-1, 3).cpu().numpy()
            rays_o = rays_o[mask].reshape(-1, 3).cpu().numpy()

            index_tri = scene_mesh.ray.intersects_first(
                ray_origins=rays_o,
                ray_directions=rays_d
            )
            index_tri = torch.tensor(index_tri, dtype=torch.int32, device="cuda")

            TRIANGLES_VIEW_COUNTER_NAIVE[index_tri] += 1

            # check again if the triangle center is in the mask
            w2c = torch.inverse(c2w)
            xyz = scene_mesh.triangles_center[index_tri.cpu().numpy()]
            xyz = torch.Tensor(xyz).to("cuda")
            xyz = torch.cat([xyz, torch.ones(xyz.size(0), 1, device="cuda")], dim=1)
            xyz = torch.matmul(xyz, w2c.T)
            xyz = xyz[:, :3]
            xyz = xyz / xyz[:, 2].unsqueeze(1)

            uv = torch.matmul(xyz, torch.FloatTensor(K).to("cuda").T)
            uv = uv[:, :2].round().long()

            # filter out out-of-bound UV coordinates
            h, w = mask.shape
            valid_uv_mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            filtered_uv = uv[valid_uv_mask]
            in_mask_indices = torch.where(mask[filtered_uv[:, 1], filtered_uv[:, 0]])[0]
            in_mask = valid_uv_mask.nonzero().flatten()[in_mask_indices]
            index_tri = index_tri[in_mask]

            TRIANGLES_VIEW_COUNTER[index_tri] += 1

    N_VIEWS = len(obj_masks.keys())

    torch.cuda.empty_cache()

    # TODO: if the object instance has holes or incomplete extraction, try turn on this
    # TRIANGLES_VIEW_COUNTER = TRIANGLES_VIEW_COUNTER_NAIVE   # since no need to check if the triangle center is in the mask

    total_missed_pixels_list = []
    RATIO_LIST = [0.05, 0.1, 0.2, 0.25, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]    # TODO: might use binary search to find the optimal ratio
    for RATIO in RATIO_LIST:
        print("===== RATIO: ", RATIO, "=====")
        MINIMUM_VIEWS = int(N_VIEWS * RATIO)  # minimum number of views to consider a gaussian as close to the object

        mask_triangles = TRIANGLES_VIEW_COUNTER >= MINIMUM_VIEWS
        print("Number of triangles that are close to the object: ", mask_triangles.sum())

        if mask_triangles.sum() < 50:  # skip if the number of triangles is too small
            continue

        mask_triangles = mask_triangles.cpu().numpy()
        masked_mesh = scene_mesh.submesh([mask_triangles], append=True)
        convex_hull = masked_mesh.convex_hull
        original_tri_centroids = scene_mesh.triangles_center
        inside_hull = convex_hull.contains(original_tri_centroids)
        mask_triangles = np.logical_or(mask_triangles, inside_hull)
        mask_triangles_idx = np.where(mask_triangles)[0]

        # keep the gaussians with closest triangle index in mask_triangles_idx
        ### mask_trianlges_idx: the index of triangles that are close to the object
        ### triangle_ids: the index of the closest triangle of each gaussian
        mask3d = np.isin(triangle_ids, mask_triangles_idx)
        mask3d = torch.tensor(mask3d, dtype=torch.bool, device="cuda")
        print("Number of gaussians that are close to the object: ", mask3d.sum())

        # crop gaussians
        cropped_object_gaussians = copy.deepcopy(gaussians)
        cropped_object_gaussians._xyz = gaussians._xyz[mask3d]
        cropped_object_gaussians._features_dc = gaussians._features_dc[mask3d]
        cropped_object_gaussians._features_rest = gaussians._features_rest[mask3d]
        cropped_object_gaussians._scaling = gaussians._scaling[mask3d]
        cropped_object_gaussians._rotation = gaussians._rotation[mask3d]
        cropped_object_gaussians._opacity = gaussians._opacity[mask3d]

        total_missed_pixels = 0

        # render the cropped gaussians to compute the object mask
        camera_views = scene_representation.cameras['cameras']
        for idx, view in tqdm(enumerate(camera_views), desc="Rendering progress"):
            # check if "{0:05d}.png".format(idx) exists in obj_masks.keys()
            if "{0:05d}.png".format(idx) not in obj_masks.keys():
                continue
            object_mask_tracking = obj_masks["{0:05d}.png".format(idx)]
            with torch.no_grad():
                results = render(view, cropped_object_gaussians, scene_representation.pipe, scene_representation.background)
            object_alpha_image = results["render"].permute(1, 2, 0)[:, :, 3].cpu().numpy()
            object_mask_rendered = object_alpha_image >= 0.8
            object_mask_tracking = np.array(object_mask_tracking) == 255
            # compute the number of pixels mismatched by XOR operation
            xor_mask = np.logical_xor(object_mask_rendered, object_mask_tracking)
            xor_pixels = np.count_nonzero(xor_mask)
            total_missed_pixels += xor_pixels
        
        print("Total missed pixels: ", total_missed_pixels)
        total_missed_pixels_list.append(total_missed_pixels)

    for ratio, missed_pixels in zip(RATIO_LIST, total_missed_pixels_list):
        print("RATIO: ", ratio, "MISSED PIXELS: ", missed_pixels)

    # select the ratio with minimum missed pixels
    best_ratio = RATIO_LIST[np.argmin(total_missed_pixels_list)]
    print("===== BEST RATIO: ", best_ratio, "=====")

    MINIMUM_VIEWS = int(N_VIEWS * best_ratio)  # minimum number of views to consider a gaussian as close to the object
    mask_triangles = TRIANGLES_VIEW_COUNTER >= MINIMUM_VIEWS
    mask_triangles = mask_triangles.cpu().numpy()
    masked_mesh = scene_mesh.submesh([mask_triangles], append=True)

    # # TODO: if the object instance has holes or incomplete extraction, try turn on this
    # # Dilate the vertices of the mesh to cover the holes. The dilate direction is unit direction between the vertex and the bottom center of the mesh.
    # # The dilate distance is 0.1 times the maximum range of the mesh.
    # vertices = masked_mesh.vertices
    # bottom_center = masked_mesh.bounds[0] + (masked_mesh.bounds[1] - masked_mesh.bounds[0]) / 2
    # # bottom_center[2] = masked_mesh.bounds[0][2]   # set z to the minimum z value
    # dilate_directions = vertices - bottom_center
    # dilate_directions = dilate_directions / np.linalg.norm(dilate_directions, axis=1)[:, None]
    # scale = 0.1 * np.max(masked_mesh.extents)  # 0.1 the best
    # dilate_vertices = vertices + scale * dilate_directions
    # masked_mesh.vertices = dilate_vertices

    convex_hull = masked_mesh.convex_hull
    original_tri_centroids = scene_mesh.triangles_center
    inside_hull = convex_hull.contains(original_tri_centroids)
    mask_triangles = np.logical_or(mask_triangles, inside_hull)
    mask_triangles_idx = np.where(mask_triangles)[0]
    mask3d = np.isin(triangle_ids, mask_triangles_idx)
    mask3d = torch.tensor(mask3d, dtype=torch.bool, device="cuda")

    print("====== Number of triangles from TRIANGLES_VIEW_COUNTER: ", mask3d.sum().item(), "======")

    if mask3d.nonzero().size(0) == 0:
        return

    # save object instance mesh
    object_mesh_save_dir = os.path.join(save_dir, "object_mesh")
    os.makedirs(object_mesh_save_dir, exist_ok=True)
    object_instance_mesh = scene_mesh.submesh([mask_triangles], append=True)
    object_instance_mesh.export(os.path.join(object_mesh_save_dir, "object_mesh.obj"))

    # save the remaining scene mesh
    scene_mesh_save_dir = os.path.join(save_dir, "removal_mesh")
    os.makedirs(scene_mesh_save_dir, exist_ok=True)
    non_mask_triangles = ~mask_triangles
    scene_mesh_remaining = scene_mesh.submesh([non_mask_triangles], append=True)
    scene_mesh_remaining.export(os.path.join(scene_mesh_save_dir, "removal_mesh.obj"))

    # save object instance gaussians for further usage
    object_gaussians = copy.deepcopy(gaussians)
    object_gaussians._xyz = gaussians._xyz[mask3d]
    object_gaussians._features_dc = gaussians._features_dc[mask3d]
    object_gaussians._features_rest = gaussians._features_rest[mask3d]
    object_gaussians._scaling = gaussians._scaling[mask3d]
    object_gaussians._rotation = gaussians._rotation[mask3d]
    object_gaussians._opacity = gaussians._opacity[mask3d]
    object_gaussians.save_ply(os.path.join(save_dir, "object_gaussians.ply"))

    # get background gaussians using different way (use TRIANGLES_VIEW_COUNTER_NAIVE to avoid floaters) --> not necessary
    total_missed_pixels_list = []
    RATIO_LIST = np.arange(best_ratio, best_ratio + 0.2, 0.02)
    for RATIO in RATIO_LIST:
        print("===== RATIO: ", RATIO, "=====")
        MINIMUM_VIEWS = int(N_VIEWS * RATIO)  # minimum number of views to consider a gaussian as close to the object

        mask_triangles = TRIANGLES_VIEW_COUNTER_NAIVE >= MINIMUM_VIEWS
        print("Number of triangles that are close to the object: ", mask_triangles.sum())

        if mask_triangles.sum() < 50:  # skip if the number of triangles is too small
            continue

        mask_triangles = mask_triangles.cpu().numpy()
        masked_mesh = scene_mesh.submesh([mask_triangles], append=True)
        convex_hull = masked_mesh.convex_hull
        original_tri_centroids = scene_mesh.triangles_center
        inside_hull = convex_hull.contains(original_tri_centroids)
        mask_triangles = np.logical_or(mask_triangles, inside_hull)
        mask_triangles_idx = np.where(mask_triangles)[0]

        # keep the gaussians with closest triangle index in mask_triangles_idx
        ### mask_trianlges_idx: the index of triangles that are close to the object
        ### triangle_ids: the index of the closest triangle of each gaussian
        mask3d = np.isin(triangle_ids, mask_triangles_idx)
        mask3d = torch.tensor(mask3d, dtype=torch.bool, device="cuda")
        print("Number of gaussians that are close to the object: ", mask3d.sum())

        # crop gaussians
        cropped_object_gaussians = copy.deepcopy(gaussians)
        cropped_object_gaussians._xyz = gaussians._xyz[mask3d]
        cropped_object_gaussians._features_dc = gaussians._features_dc[mask3d]
        cropped_object_gaussians._features_rest = gaussians._features_rest[mask3d]
        cropped_object_gaussians._scaling = gaussians._scaling[mask3d]
        cropped_object_gaussians._rotation = gaussians._rotation[mask3d]
        cropped_object_gaussians._opacity = gaussians._opacity[mask3d]

        total_missed_pixels = 0

        # render the cropped gaussians to compute the object mask
        camera_views = scene_representation.cameras['cameras']
        for idx, view in tqdm(enumerate(camera_views), desc="Rendering progress"):
            # check if "{0:05d}.png".format(idx) exists in obj_masks.keys()
            if "{0:05d}.png".format(idx) not in obj_masks.keys():
                continue
            object_mask_tracking = obj_masks["{0:05d}.png".format(idx)]
            with torch.no_grad():
                results = render(view, cropped_object_gaussians, scene_representation.pipe, scene_representation.background)
            object_alpha_image = results["render"].permute(1, 2, 0)[:, :, 3].cpu().numpy()
            object_mask_rendered = object_alpha_image >= 0.8
            object_mask_tracking = np.array(object_mask_tracking) == 255
            # compute the number of pixels mismatched by XOR operation
            xor_mask = np.logical_xor(object_mask_rendered, object_mask_tracking)
            xor_pixels = np.count_nonzero(xor_mask)
            total_missed_pixels += xor_pixels
        
        print("Total missed pixels: ", total_missed_pixels)
        total_missed_pixels_list.append(total_missed_pixels)

    for ratio, missed_pixels in zip(RATIO_LIST, total_missed_pixels_list):
        print("RATIO: ", ratio, "MISSED PIXELS: ", missed_pixels)

    # select the ratio with minimum missed pixels
    best_ratio = RATIO_LIST[np.argmin(total_missed_pixels_list)]
    print("===== BEST RATIO: ", best_ratio, "=====")

    MINIMUM_VIEWS = int(N_VIEWS * best_ratio)  # minimum number of views to consider a gaussian as close to the object
    mask_triangles = TRIANGLES_VIEW_COUNTER_NAIVE >= MINIMUM_VIEWS
    mask_triangles = mask_triangles.cpu().numpy()
    masked_mesh = scene_mesh.submesh([mask_triangles], append=True)
    convex_hull = masked_mesh.convex_hull
    original_tri_centroids = scene_mesh.triangles_center
    inside_hull = convex_hull.contains(original_tri_centroids)
    mask_triangles = np.logical_or(mask_triangles, inside_hull)
    mask_triangles_idx = np.where(mask_triangles)[0]
    mask3d = np.isin(triangle_ids, mask_triangles_idx)
    mask3d = torch.tensor(mask3d, dtype=torch.bool, device="cuda")

    print("====== Number of triangles from TRIANGLES_VIEW_COUNTER_NAIVE: ", mask3d.sum().item(), "======")

    # save the remaining gaussians for further usage
    non_mask3d = ~mask3d
    object_removal_gaussians = copy.deepcopy(gaussians)
    object_removal_gaussians._xyz = gaussians._xyz[non_mask3d]
    object_removal_gaussians._features_dc = gaussians._features_dc[non_mask3d]
    object_removal_gaussians._features_rest = gaussians._features_rest[non_mask3d]
    object_removal_gaussians._scaling = gaussians._scaling[non_mask3d]
    object_removal_gaussians._rotation = gaussians._rotation[non_mask3d]
    object_removal_gaussians._opacity = gaussians._opacity[non_mask3d]
    object_removal_gaussians.save_ply(os.path.join(save_dir, "removal_gaussians.ply"))

    # Optional: render the object instance with custom trajectory
    render_path = os.path.join(save_dir, "render_removal_mesh")
    os.makedirs(render_path, exist_ok=True)
    obj_render_path = os.path.join(save_dir, "render_object_mesh")
    os.makedirs(obj_render_path, exist_ok=True)
    camera_views = scene_representation.cameras['cameras']
    for idx, view in tqdm(enumerate(camera_views), desc="Rendering progress"):
        with torch.no_grad():
            pipeline = scene_representation.pipe
            background = scene_representation.background
            results = render(view, object_removal_gaussians, pipeline, background)
            torchvision.utils.save_image(results["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            results = render(view, object_gaussians, pipeline, background)
            torchvision.utils.save_image(results["render"], os.path.join(obj_render_path, '{0:05d}'.format(idx) + ".png"))

    return os.path.join(object_mesh_save_dir, "object_mesh.obj")


def extract_object_from_single_view(scene_representation, object_name, object_id, use_ray_mesh_intersection=True):
    '''
    Extract 3D points of the object instance from a single view.
    '''
    anchor_frame_idx = scene_representation.anchor_frame_idx
    print("Extracting object: {} with id {} from view {}".format(object_name, object_id, anchor_frame_idx))

    object_tracking_results_dir = os.path.join(scene_representation.tracking_results_dir, '_'.join(object_name.split(' ')))    
    object_instance_dir = os.path.join(object_tracking_results_dir, str(object_id))

    if not os.path.exists(os.path.join(object_instance_dir, '{0:05d}.png'.format(anchor_frame_idx))):
        print("Object instance {} not found in the tracking results.".format(object_id))
        return None

    c2w = scene_representation.cameras['c2w_dict']["{0:05d}.png".format(anchor_frame_idx)]
    w, h = scene_representation.cameras['img_wh']
    K = scene_representation.cameras['K']

    # get segmented mask of the object and depth map from the anchored view
    mask_path = os.path.join(object_instance_dir, '{0:05d}.png'.format(anchor_frame_idx))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = torch.tensor(mask, dtype=torch.bool, device="cuda")

    # set up rays
    c2w = torch.FloatTensor(c2w).to("cuda")
    directions = get_ray_directions(h, w, torch.FloatTensor(K), device="cuda", flatten=False)  # (H, W, 3)
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand_as(rays_d)

    if use_ray_mesh_intersection:
        scene_mesh_path = scene_representation.hparams.scene_mesh_path
        scene_mesh = trimesh.load_mesh(scene_mesh_path)
        ray_origins = rays_o[mask].cpu().numpy()
        ray_directions = rays_d[mask].cpu().numpy()
        locations, index_ray, index_tri = scene_mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )
        obj_points3D = locations
    else:
        depth_path = os.path.join(scene_representation.traj_results_dir, 'depth', '{0:05d}.npy'.format(anchor_frame_idx))
        if not os.path.exists(mask_path) or not os.path.exists(depth_path):
            raise FileNotFoundError("Mask or depth map not found for object {} in view {}.".format(object_name, anchor_frame_idx))
        depth = np.load(depth_path)
        depth_raw = torch.FloatTensor(depth).unsqueeze(-1).to("cuda")
        points3D = rays_o + rays_d * depth_raw
        obj_points3D = points3D[mask]
        obj_points3D = obj_points3D.cpu().numpy()

    return obj_points3D


def inpaint_object(scene_representation, object_name, object_id, dilate_kernel_size=25):
    '''
    (1) add a planar mesh to cover the gap left by the object instance
    (2) render the mask of planar mesh as inpainted object mask
    (3) re-train 3DGS on those inpainted regions
    '''
    print("Inpainting object: {} with id {}".format(object_name, object_id))

    save_dir = os.path.join(scene_representation.results_dir, "object_instance", scene_representation.custom_traj_name, '_'.join(object_name.split(' ')), str(object_id))

    object_removal_gaussians_path = os.path.join(save_dir, "removal_gaussians.ply")
    object_gaussians_path = os.path.join(save_dir, "object_gaussians.ply")
    object_removal_mesh_path = os.path.join(save_dir, "removal_mesh", "removal_mesh.obj")
    object_mesh_path = os.path.join(save_dir, "object_mesh", "object_mesh.obj")

    object_mesh = trimesh.load_mesh(object_mesh_path)
    object_removal_mesh = trimesh.load_mesh(object_removal_mesh_path)

    # get xyz coordinates with and z-min from object mesh
    obj_xyz = object_mesh.vertices
    z_min = object_mesh.vertices[:, 2].min()

    # create a planar mesh to cover the gap (use convex hull)
    coords = obj_xyz[:, :2]
    vertices = np.hstack([coords, np.full((len(coords), 1), z_min)])
    planar_mesh = trimesh.convex.convex_hull(vertices)

    # cast a ray from each point towards -z direction to get the first intersection point to the object removal mesh
    ray_origins = object_removal_mesh.triangles_center
    ray_directions = np.array([[0, 0, -1]] * len(ray_origins))
    locations, index_ray, index_tri = planar_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )

    # remove those points in the object_removal_mesh that are higher than z_min
    triangles_to_remove = np.unique(index_ray)
    mask = np.ones(len(object_removal_mesh.faces), dtype=bool)
    mask[triangles_to_remove] = False
    object_removal_mesh.update_faces(mask)

    # save the planar mesh
    planar_mesh_path = os.path.join(save_dir, "planar_mesh")
    os.makedirs(planar_mesh_path, exist_ok=True)
    planar_mesh.export(os.path.join(planar_mesh_path, "planar_mesh.obj"))

    # save the inpainted object mask
    inpaint_object_removal_mesh_path = os.path.join(save_dir, "inpaint_removal_mesh")
    os.makedirs(inpaint_object_removal_mesh_path, exist_ok=True)
    inpaint_object_removal_mesh = object_removal_mesh + planar_mesh
    inpaint_object_removal_mesh.export(os.path.join(inpaint_object_removal_mesh_path, "inpaint_removal_mesh.obj"))

    # render the inpainted object mask
    c2w_dict = scene_representation.cameras['c2w_dict']
    w, h = scene_representation.cameras['img_wh']
    K = scene_representation.cameras['K']

    # sample custom camera views around the region
    poses = []
    planar_center = (planar_mesh.bounds[0] + planar_mesh.bounds[1]) / 2    # TODO: this might be bugged
    max_object_scale = max(object_mesh.bounds[1] - object_mesh.bounds[0])
    cam_pos_list = grid_half_sphere(radius=max_object_scale * 1.5, num_views=150, theta=[45], phi_range=(0, 360))
    for t in cam_pos_list:
        cam_pos = planar_center + t
        lookat = planar_center - cam_pos
        R = rotm_from_lookat(lookat, np.array([0, 0, 1]))
        c2w = np.hstack((R, cam_pos.reshape(3, 1)))
        c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))
        # poses.append(c2w)  # TODO: uncomment this line to use upper-hemisphere views
    poses += [c2w_dict[key] for key in sorted(c2w_dict.keys())]   # use both custom sample path and original camera views
    transforms_dict = {
        "trajectory_name": "inpaint",
        "camera_model": "OPENCV",
        "fl_x": K[0, 0],
        "fl_y": K[1, 1],
        "cx": K[0, 2],
        "cy": K[1, 2],
        "w": w,
        "h": h,
    }
    frames = [{"filename": "{:05d}.png".format(i), "transform_matrix": c2w.tolist()} for i, c2w in enumerate(poses)]
    transforms_dict["frames"] = frames
    with open(os.path.join(save_dir, "inpaint_camera_poses.json"), "w") as f:
        json.dump(transforms_dict, f, indent=4)

    # create camera views
    view_list = []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for cam_idx, c2w in enumerate(poses):
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
        view_list.append(view)

    # render the image with custom camera views
    object_removal_gaussians = GaussianModel(scene_representation.hparams.max_sh_degree - 1)  # SuGaR: 4, vanilla 3DGS: 3
    object_removal_gaussians.load_ply(object_removal_gaussians_path)
    render_inpaint_img_path = os.path.join(save_dir, "render_inpaint_img")
    os.makedirs(render_inpaint_img_path, exist_ok=True)
    with torch.no_grad():
        for idx, view in tqdm(enumerate(view_list), desc="Rendering inpaint image"):
            results = render(view, object_removal_gaussians, scene_representation.pipe, scene_representation.background)
            torchvision.utils.save_image(results["render"], os.path.join(render_inpaint_img_path, '{0:05d}'.format(idx) + ".png"))

    # render the inpainted object mask with custom camera views
    render_inpaint_mask_path = os.path.join(save_dir, "render_inpaint_mask")
    os.makedirs(render_inpaint_mask_path, exist_ok=True)
    for idx, c2w in tqdm(enumerate(poses), desc="Rendering inpaint mask"):
        c2w = torch.FloatTensor(c2w).to("cuda")
        directions = get_ray_directions(h, w, torch.FloatTensor(K), device="cuda", flatten=False)  # (H, W, 3)
        rays_d = directions @ c2w[:3, :3].T
        rays_o = c2w[:3, 3].expand_as(rays_d)
        rays_d = rays_d.reshape(-1, 3).cpu().numpy()
        rays_o = rays_o.reshape(-1, 3).cpu().numpy()
        # find which ray intersects with the inpainted object mask
        locations_planar, index_ray_planar, index_tri_planar = planar_mesh.ray.intersects_location(
            ray_origins=rays_o,
            ray_directions=rays_d,
            multiple_hits=False
        )
        if len(index_ray_planar) == 0:
            continue
        # find which ray is not blocked by the object removal mesh
        locations_all, index_ray_all, index_tri_all = inpaint_object_removal_mesh.ray.intersects_location(
            ray_origins=rays_o[index_ray_planar],
            ray_directions=rays_d[index_ray_planar],
            multiple_hits=False
        )
        if len(index_ray_all) == 0:
            continue
        # projection to get depth
        w2c = torch.inverse(c2w).cpu().numpy()
        xyz = locations_planar
        xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        xyz = np.matmul(xyz, w2c.T)
        depth_planar = xyz[:, 2]
        # projection to get depth
        depth_all = np.zeros(depth_planar.shape)
        xyz = locations_all
        xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        xyz = np.matmul(xyz, w2c.T)
        depth_all[index_ray_all] = xyz[:, 2]
        mask = np.abs(depth_all - depth_planar) < 0.01
        index_ray_planar = index_ray_planar[mask]
        # generate binary mask image
        mask_image = np.zeros((h, w), dtype=np.uint8)
        mask_image = mask_image.reshape(-1)
        mask_image[index_ray_planar] = 255
        mask_image = mask_image.reshape(h, w)
        mask_image = Image.fromarray(mask_image)
        mask_image.save(os.path.join(render_inpaint_mask_path, '{0:05d}'.format(idx) + ".png"))

    # Inpaint the object mask using LAMA
    lama_config = "./inpaint/lama/configs/prediction/default.yaml"
    lama_ckpt = "./inpaint/ckpts/big-lama/"
    render_inpaint_img_path = os.path.join(save_dir, "render_inpaint_img")
    render_inpaint_mask_path = os.path.join(save_dir, "render_inpaint_mask")
    render_inpaint_lama_path = os.path.join(save_dir, "render_inpaint_lama")
    os.makedirs(render_inpaint_lama_path, exist_ok=True)
    for idx in tqdm(range(len(poses)), desc="Rendering inpaint with LaMa"):
        img_path = os.path.join(render_inpaint_img_path, '{0:05d}.png'.format(idx))
        img = load_img_to_array(img_path)[..., :3]
        mask_path = os.path.join(render_inpaint_mask_path, '{0:05d}.png'.format(idx))
        if not os.path.exists(mask_path):
            save_array_to_img(img, os.path.join(render_inpaint_lama_path, '{0:05d}.png'.format(idx)))
        else:
            mask = load_img_to_array(mask_path) == 255
            if dilate_kernel_size is not None:
                mask = dilate_mask(mask, dilate_kernel_size)
            img_inpainted = inpaint_img_with_lama(img, mask, lama_config, lama_ckpt)
            save_array_to_img(img_inpainted, os.path.join(render_inpaint_lama_path, '{0:05d}.png'.format(idx)))


def get_largest_object(scene_representation, object_name, obj_ids):
    '''
    Get the largest object instance from the list of object ids.
    '''
    obj_sizes = []
    for obj_id in obj_ids:
        object_instance_dir = os.path.join(scene_representation.tracking_results_dir, '_'.join(object_name.split(' ')), str(obj_id))
        mask_paths = sorted(glob.glob(os.path.join(object_instance_dir, "*.png")))
        pixel_count = 0
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.array(mask) == 255
            pixel_count += mask.sum()
        obj_sizes.append(pixel_count)

    largest_obj_id = obj_ids[np.argmax(obj_sizes)]
    print("Largest object instance id: ", largest_obj_id)

    return largest_obj_id