import os
import json
import cv2
import numpy as np
import argparse
import torch
import importlib
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = 'tracking.Grounded-Segment-Anything.automatic_label_ram_demo'
ram_demo_module = importlib.import_module(module_path)
from sugar.gaussian_splatting.gaussian_renderer import get_ray_directions
import trimesh
from gpt.gpt4v_utils import estimate_object_scale


def run_scale_estimation(dataset_dir, scene_mesh_path, anchor_frame_idx=0):
    """
    Estimate the scale of the scene (compared to the real world scale) using the following steps:
    (1) Segment the anchor frame using RAM
    (2) Get the median depth value of each segmented object
    (3) Unproject the mask to 3D space with median depth to get estimated distance
    (4) Estimate the scale of the object using GPT-4V
    (5) Calculate the scene scale by dividing the estimated scale by the estimated distance
    """
    ram_seg_result_dir = os.path.join(dataset_dir, 'ram_segment_output')
    os.makedirs(ram_seg_result_dir, exist_ok=True)

    scene_scale_txt_path = os.path.join(ram_seg_result_dir, 'scene_scale_logs.txt')
    with open(scene_scale_txt_path, 'a') as f:
        f.write(f"===== Scene scale estimation for dataset {dataset_dir} =====\n\n")

    image_dir = os.path.join(dataset_dir, 'images')
    all_images_path = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    anchor_frame_path = all_images_path[anchor_frame_idx]

    # run RAM segmentation
    ram_demo_module.run_ram_segmentation(anchor_frame_path, ram_seg_result_dir)
    with open(os.path.join(ram_seg_result_dir, 'label.json'), 'r') as f:
        seg_info = json.load(f)
    print("Number of objects segmented in the scene: ", len(seg_info['mask']))

    # load camera infos and scene mesh
    camera_info_path = os.path.join(dataset_dir, 'transforms.json')
    with open(camera_info_path, 'r') as f:
        camera_info = json.load(f)
    w, h = camera_info['w'], camera_info['h']
    fx, fy, cx, cy = camera_info['fl_x'], camera_info['fl_y'], camera_info['cx'], camera_info['cy']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    directions = get_ray_directions(h, w, torch.FloatTensor(K), device="cuda", flatten=False)  # (H, W, 3)
    c2w_dict = {}
    for frame_info in camera_info['frames']:
        c2w_dict[frame_info['file_path'].split('/')[-1]] = np.array(frame_info['transform_matrix'])
    scene_mesh = trimesh.load_mesh(scene_mesh_path)

    # estimate scene scale
    scene_scale_list = []
    for obj in seg_info['mask']:
        if 'box' in obj:
            obj_index = obj['value']
            obj_label = obj['label']
            obj_logit = obj['logit']
            x_min, y_min, x_max, y_max = obj['box']   # x is width, y is height
            mask_img = cv2.imread(os.path.join(ram_seg_result_dir, str(obj_index), 'mask.jpg'), cv2.IMREAD_GRAYSCALE)
            mask = mask_img > 0
            # reject bounding box that is too small or has it side near the corner
            MINIMUM_BBOX_SIZE = 100 * 100
            bbox_size = (x_max - x_min) * (y_max - y_min)
            if bbox_size < MINIMUM_BBOX_SIZE:
                continue
            MINIMUM_CORNER_SIZE = 15
            if x_min < MINIMUM_CORNER_SIZE or y_min < MINIMUM_CORNER_SIZE or w - x_max < MINIMUM_CORNER_SIZE or h - y_max < MINIMUM_CORNER_SIZE:
                continue
            # get the median depth value of the mask
            c2w = c2w_dict[anchor_frame_path.split('/')[-1]]
            c2w = torch.FloatTensor(c2w).to("cuda")
            rays_d = directions @ c2w[:3, :3].T
            rays_o = c2w[:3, 3].expand_as(rays_d)
            rays_d = rays_d.cpu().numpy()
            rays_o = rays_o.cpu().numpy()
            ray_directions = rays_d[mask].reshape(-1, 3)
            ray_origins = rays_o[mask].reshape(-1, 3)
            locations, index_ray, index_tri = scene_mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=False
            )
            w2c = torch.inverse(c2w).cpu().numpy()
            xyz = locations
            xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
            xyz = np.matmul(xyz, w2c.T)
            depth = xyz[:, 2]
            median_depth = np.median(depth)
            ### Option 1: unproject the leftmost and rightmost points of the mask to 3D space ###
            # non_zero_coord = np.where(mask)  # 0: x, 1: y
            # sort_idx = np.argsort(non_zero_coord[1])  # from small y to large y
            # left_points_x, left_points_y = non_zero_coord[0][sort_idx[0]], non_zero_coord[1][sort_idx[0]]
            # right_points_x, right_points_y = non_zero_coord[0][sort_idx[-1]], non_zero_coord[1][sort_idx[-1]]
            # left_rays_o, left_rays_d = rays_o[left_points_x, left_points_y], rays_d[left_points_x, left_points_y]
            # right_rays_o, right_rays_d = rays_o[right_points_x, right_points_y], rays_d[right_points_x, right_points_y]
            # left_points = left_rays_o + left_rays_d * median_depth
            # right_points = right_rays_o + right_rays_d * median_depth
            # distance = np.linalg.norm(left_points - right_points)
            ### Option 2: unproject the whole mask to 3D space ###
            points3D = rays_o[mask] + rays_d[mask] * median_depth
            xyz_max, xyz_min = np.max(points3D, axis=0), np.min(points3D, axis=0)
            max_scale = np.max(xyz_max - xyz_min)
            distance = max_scale
            # get estimated scale from GPT-4V
            bbox_img_path = os.path.join(ram_seg_result_dir, str(obj_index), 'bbox.jpg')
            estimated_scale = estimate_object_scale(bbox_img_path, obj_label)
            with open(scene_scale_txt_path, 'a') as f:
                f.write(f"object name: {obj_label}, estimated scale: {estimated_scale}, estimated distance: {distance}, estimated median depth: {median_depth}, estimated scene scale: {estimated_scale / distance}\n")
            print(f"-- Object {obj_label} has estimated scale: {estimated_scale} meters")
            print(f"-- Object {obj_label} has estimated distance: {distance} meters")
            print(f"-- Object {obj_label} has estimated median depth: {median_depth} meters")
            print(f"-- Object {obj_label} has estimated scene scale: {estimated_scale / distance} meters")
            scene_scale_list.append(estimated_scale / distance)

    print("-- Estimated scene scale: ", np.median(scene_scale_list))

    # store the estimated scene scale in .txt file
    with open(scene_scale_txt_path, 'a') as f:
        f.write(f"*** average estimated scale: {np.mean(scene_scale_list)} ***\n")
        f.write(f"*** median estimated scale: {np.median(scene_scale_list)} ***\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets/tmp/', help='Path to dataset folder')
    parser.add_argument('--scene_mesh_path', type=str, default='datasets/tmp/scene_mesh.obj', help='Path to scene mesh file')
    parser.add_argument('--anchor_frame_idx', type=int, default=0, help='Index of the anchor frame to estimate scene scale')
    args = parser.parse_args()

    run_scale_estimation(args.dataset_dir, args.scene_mesh_path, args.anchor_frame_idx)