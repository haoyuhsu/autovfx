import numpy as np
import os
import json
import argparse
import vedo
from colmap_read_model import read_points3D_binary


def visualize_camera_and_points3D(poses_array, points3D, point_pos=None, arrow_len=1, s=1, points_r=1):
    """
    params: poses_array: (N, 4, 4) or (N, 3, 4)
    params: points3D: (N_pts, 3)
    """
    plt = vedo.Plotter()
    pos = poses_array[:, 0:3, 3]
    x_end = pos + arrow_len * poses_array[:, 0:3, 0]
    y_end = pos + arrow_len * poses_array[:, 0:3, 1]
    z_end = pos + arrow_len * poses_array[:, 0:3, 2]
    x = vedo.Arrows(pos, x_end, c="r", s=s)
    y = vedo.Arrows(pos, y_end, c="g", s=s)
    z = vedo.Arrows(pos, z_end, c="b", s=s)

    points3D = vedo.Points(points3D, r=points_r, c=(0.3, 0.3, 0.3), alpha=0.5)

    if point_pos is not None:
        # create a sphere as a reference point
        points = vedo.Sphere(np.array(point_pos), r=0.1, c="r")
        # points = vedo.Sphere(np.array([-1.1491928100585938,
        #         1.199149489402771,
        #         -2.420379161834717]), r=0.1, c="r")
        plt.show(x, y, z, points3D, points, axes=1, viewup="z")
    else:
        plt.show(x, y, z, points3D, axes=1, viewup="z")

    return x, y, z


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
        theta = np.deg2rad([theta])
    phi = np.deg2rad(np.linspace(phi_range[0], phi_range[1], num_views // len(theta)+1)[:-1])
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()
    x = np.cos(theta) * np.cos(phi) * radius
    y = np.cos(theta) * np.sin(phi) * radius
    z = np.sin(theta) * radius
    t = np.stack((x, y, z), axis=-1)
    return t


def sample_camera_poses(dataset_dir, traj_output_path, sampled_config, vis_traj=False):
    
    radius = sampled_config['camera_radius']
    num_views = sampled_config['n_views']
    theta_angle = sampled_config['theta_angle']
    phi_range = sampled_config['phi_range']
    center_pos = np.array(sampled_config['camera_center'])

    cam_pos_list = grid_half_sphere(radius, num_views, theta_angle, phi_range) + center_pos

    poses = []
    poses = []
    for t in cam_pos_list:
        lookat = center_pos - t
        R = rotm_from_lookat(lookat, np.array([0, 0, 1]))
        c2w = np.hstack((R, t.reshape(3, 1)))
        c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))
        poses.append(c2w)
    poses = np.stack(poses, axis=0)

    if vis_traj:
        xyz, rgb, _ = read_points3D_binary(os.path.join(dataset_dir, 'sparse/0', 'points3D.bin'))
        visualize_camera_and_points3D(poses, xyz, point_pos=center_pos)

    # Save the trajectory
    save_name = os.path.basename(traj_output_path).split('.')[0]
    transforms_dict = {
        "trajectory_name": save_name,
        "camera_model": "OPENCV",
        "fl_x": sampled_config['fx'],
        "fl_y": sampled_config['fy'],
        "cx": sampled_config['cx'],
        "cy": sampled_config['cy'],
        "w": sampled_config['width'],
        "h": sampled_config['height'],
    }
    frames = [{"filename": "{:05d}.png".format(i), "transform_matrix": c2w.tolist()} for i, c2w in enumerate(poses)]
    transforms_dict["frames"] = frames
    with open(traj_output_path, 'w') as f:
        json.dump(transforms_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a sample trajectory')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/tmp_dataset', help='Path to the dataset directory')
    parser.add_argument('--traj_name', type=str, default='transforms_001', help='Name of the trajectory file')
    parser.add_argument('--vis_traj', action='store_true', help='Visualize the generated trajectory with reconstructed points')
    args = parser.parse_args()

    # SAMPLED PARAMETERS (YOU SHOULD MODIFY THESE)
    sampled_config = {
        # pose sampling
        'camera_center': (0, 0, 0),
        'camera_radius': 1.0,
        'n_views': 50,
        'theta_angle': 30,       # degrees
        'phi_range': (0, 360),   # degrees
        # intrinsics
        'width': 1296,
        'height': 840,
        'fx': 960.98,
        'fy': 963.15,
        'cx': 648,
        'cy': 420
    }

    traj_output_dir = os.path.join(args.dataset_dir, 'custom_camera_path')
    os.makedirs(traj_output_dir, exist_ok=True)

    traj_output_path = os.path.join(traj_output_dir, args.traj_name + '.json')

    sample_camera_poses(args.dataset_dir, traj_output_path, sampled_config, args.vis_traj)