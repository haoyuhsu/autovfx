import glob
import numpy as np
import os
from colmap_read_model import *
import json


# rotate all camera poses based on the estimated up vector of the scene
# scene_up_vector_dict = {
#     'tnt': {
#         'Playground': [-0.00720354, -0.9963133, -0.08548705],
#     },
#     '360': {
#         'bonsai': [ 0.02405242, -0.77633506, -0.6298614 ],
#         'counter': [ 0.07449666, -0.80750495, -0.5851376 ],
#         'garden': [-0.03292375, -0.8741887, -0.48446894],
#     },
#     'lerf': {
#         'donuts': [0.0, 0.0, 1.0],
#         'dozer_nerfgun_waldo': [-0.76060444, 0.00627117, 0.6491853 ],
#         'espresso': [0.0, 0.0, 1.0],
#         'figurines': [0.0, 0.0, 1.0],
#         'ramen': [0.0, 0.0, 1.0],
#         'shoe_rack': [0.0, 0.0, 1.0],
#         'teatime': [0.0, 0.0, 1.0],
#         'waldo_kitchen': [0.0, 0.0, 1.0],
#     }
# }

# Updated Version
# rotate all camera poses based on the estimated up vector of the scene
scene_up_vector_dict = {
    'tnt': {
        'Playground': [-0.00720354, -0.9963133, -0.08548705],
    },
    '360': {
        'bonsai': [ 0.02405242, -0.77633506, -0.6298614 ],
        'counter': [ 0.07449666, -0.80750495, -0.5851376 ],
        'garden': [-0.03292375, -0.8741887, -0.48446894],

        'donuts': [ 0.07987297, -0.8506788, -0.5195825 ],  
        'dozer_nerfgun_waldo': [ 0.1031235, -0.83134925, -0.5460989 ],
        'espresso': [ 0.0531004, -0.8072565, -0.58780724],
        'figurines': [ 0.16696297, -0.9803059, -0.10546955],
        'ramen': [ 0.02134954, -0.74014527, -0.6721081 ],
        'shoe_rack': [ 0.00508022, -0.8688783, -0.4949998 ],
        'teatime': [ 0.0540938, -0.8366087, -0.54512364],
        'waldo_kitchen': [-0.01319592, -0.9988512, -0.04606834],
    },
    'lerf': {
        'donuts': [ 0.07987297, -0.8506788, -0.5195825 ],  
        'dozer_nerfgun_waldo': [ 0.1031235, -0.83134925, -0.5460989 ],
        'espresso': [ 0.0531004, -0.8072565, -0.58780724],
        'figurines': [ 0.16696297, -0.9803059, -0.10546955],
        'ramen': [ 0.02134954, -0.74014527, -0.6721081 ],
        'shoe_rack': [ 0.00508022, -0.8688783, -0.4949998 ],
        'teatime': [ 0.0540938, -0.8366087, -0.54512364],
        'waldo_kitchen': [-0.01319592, -0.9988512, -0.04606834],
    }
}


def read_json(path):
    with open(path) as f:
        content = json.load(f)
        return content
    

# get rotation matrix from aligning one vector to another vector
def get_rotation_matrix_from_vectors(v1, v2):
    # if two numpy array are the same, return identity matrix
    if np.allclose(v1, v2):
        return np.eye(3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s ** 2)
    # print('R:', R)
    return R


def align_pose_c2w(pose, up_vector):
    '''
    Align a single pose in c2w by rotation with up vector
    '''
    v1 = up_vector
    v2 = np.array([0., 0., 1.])
    R = get_rotation_matrix_from_vectors(v1, v2)
    # rotate c2w matrix by R
    pose = R @ pose
    return pose, R


def convert_c2w_to_w2c(c2w):
    '''
    Convert camera to world to world to camera (3 x 4)
    '''
    R = c2w[:, :3]  # Extract rotation matrix (first 3 columns)
    t = c2w[:, 3]   # Extract translation vector (last column)

    R_inv = R.T  # Transpose of rotation matrix
    t_inv = -np.dot(R_inv, t)  # Inverted translation

    w2c = np.column_stack((R_inv, t_inv))  # Combine them back
    return w2c


def convert_w2c_to_c2w(w2c):
    '''
    Convert world to camera to camera to world (3 x 4)
    '''
    R = w2c[:, :3]  # Extract rotation matrix (first 3 columns)
    t = w2c[:, 3]   # Extract translation vector (last column)

    R_inv = R.T  # Transpose of rotation matrix
    t_inv = -np.dot(R_inv, t)  # Inverted translation

    c2w = np.column_stack((R_inv, t_inv))  # Combine them back
    return c2w


def read_tnt_poses(dataset_dir):
    '''
    Read Tanks and Temples c2w poses (3 x 4)
    '''
    scene_name = dataset_dir.split('/')[-1]
    up_vector = scene_up_vector_dict['tnt'][scene_name]
    poses_file = sorted(glob.glob(dataset_dir + "/pose/*.txt"))
    poses = {}
    for pose_file in poses_file:
        filename = pose_file.split('/')[-1].split('.')[0]
        pose = np.loadtxt(pose_file)
        pose = pose.reshape(-1, 4)
        pose = pose[:3, :]          # (3, 4)
        pose, _ = align_pose_c2w(pose, up_vector)
        poses[filename + ".png"] = pose
    return poses


def read_tnt_intrinsics(dataset_dir):
    '''
    Read Tanks and Temples intrinsics (3 x 3)
    '''
    intrinsic_file = dataset_dir + "/intrinsics.txt"
    intrinsics = np.loadtxt(intrinsic_file)
    intrinsics = intrinsics[:3, :3]
    return [intrinsics]


def read_360_poses(dataset_dir):
    '''
    Read 360 dataset w2c poses (3 x 4)
    '''
    scene_name = dataset_dir.split('/')[-1]
    up_vector = scene_up_vector_dict['360'][scene_name]
    poses_file = dataset_dir + "/sparse/0/images.bin"
    imdata = read_images_binary(poses_file)
    # already w2c
    poses = {}
    for k in imdata:
        im = imdata[k]
        name = im.name
        R = im.qvec2rotmat()
        t = im.tvec.reshape(3, 1)
        pose = np.concatenate([R, t], 1)
        pose = convert_w2c_to_c2w(pose)
        pose, _ = align_pose_c2w(pose, up_vector)  # align only in c2w
        # pose = convert_c2w_to_w2c(pose)
        poses[name] = pose
    ##############################
    # make sure the poses are zero-centered and normalized
    ##############################
    # normzliae c2w poses
    cam_centers = []
    for f_name, c2w in poses.items():
        cam_centers.append(c2w[:3, 3:4])
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    # the update parameters
    radius = diagonal * 1.1
    translate = -center
    # update the poses
    for f_name in poses:
        poses[f_name][:3, 3:4] += translate
        poses[f_name][:3, 3:4] /= radius
    ##############################
    # convert to w2c
    for f_name, c2w in poses.items():
        poses[f_name] = convert_c2w_to_w2c(c2w)
    return poses


def read_360_intrinsics(dataset_dir):
    '''
    Read 360 dataset intrinsics (3 x 3)
    '''
    cam_file = dataset_dir + "/sparse/0/cameras.bin"
    camdata = read_cameras_binary(cam_file)

    if camdata[1].model == 'SIMPLE_RADIAL':
        fx = fy = camdata[1].params[0]
        cx = camdata[1].params[1]
        cy = camdata[1].params[2]
    elif camdata[1].model in ['PINHOLE', 'OPENCV']:
        fx = camdata[1].params[0]
        fy = camdata[1].params[1]
        cx = camdata[1].params[2]
        cy = camdata[1].params[3]
    else:
        raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        
    intrinsics = np.array(
        [[fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]]).reshape(3, 3)
    
    return [intrinsics]


def read_lerf_poses_and_intrinsics(dataset_dir):
    '''
    Read LERF dataset c2w poses (3 x 4) and intrinsics (3 x 3)
    '''
    scene_name = dataset_dir.split('/')[-1]
    up_vector = scene_up_vector_dict['lerf'][scene_name]

    meta_file = dataset_dir + "/transforms.json"
    
    # read poses and intrinsics of each frame from json file
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    # Sort the 'frames' list by 'file_path' to make sure that the order of images is correct
    # https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary
    all_file_paths = [frame_info['file_path'] for frame_info in meta['frames']]
    sort_indices = [i[0] for i in sorted(enumerate(all_file_paths), key=lambda x:x[1])]
    meta['frames'] = [meta['frames'][i] for i in sort_indices]

    # get c2w poses
    poses = {}
    for frame_info in meta['frames']:
        frame_name = frame_info['file_path'].split('/')[-1]
        cam_mtx = np.array(frame_info['transform_matrix'])
        cam_mtx = cam_mtx @ np.diag([1, -1, -1, 1])  # OpenGL to OpenCV camera
        pose = cam_mtx[:3, :]
        pose, _ = align_pose_c2w(cam_mtx, up_vector)
        poses[frame_name] = pose  # (3, 4)

    ##############################
    # make sure the poses are zero-centered and normalized
    ##############################
    # normzliae c2w poses
    # cam_centers = []
    # for f_name, c2w in poses.items():
    #     cam_centers.append(c2w[:3, 3:4])
    # cam_centers = np.hstack(cam_centers)
    # avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    # center = avg_cam_center
    # dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    # diagonal = np.max(dist)
    # # the update parameters
    # radius = diagonal * 1.1
    # translate = -center
    # # update the poses
    # for f_name in poses:
    #     poses[f_name][:3, 3:4] += translate
    #     poses[f_name][:3, 3:4] /= radius
    ##############################

    # get intrinsics
    all_K = []
    if 'fl_x' in meta:
        fx, fy, cx, cy = meta['fl_x'], meta['fl_y'], meta['cx'], meta['cy']
        cam_K = np.array([
            [fx, 0, cx], 
            [0, fy, cy],
            [0, 0, 1]]
        )
        all_K.append(cam_K)
    else:
        for frame_info in meta['frames']:
            fx, fy, cx, cy = frame_info['fl_x'], frame_info['fl_y'], frame_info['cx'], frame_info['cy']
            cam_K = np.array([
                [fx, 0, cx], 
                [0, fy, cy],
                [0, 0, 1]]
            )
            all_K.append(cam_K)
            break # only need to read one K

    return poses, all_K


def read_scannet_poses_and_intrinsics(dataset_dir):
    '''
    Read ScanNet++ dataset c2w poses (3 x 4) and intrinsics (3 x 3)
    '''
    scene_name = dataset_dir.split('/')[-1]

    # camera extrinsics, load from colmap files which is aligned with scans
    images_path = os.path.join(dataset_dir, 'dslr', 'colmap', 'images.txt')
    imdata = read_images_text(images_path)
    img_names = [imdata[k].name for k in imdata]
    w2c_mats = []
    bottom = np.array([[0, 0, 0, 1.]])
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
    w2c_mats = np.stack(w2c_mats, 0)
    c2w = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4) cam2world matrices
    c2w_dict = {img_names[i]: c2w[i] for i in range(len(img_names))}

    # camera intrinsics
    meta = read_json(os.path.join(dataset_dir, 'dslr', 'nerfstudio', 'transforms_undistorted.json'))
    res_scale = 1.0
    all_K = []
    if 'fl_x' in meta:
        fx, fy, cx, cy = meta['fl_x'], meta['fl_y'], meta['cx'], meta['cy']
        cam_K = np.array([
            [fx, 0, cx], 
            [0, fy, cy],
            [0, 0, 1]]
        )
        cam_K[:2] *= res_scale
        all_K.append(cam_K)
    else:
        print('Only support one camera intrinsics for ScanNet++ dataset!')
    img_wh = np.array([meta['w'], meta['h']])

    return c2w_dict, all_K, img_wh


def read_nerfstudio_poses_and_intrinsics(dataset_dir):
    '''
    Read NerfStudio dataset c2w poses (3 x 4) and intrinsics (3 x 3)
    '''
    meta_file = os.path.join(dataset_dir, "transforms.json")
    
    # read poses and intrinsics of each frame from json file
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    
    # camera extrinsics, load from colmap files which is aligned with scans
    all_file_paths = [frame_info['file_path'] for frame_info in meta['frames']]
    sort_indices = [i[0] for i in sorted(enumerate(all_file_paths), key=lambda x:x[1])]
    meta['frames'] = [meta['frames'][i] for i in sort_indices]

    # get c2w poses
    c2w_dict = {}
    for frame_info in meta['frames']:
        frame_name = frame_info['file_path'].split('/')[-1]
        cam_mtx = np.array(frame_info['transform_matrix'])
        cam_mtx = cam_mtx @ np.diag([1, -1, -1, 1])  # OpenGL to OpenCV camera
        pose = cam_mtx[:3, :]
        c2w_dict[frame_name] = pose  # (3, 4)

    # camera intrinsics
    res_scale = 1.0
    all_K = []
    if 'fl_x' in meta:
        fx, fy, cx, cy = meta['fl_x'], meta['fl_y'], meta['cx'], meta['cy']
        cam_K = np.array([
            [fx, 0, cx], 
            [0, fy, cy],
            [0, 0, 1]]
        )
        cam_K[:2] *= res_scale
        all_K.append(cam_K)
    else:
        print('Only support one camera intrinsics for NerfStudio dataset!')
    img_wh = np.array([meta['w'], meta['h']])

    return c2w_dict, all_K, img_wh