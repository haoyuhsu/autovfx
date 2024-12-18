# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License
# Authors:
#  - Suttisak Wizadwongsa <suttisak.w_s19[-at-]vistec.ac.th>
#  - Pakkapon Phongthawee <pakkapon.p_s19[-at-]vistec.ac.th>
#  - Jiraphon Yenphraphai <jiraphony_pro[-at-]vistec.ac.th>
#  - Supasorn Suwajanakorn <supasorn.s[-at-]vistec.ac.th>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import glob
from scipy import misc
import sys
import json
import argparse
from colmap_read_model import *
import shutil
from read_dataset import *
from database import modify_db
from PIL import Image
from tqdm import tqdm
import cv2


def cmd(s):
  print(s)
  exit_code = os.system(s)
  if exit_code != 0:
    print("Error: ", exit_code)
    exit(exit_code)


def write_images_txt(poses, intrinsics, output_path, is_c2w=True):
  '''
  Write images.txt used for COLMAP
  Input:
    poses: dictionary of poses
    output_path: path to save images.txt
  '''
  # COLMAP format
  # 1st line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
  # 2nd line: leave it blank~~
  with open(output_path, 'w') as f:
    for idx, img_name in enumerate(sorted(poses.keys())):
      pose = poses[img_name]
      if is_c2w:
        pose = convert_c2w_to_w2c(pose[:3, :])  # (3, 4)
      Q = rotmat2qvec(pose[:3, :3])
      T = pose[:3, 3]

      IMAGE_ID = idx+1
      QW, QX, QY, QZ = Q
      TX, TY, TZ = T
      if len(intrinsics) == 1:
        CAMERA_ID = 1
      else:
        CAMERA_ID = idx+1
      NAME = img_name

      f.writelines(f"{IMAGE_ID} {QW} {QX} {QY} {QZ} {TX} {TY} {TZ} {CAMERA_ID} {NAME}\n")
      f.write('\n')


def write_cameras_txt(intrinsics, output_path, HEIGHT, WIDTH):
  '''
  Write cameras.txt used for COLMAP
  Input:
    intrinsics: (list of) 3 x 3 intrinsics matrix
    output_path: path to save cameras.txt
  '''
  # Camera list with one line of data per camera:
  #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
  with open(output_path, 'w') as f:
    for idx, intrinsics in enumerate(intrinsics):
      CAMERA_ID = idx+1
      MODEL = "PINHOLE"
      PARAMS = intrinsics
      f.writelines(f"{CAMERA_ID} {MODEL} {WIDTH} {HEIGHT} {PARAMS[0,0]} {PARAMS[1,1]} {PARAMS[0,2]} {PARAMS[1,2]}\n")
      f.write('\n')
  

def runner_from_scratch(dataset_dir, output_dir):
  '''
    Use colmap without any prior
      1. feature extractor
      2. exhaustive matcher
      3. Mapper
  '''
  # Feature Extraction (use single camera model)
  cmd("colmap feature_extractor \
    --database_path " + output_dir + "/database.db \
    --image_path " + dataset_dir + "/images \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 " )

  # Feature Matching
  cmd("colmap exhaustive_matcher \
    --database_path " + output_dir + "/database.db \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.use_gpu 1")
  
  os.makedirs(output_dir + "/sparse/0", exist_ok=True)

  # Mapper (bundle adjustment)
  cmd("colmap mapper \
    --database_path " + output_dir + "/database.db \
    --image_path " + dataset_dir + "/images \
    --output_path " + output_dir + "/sparse \
    --Mapper.ba_global_function_tolerance=0.000001")

  # Convert .bin to .txt
  cmd("colmap model_converter \
    --input_path " + output_dir + "/sparse/0 \
    --output_path " + output_dir + "/sparse/0 \
    --output_type TXT")
  

def runner_with_known_poses(dataset_dir, output_dir):
  '''
    Use colmap with known poses and intrinsics
      1. feature extractor
      2. modify database
      3. exhaustive matcher
      4. point triangulation 
  '''
  # Feature Extraction  
  cmd("colmap feature_extractor \
    --database_path " + output_dir + "/database.db \
    --image_path " + dataset_dir + "/images \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.use_gpu 1 " )
  
  # modify database
  modify_db(
    output_dir + "/database.db", \
    output_dir + "/images.txt", \
    output_dir + "/cameras.txt")

  # Feature Matching
  cmd("colmap exhaustive_matcher \
    --database_path " + output_dir + "/database.db \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.use_gpu 1")

  os.makedirs(output_dir + "/sparse/0", exist_ok=True)

  cmd("cp " + output_dir + "/images.txt " + output_dir + "/sparse/0/images.txt")
  cmd("cp " + output_dir + "/cameras.txt " + output_dir + "/sparse/0/cameras.txt")
  cmd("touch " + output_dir + "/sparse/0/points3D.txt")

  # Point Triangulation
  cmd("colmap point_triangulator \
    --database_path " + output_dir + "/database.db \
    --image_path " + dataset_dir + "/images \
    --input_path " + output_dir + "/sparse/0 \
    --output_path " + output_dir + "/sparse/0")

  # Convert .bin to .txt
  cmd("colmap model_converter \
    --input_path " + output_dir + "/sparse/0 \
    --output_path " + output_dir + "/sparse/0 \
    --output_type TXT")
  

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_dir', type=str, default='datasets/tmp/', help='Path to dataset folder')
  parser.add_argument('--text_prompt', type=str, default='ground', help='Text prompt for flat surface (e.g., floor, plane, table, ground, etc.)')
  args = parser.parse_args()

  ##### (1) SfM from scratch, make sure all images are under dataset_dir/images #####
  colmap_first_output_dir = os.path.join(args.dataset_dir, "colmap_first_stage")
  os.makedirs(colmap_first_output_dir, exist_ok=True)
  runner_from_scratch(args.dataset_dir, colmap_first_output_dir)

  ##### (2) convert reconstructed poses & intrinsics to transforms.json #####
  camdata, imdata, _ = read_model(os.path.join(colmap_first_output_dir, "sparse/0"))
  transforms = {}
  transforms["camera_model"] = "OPENCV"
  fx, fy, cx, cy = camdata[1].params
  transforms["fl_x"] = fx
  transforms["fl_y"] = fy
  transforms["cx"] = cx
  transforms["cy"] = cy
  transforms["w"] = camdata[1].width
  transforms["h"] = camdata[1].height
  frames_info = []
  img_names = [imdata[k].name for k in imdata]
  perm = np.argsort(img_names)
  w2c_mats = []
  bottom = np.array([[0, 0, 0, 1.]])
  for k in imdata:
    im = imdata[k]
    R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
    w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
  w2c_mats = np.stack(w2c_mats, 0)
  poses = np.linalg.inv(w2c_mats)[perm] # (N_images, 4, 4) cam2world matrices
  img_names_sorted = [img_names[i] for i in perm]
  for img_name, pose in zip(img_names_sorted, poses):
    frame_info = {}
    frame_info["file_path"] = os.path.join("images", img_name)
    frame_info["transform_matrix"] = pose.tolist()
    frames_info.append(frame_info)
  transforms["frames"] = frames_info
  with open(os.path.join(args.dataset_dir, "transforms.json"), "w") as f:
    json.dump(transforms, f, indent=4)

  ##### (3) use monocular normal map to realign the camera poses #####

  # Get the flat surface mask for each image
  gsam_output_dir = os.path.join(args.dataset_dir, "gsam_output")
  os.makedirs(gsam_output_dir, exist_ok=True)
  cmd("python tracking/Grounded-Segment-Anything/grounded_sam_demo.py \
    --config tracking/saves/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint tracking/saves/groundingdino_swint_ogc.pth \
    --sam_checkpoint tracking/saves/sam_vit_h_4b8939.pth \
    --input_image_dir " + os.path.join(args.dataset_dir, "images") + " \
    --output_dir " + gsam_output_dir + " \
    --box_threshold 0.3 \
    --text_threshold 0.4 \
    --text_prompt " + args.text_prompt + " \
    --device cuda"
  )

  # Get monocular normal prediction for each image (use either omni-data, Metric3D-v2, or DSINE)
  normal_output_dir = os.path.join(args.dataset_dir, "normal")
  if not os.path.exists(normal_output_dir):
    raise ValueError("Please provide monocular normal map for each image")
  
  # Align the camera poses using the normal map
  with open(os.path.join(args.dataset_dir, "transforms.json"), "r") as f:
    transforms = json.load(f)

  normals = []
  for frame in transforms["frames"]:
    img_name = frame["file_path"].split("/")[-1]
    normal_path = os.path.join(normal_output_dir, img_name[:-4] + "_normal.png")
    normal = np.array(Image.open(normal_path))
    normal = normal.astype(np.float32) / 255.0        # normalize to [0, 1]
    normal = (normal - 0.5) * 2                       # normalize to [-1, 1]
    C2W = np.array(frame["transform_matrix"])         # (4, 4) camera-to-world matrix
    normal_world = normal @ C2W[:3, :3].T             # transform normal to world space
    normal_world = normal_world / np.linalg.norm(normal_world, axis=-1, keepdims=True)  # normalize to unit length
    mask_path = os.path.join(gsam_output_dir, img_name[:-4] + ".jpg")
    mask = np.array(Image.open(mask_path)) > 0
    if not mask.any():
      continue
    normals.append(normal_world[mask])
  normals = np.concatenate(normals, axis=0)

  # RANSAC
  n_iter = 100
  n_sample = 10000
  threshold = 0.99
  best_inliers = 0
  best_normal = None
  for i in tqdm(range(n_iter)):
    idx = np.random.choice(len(normals), n_sample, replace=False)
    normal = np.mean(normals[idx], axis=0)
    normal /= np.linalg.norm(normal)
    inliers = normals @ normal > threshold
    n_inliers = np.sum(inliers)
    if n_inliers > best_inliers:
      print(f'Inliers: {n_inliers/len(normals)}')
      best_inliers = n_inliers
      best_normal = normal

  # Get the rotation matrix that aligns the best normal to (0, 0, 1)
  v = np.array([0, 0, 1])
  normal_temp = best_normal.copy()
  R_all = np.eye(3)
  for i in range(100):
    axis = np.cross(normal_temp, v)
    angle = np.arccos(np.dot(normal_temp, v))
    R = cv2.Rodrigues(axis * angle)[0]
    normal_temp = R @ normal_temp
    R_all = R @ R_all
  normal_aligned = R_all @ best_normal
  np.save(os.path.join(args.dataset_dir, 'rotation_aligned.npy'), R_all)

  # Update the camera poses with the aligned rotation matrix, and normalize poses to unit length
  new_transforms = transforms.copy()
  c2ws = np.array([frame['transform_matrix'] for i, frame in enumerate(transforms['frames'])])
  ts = c2ws[:, :3, 3]
  t_max = np.max(ts, axis=0)
  t_min = np.min(ts, axis=0)
  t_center = (t_max + t_min) / 2
  scale = np.max(t_max - t_min) / 2
  ts_normalized = (ts - t_center) / scale
  c2ws[:, :3, 3] = ts_normalized
  R_align = np.eye(4)
  R_align[:3, :3] = np.load(os.path.join(args.dataset_dir, 'rotation_aligned.npy'))
  c2ws = np.matmul(R_align, c2ws)
  for i in range(len(transforms['frames'])):
    new_transforms['frames'][i]['transform_matrix'] = c2ws[i].tolist()
  with open(os.path.join(args.dataset_dir, "transforms.json"), "w") as f:
    json.dump(new_transforms, f, indent=4)

  ##### (4) re-run colmap with known poses and intrinsics #####
  with open(os.path.join(args.dataset_dir, "transforms.json"), "r") as f:
    transforms = json.load(f)
  fx, fy, cx, cy = transforms["fl_x"], transforms["fl_y"], transforms["cx"], transforms["cy"]
  h, w = transforms["h"], transforms["w"]
  intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
  poses = {}
  for frame in transforms["frames"]:
    img_name = frame["file_path"].split("/")[-1]
    poses[img_name] = np.array(frame["transform_matrix"])
  colmap_second_output_dir = os.path.join(args.dataset_dir, "colmap_second_stage")
  os.makedirs(colmap_second_output_dir, exist_ok=True)
  write_cameras_txt([intrinsics], os.path.join(colmap_second_output_dir, "cameras.txt"), h, w)
  write_images_txt(poses, [intrinsics], os.path.join(colmap_second_output_dir, "images.txt"), is_c2w=True)
  runner_with_known_poses(args.dataset_dir, colmap_second_output_dir)
  cmd('cp -r ' + colmap_second_output_dir + '/sparse ' + args.dataset_dir)

  ##### (5) convert the camera poses from OpenCV to OpenGL (used in SDF training) #####
  transforms = {}
  with open(os.path.join(args.dataset_dir, "transforms.json"), "r") as f:
    transforms = json.load(f)
  for i in range(len(transforms["frames"])):
    c2w_opencv = np.array(transforms["frames"][i]["transform_matrix"])
    c2w_opengl = c2w_opencv @ np.diag([1, -1, -1, 1])
    transforms["frames"][i]["transform_matrix"] = c2w_opengl.tolist()
  with open(os.path.join(args.dataset_dir, "transforms.json"), "w") as f:
    json.dump(transforms, f, indent=4)
    
