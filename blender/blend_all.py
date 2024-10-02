import numpy as np
import os 
import argparse
from PIL import Image, ImageFilter
import cv2
import imageio.v2 as imageio
from tqdm import tqdm
import glob
import skimage
import json


"""
Blending frames for all visual effects
"""


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def downsample_image(image, new_size):
    img = Image.fromarray(image)
    if len(image.shape) == 3:
        # img = img.filter(ImageFilter.GaussianBlur(radius=2))  # anti-aliasing for rgb image
        img = img.resize(new_size, resample=Image.BILINEAR)
    else:
        img = img.resize(new_size, Image.NEAREST)   # use nearest neighbour for depth map
    return np.array(img)


def generate_video_from_frames(frame_series, video_name, fps=30):
    # frame_series = [np.array(Image.open(frame_path)) for frame_path in frames_path]  # return (0~255 in uint8)
    # reshape the size of the frames to be divisible by 2 for video rendering
    h, w = frame_series[0].shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    frame_series = [(skimage.transform.resize(frame, (new_h, new_w)) * 255.).astype(np.uint8) for frame in frame_series]
    # generate video with proper quality
    imageio.mimsave(video_name,
        frame_series,
        fps=fps,
        macro_block_size=1
    )
    # generate video with high quality
    # imageio.mimsave(video_name,
    #     frame_series,
    #     fps=fps, 
    #     codec='libx264', 
    #     macro_block_size=None, 
    #     quality=10,
    #     pixelformat='yuv444p'
    # )
    print("Video saved at: {}".format(video_name))


def load_rgb(path):
    if not os.path.exists(path):
        return None
    else:
        return np.array(Image.open(path).convert("RGBA"))


def load_depth(path):
    if not os.path.exists(path):
        return None
    else:
        return np.load(path)


def load_depth_exr(path):
    if not os.path.exists(path):
        return None
    else:
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return d[:, :, 0]


def depth_check(depth1, depth2, option='naive', d_tol=0.1):
    '''
    Determine whether depth1 is closer than depth2 with a tolerance d_tol
    '''
    if option == 'naive':
        return depth1 <= depth2
    elif option == 'tolerance':
        return np.abs(depth1 - depth2) < d_tol
    elif option == 'naive_or_tolerance':
        return np.logical_or(depth1 <= depth2, np.abs(depth1 - depth2) < d_tol)
    else:
        raise ValueError('Invalid option: {}'.format(option))


def blend_frames(blend_results_dir, input_config_path=None):

    root_dir = os.path.dirname(os.path.normpath(blend_results_dir))

    # has_3dgs, has_smoke, has_fire = False, False, False
    # if os.path.exists(os.path.join(blend_results_dir, 'rgb_obj_3dgs')):
    #     has_3dgs = True
    # if os.path.exists(os.path.join(blend_results_dir, 'rgb_smoke_fire')):
    #     has_smoke = True
    # if os.path.exists(os.path.join(blend_results_dir, 'rgb_smoke_fire_pre')):
    #     has_smoke = True
    #     has_fire = True

    # preload all frames path instead of loading all frames into memory
    if input_config_path is not None:
        with open(input_config_path, 'r') as f:
            input_config = json.load(f)
        if 'render_type' in input_config and input_config['render_type'] == 'SINGLE_VIEW':
            anchor_frame_idx = input_config['anchor_frame_idx']
            num_frames = input_config['num_frames']
            anchor_frame_rgb_path = os.path.join(root_dir, 'images', f'{anchor_frame_idx:05}.png')
            anchor_frame_depth_path = os.path.join(root_dir, 'depth', f'{anchor_frame_idx:05}.npy')
            # copy both rgb & depth paths for num_frames times into bg_rgb and bg_depth
            bg_rgb = [anchor_frame_rgb_path] * num_frames
            bg_depth = [anchor_frame_depth_path] * num_frames
        else:
            # default: MULTI_VIEW option
            bg_rgb = sorted(glob.glob(os.path.join(root_dir, 'images', '*.png')))
            bg_depth = sorted(glob.glob(os.path.join(root_dir, 'depth', '*.npy')))
    else:
        bg_rgb = sorted(glob.glob(os.path.join(root_dir, 'images', '*.png')))
        bg_depth = sorted(glob.glob(os.path.join(root_dir, 'depth', '*.npy')))

    # obj_rgb = sorted(glob.glob(os.path.join(blend_results_dir, 'rgb_obj', '*.png')))
    # obj_depth = sorted(glob.glob(os.path.join(blend_results_dir, 'depth_obj', '*/*.exr'), recursive=True))
    # shadow_rgb = sorted(glob.glob(os.path.join(blend_results_dir, 'rgb_shadow', '*.png')))
    # shadow_depth = sorted(glob.glob(os.path.join(blend_results_dir, 'depth_shadow', '*/*.exr'), recursive=True))
    # all_rgb = sorted(glob.glob(os.path.join(blend_results_dir, 'rgb_all', '*.png')))
    # all_depth = sorted(glob.glob(os.path.join(blend_results_dir, 'depth_all', '*/*.exr'), recursive=True))

    # if has_3dgs:
    #     obj_3dgs_rgb = sorted(glob.glob(os.path.join(blend_results_dir, 'rgb_obj_3dgs', '*.png')))
    #     obj_3dgs_depth = sorted(glob.glob(os.path.join(blend_results_dir, 'depth_obj_3dgs', '*/*.exr'), recursive=True))

    # if has_smoke:
    #     smoke_fire_rgb = sorted(glob.glob(os.path.join(blend_results_dir, 'rgb_smoke_fire', '*.png')))
    #     smoke_fire_depth = sorted(glob.glob(os.path.join(blend_results_dir, 'depth_smoke_fire', '*/*.exr'), recursive=True))
    #     if has_fire:
    #         smoke_fire_rgb_pre = sorted(glob.glob(os.path.join(blend_results_dir, 'rgb_smoke_fire_pre', '*.png')))
    #         smoke_fire_depth_pre = sorted(glob.glob(os.path.join(blend_results_dir, 'depth_smoke_fire_pre', '*/*.exr'), recursive=True))

    rgb_all_img_path = glob.glob(os.path.join(blend_results_dir, 'rgb_all', '*.png'))  # this would still output a video even if Blender crashes
    n_frame = len(rgb_all_img_path)

    # n_frame = len(bg_rgb)
    out_img_dir = os.path.join(blend_results_dir, 'frames')
    os.makedirs(out_img_dir, exist_ok=True)
    
    ############################################################
    # store temporary results
    ############################################################
    # save_temp_results = True
    # if save_temp_results:
    #     orig_frames = []
    #     fg_obj_frames = []
    #     fg_obj_mask_frames = []
    #     fg_obj_shadow_frames = []
    #     shadow_frames = []
    #     shadow_catcher_frames = []

    #     before_shadow_frames = [] # for debugging
    ############################################################

    ################################################
    # Example format of image paths
    ################################################
    # /depth_all/001/Image0001.exr
    # /depth_obj/001/Image0001.exr
    # /depth_obj_3dgs/001/Image0001.exr
    # /depth_shadow/001/Image0001.exr
    # /depth_smoke_fire/001/Image0001.exr
    # /depth_smoke_fire_pre/001/Image0001.exr
    # /rgb_all/001.png
    # /rgb_obj/001.png
    # /rgb_obj_3dgs/001.png
    # /rgb_shadow/001.png
    # /rgb_smoke_fire/001.png
    # /rgb_smoke_fire_pre/001.png
    ################################################

    frames = []
    for i in tqdm(range(n_frame)):

        # Get the paths for each frame
        obj_rgb_path = os.path.join(blend_results_dir, 'rgb_obj', '{:0>3d}.png'.format(i+1))
        obj_depth_path = os.path.join(blend_results_dir, 'depth_obj', '{:0>3d}'.format(i+1), 'Image{:0>4d}.exr'.format(i+1))
        shadow_rgb_path = os.path.join(blend_results_dir, 'rgb_shadow', '{:0>3d}.png'.format(i+1))
        shadow_depth_path = os.path.join(blend_results_dir, 'depth_shadow', '{:0>3d}'.format(i+1), 'Image{:0>4d}.exr'.format(i+1))
        all_rgb_path = os.path.join(blend_results_dir, 'rgb_all', '{:0>3d}.png'.format(i+1))
        all_depth_path = os.path.join(blend_results_dir, 'depth_all', '{:0>3d}'.format(i+1), 'Image{:0>4d}.exr'.format(i+1))

        obj_3dgs_rgb_path = os.path.join(blend_results_dir, 'rgb_obj_3dgs', '{:0>3d}.png'.format(i+1))
        obj_3dgs_depth_path = os.path.join(blend_results_dir, 'depth_obj_3dgs', '{:0>3d}'.format(i+1), 'Image{:0>4d}.exr'.format(i+1))
        smoke_fire_rgb_path = os.path.join(blend_results_dir, 'rgb_smoke_fire', '{:0>3d}.png'.format(i+1))
        smoke_fire_depth_path = os.path.join(blend_results_dir, 'depth_smoke_fire', '{:0>3d}'.format(i+1), 'Image{:0>4d}.exr'.format(i+1))
        smoke_fire_rgb_pre_path = os.path.join(blend_results_dir, 'rgb_smoke_fire_pre', '{:0>3d}.png'.format(i+1))
        smoke_fire_depth_pre_path = os.path.join(blend_results_dir, 'depth_smoke_fire_pre', '{:0>3d}'.format(i+1), 'Image{:0>4d}.exr'.format(i+1))


        bg_c = load_rgb(bg_rgb[i])                    # bg_c: background image
        bg_d = load_depth(bg_depth[i])                # bg_d: background depth map
        o_c = load_rgb(obj_rgb_path)                  # o_c: object image from Blender
        o_d = load_depth_exr(obj_depth_path)          # o_d: object depth map from Blender
        s_c = load_rgb(shadow_rgb_path)               # s_c: shadow catcher image from Blender
        s_d = load_depth_exr(shadow_depth_path)       # s_d: shadow catcher depth map from Blender
        o_s_c = load_rgb(all_rgb_path)                # o_s_c: object with shadow catcher image from Blender
        o_s_d = load_depth_exr(all_depth_path)        # o_s_d: object with shadow catcher depth map from Blender

        o_gs_c = load_rgb(obj_3dgs_rgb_path)                  # o_c: 3DGS object image from Blender
        o_gs_d = load_depth_exr(obj_3dgs_depth_path)          # o_d: 3DGS object depth map from Blender
        has_3dgs = o_gs_c is not None

        s_f_c = load_rgb(smoke_fire_rgb_path)               # s_f_c: smoke and fire image from Blender
        s_f_d = load_depth_exr(smoke_fire_depth_path)       # s_f_d: smoke and fire depth map from Blender
        has_smoke = s_f_c is not None

        s_f_c_pre = load_rgb(smoke_fire_rgb_pre_path)               # s_f_c_pre: smoke and fire pre-multiplied image from Blender
        s_f_d_pre = load_depth_exr(smoke_fire_depth_pre_path)       # s_f_d_pre: smoke and fire pre-multiplied depth map from Blender
        has_fire = s_f_c_pre is not None

        if has_smoke:
            mask = (s_f_c[..., 3] / 255.) > 0.0         # only overwrite depth values on region with non-zero alphas
            s_f_d[mask] = np.percentile(s_f_d, 0.001)
            # s_f_d[:] = np.percentile(s_f_d, 0.001)    # smoke & fire has no concrete depth values
            # s_f_d[:] = np.percentile(o_d, 0.001)
            if has_fire and s_f_d_pre is not None:
                s_f_d_pre[mask] = np.percentile(s_f_d_pre, 0.001)
                # s_f_d_pre[:] = np.percentile(s_f_d_pre, 0.001)
                # s_f_d_pre[:] = np.percentile(o_d, 0.001)

        # anti-aliasing
        new_size = (bg_c.shape[1], bg_c.shape[0])
        o_c = downsample_image(o_c, new_size)
        o_d = downsample_image(o_d, new_size)
        s_c = downsample_image(s_c, new_size)
        s_d = downsample_image(s_d, new_size)
        o_s_c = downsample_image(o_s_c, new_size)
        o_s_d = downsample_image(o_s_d, new_size)

        if has_3dgs:
            o_gs_c = downsample_image(o_gs_c, new_size)
            o_gs_d = downsample_image(o_gs_d, new_size)

        if has_smoke:
            s_f_c = downsample_image(s_f_c, new_size)
            s_f_d = downsample_image(s_f_d, new_size)
            if has_fire:
                s_f_c_pre = downsample_image(s_f_c_pre, new_size)
                s_f_d_pre = downsample_image(s_f_d_pre, new_size)

        bg_c = bg_c.astype(np.float32)
        o_c = o_c.astype(np.float32)
        s_c = s_c.astype(np.float32)
        o_s_c = o_s_c.astype(np.float32)

        if has_3dgs:
            o_gs_c = o_gs_c.astype(np.float32)

        if has_smoke:
            s_f_c = s_f_c.astype(np.float32)
            if has_fire:
                s_f_c_pre = s_f_c_pre.astype(np.float32)

        # New Implementation of blending
        frame = bg_c.copy()

        ##### Step 1: blend shadow into background image #####
        if has_3dgs:
            depth_mask = depth_check(s_d, o_gs_d, option='naive', d_tol=0.1)
            obj_3dgs_alpha = o_gs_c[..., 3] / 255.
            non_obj_3dgs_alpha = 1. - obj_3dgs_alpha
            non_obj_3dgs_alpha[depth_mask] = 1.0
        
        # if has_smoke or has_fire:
        #     obj_alpha = s_f_c[..., 3] / 255.
        #     depth_mask = depth_check(s_f_d, s_d, option='naive', d_tol=0.1)
        # else:
        #     obj_alpha = o_c[..., 3] / 255.
        #     depth_mask = depth_check(o_d, s_d, option='naive', d_tol=0.1)
            
        ############################################################
        # TODO: test fireball effects
        obj_alpha = o_c[..., 3] / 255.
        depth_mask = depth_check(o_d, s_d, option='naive', d_tol=0.1)

        if has_smoke or has_fire:
            obj_alpha_smoke = s_f_c[..., 3] / 255.
            depth_mask_smoke = depth_check(s_f_d, s_d, option='naive', d_tol=0.1)
            obj_alpha = np.maximum(obj_alpha, obj_alpha_smoke)
            depth_mask = np.logical_or(depth_mask, depth_mask_smoke)
        ############################################################

        obj_mask = obj_alpha > 0.0
        mask = np.logical_and(obj_mask, depth_mask)
        obj_alpha[~mask] = 0.0
        non_object_alpha = 1. - obj_alpha

        fg_alpha = o_s_c[..., 3] / 255.
        if has_3dgs:
            shadow_catcher_alpha = non_object_alpha * fg_alpha * non_obj_3dgs_alpha
        else:
            shadow_catcher_alpha = non_object_alpha * fg_alpha
        shadow_catcher_mask = shadow_catcher_alpha > 0.0

        color_diff = np.ones_like(o_c)
        color_diff[shadow_catcher_mask, 0:3] = o_s_c[shadow_catcher_mask, :3] / (s_c[shadow_catcher_mask, :3] + 1e-6)
        color_diff = np.clip(color_diff, 0, 1)
        shadow_mask = np.logical_not(np.all(np.abs(color_diff - 1) < 0.01, axis=-1))
        mask = shadow_mask

        frame[mask] = frame[mask] * color_diff[mask] * shadow_catcher_alpha[mask, None] + frame[mask] * (1 - shadow_catcher_alpha[mask, None])

        ##### Step 2: blend object and 3DGS object into background image #####
        # obj_alpha = o_c[..., 3] / 255.
        # obj_mask = obj_alpha > 0.0
        # depth_mask = depth_check(o_d, s_d, option='naive', d_tol=0.1)

        # mask = np.logical_and(obj_mask, depth_mask)
        # if has_fire:
        #     mask = depth_mask
        #     frame[:, :, :3][mask] = s_f_c_pre[:, :, :3][mask] + frame[:, :, :3][mask] * (1 - obj_alpha[mask, None])
        # elif has_smoke:
        #     mask = depth_mask
        #     frame[:, :, :3][mask] = s_f_c[:, :, :3][mask] * obj_alpha[mask, None] + frame[:, :, :3][mask] * (1 - obj_alpha[mask, None])
        # else:
        #     frame[:, :, :3][mask] = o_c[:, :, :3][mask] * obj_alpha[mask, None] + frame[:, :, :3][mask] * (1 - obj_alpha[mask, None])

        ############################################################
        # TODO: test fireball effects
        ############################################################
        frame_tmp = frame.copy()
        mask = np.logical_and(obj_mask, depth_mask)
        frame[:, :, :3][mask] = o_c[:, :, :3][mask] * obj_alpha[mask, None] + frame_tmp[:, :, :3][mask] * (1 - obj_alpha[mask, None])
        if has_fire:
            mask = depth_mask_smoke
            frame[:, :, :3][mask] = s_f_c_pre[:, :, :3][mask] + frame_tmp[:, :, :3][mask] * (1 - obj_alpha_smoke[mask, None])
        ############################################################

        ############################################################
        # temporary results (original frame, foreground object, foreground object mask, foreground object with shadow, shadow only)
        ############################################################
        # if save_temp_results:
        #     orig_frame = bg_c.copy()
        #     orig_frame = orig_frame.astype(np.uint8)

        #     fg_obj_frame = o_c.copy()
        #     fg_obj_frame = fg_obj_frame.astype(np.uint8)

        #     fg_obj_mask = np.zeros_like(o_c)
        #     fg_obj_mask[obj_mask] = 255
        #     fg_obj_mask[obj_mask, 3] = o_c[obj_mask, 3]  # keep the original alpha value
        #     fg_obj_mask = fg_obj_mask.astype(np.uint8)

        #     fg_obj_shadow_frame = o_s_c.copy()
        #     fg_obj_shadow_frame = fg_obj_shadow_frame.astype(np.uint8)

        #     color_diff = np.clip(color_diff, 0, 1)  # to avoid numerical issue (clip color_diff to [0, 1])
        #     shadow_frame = color_diff.copy() * 255
        #     shadow_frame = shadow_frame.astype(np.uint8)

        #     shadow_catcher_frame = s_c.copy()
        #     shadow_catcher_frame = shadow_catcher_frame.astype(np.uint8)

        #     orig_frames.append(orig_frame)
        #     fg_obj_frames.append(fg_obj_frame)
        #     fg_obj_mask_frames.append(fg_obj_mask)
        #     fg_obj_shadow_frames.append(fg_obj_shadow_frame)
        #     shadow_frames.append(shadow_frame)
        #     shadow_catcher_frames.append(shadow_catcher_frame)

        #     frame_before_shadow = frame_before_shadow.astype(np.uint8)
        #     before_shadow_frames.append(frame_before_shadow) # for debugging
        ############################################################

        # convert frame to uint8
        frame = np.clip(frame, 0, 255)
        frame = frame.astype(np.uint8)

        frames.append(frame)
        path = os.path.join(out_img_dir, '{:0>4d}.png'.format(i))
        Image.fromarray(frame).save(path)
    
    generate_video_from_frames(np.array(frames), os.path.join(blend_results_dir, 'blended.mp4'), fps=15)

    ############################################################
    # save video for temporary results
    ############################################################
    # FPS = 15
    # if save_temp_results:
    #     generate_video_from_frames(np.array(orig_frames), os.path.join(blend_results_dir, 'orig.mp4'), fps=FPS)
    #     generate_video_from_frames(np.array(fg_obj_frames), os.path.join(blend_results_dir, 'fg_obj.mp4'), fps=FPS)
    #     generate_video_from_frames(np.array(fg_obj_mask_frames), os.path.join(blend_results_dir, 'fg_obj_mask.mp4'), fps=FPS)
    #     generate_video_from_frames(np.array(fg_obj_shadow_frames), os.path.join(blend_results_dir, 'fg_obj_shadow.mp4'), fps=FPS)
    #     generate_video_from_frames(np.array(shadow_frames), os.path.join(blend_results_dir, 'shadow.mp4'), fps=FPS)
    #     generate_video_from_frames(np.array(shadow_catcher_frames), os.path.join(blend_results_dir, 'shadow_catcher.mp4'), fps=FPS)
    #     generate_video_from_frames(np.array(before_shadow_frames), os.path.join(blend_results_dir, 'before_shadow.mp4'), fps=FPS) # for debugging
    ############################################################
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend_results_dir', type=str, required=True, help='root directory of the blend results')
    parser.add_argument('--input_config_path', type=str, default=None, help='path to the blender config file')
    args = parser.parse_args()
    blend_frames(args.blend_results_dir, args.input_config_path)