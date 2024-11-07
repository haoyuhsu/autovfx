import argparse
import torch
import cv2
import numpy as np
import os
from tqdm import tqdm


def get_normal_with_DSINE(image_dir, output_dir):
    normal_predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)
    for img_name in tqdm(sorted(os.listdir(image_dir))):
        input_path = os.path.join(image_dir, img_name)
        output_normal_path = os.path.join(output_dir, img_name[:-4] + '_normal.png')
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        with torch.inference_mode():
            normal = normal_predictor.infer_cv2(image)[0]
            normal = -normal            # outward normal --> inward normal
            normal = (normal + 1) / 2
        normal = (normal * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_normal_path, normal)


def get_normal_with_Metric3D(iamge_dir, output_dir):
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
    model.cuda().eval()

    for img_name in tqdm(sorted(os.listdir(image_dir))):

        input_path = os.path.join(image_dir, img_name)
        output_normal_path = os.path.join(output_dir, img_name[:-4] + '_normal.png')

        rgb_origin = cv2.imread(input_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        orig_h, orig_w = rgb_origin.shape[:2]
        input_size = (616, 1064) # for vit model
        scale = min(input_size[0] / orig_h, input_size[1] / orig_w)
        rgb = cv2.resize(rgb_origin, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_LINEAR)

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        #### normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()

        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({'input': rgb})
        pred_normal = output_dict['prediction_normal'][:, :3, :, :]    # only available for Metric3Dv2 i.e., ViT models

        # unpad and resize to some size if needed
        pred_normal = pred_normal.squeeze()
        pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
        
        pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
        pred_normal_vis = (pred_normal_vis + 1) / 2
        pred_normal_vis = (pred_normal_vis * 255).astype(np.uint8)
        pred_normal_vis = cv2.resize(pred_normal_vis, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        pred_normal_vis = cv2.cvtColor(pred_normal_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_normal_path, pred_normal_vis)


def get_normal_with_omnidata(image_dir, output_dir):
    # Please check the forked version of Omnidata from here: https://github.com/zhihao-lin/omnidata
    # Example running command usage will be: omnidata/scripts/sdfstudio.sh
    assert False, 'Please refer to the forked version of Omnidata here: https://github.com/zhihao-lin/omnidata \
                    and check the example script omnidata/scripts/sdfstudio.sh'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets/tmp/', help='Path to dataset folder')
    parser.add_argument('--method', type=str, choices=['metric3d', 'dsine', 'omnidata'], default='Metric3D', help='Method for monocular normal estimation')
    args = parser.parse_args()

    image_dir = os.path.join(args.dataset_dir, 'images')
    output_dir = os.path.join(args.dataset_dir, 'normal')
    os.makedirs(output_dir, exist_ok=True)

    if args.method == 'metric3d':
        get_normal_with_Metric3D(image_dir, output_dir)
    elif args.method == 'dsine':
        get_normal_with_DSINE(image_dir, output_dir)
    elif args.method == 'omnidata':
        get_normal_with_omnidata(image_dir, output_dir)
    else:
        print('Invalid method for monocular normal estimation')
        exit()