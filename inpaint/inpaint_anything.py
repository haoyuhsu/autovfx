import torch
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from .lama_inpaint import inpaint_img_with_lama
from .stable_diffusion_inpaint import fill_img_with_sd
from .utils import load_img_to_array, save_array_to_img, dilate_mask, erode_mask, \
    show_mask, show_points, get_clicked_point


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--text_prompt", type=str, default='',
        help="Text prompt (if None then perform standard inpainting)",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=70,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--erode_kernel_size", type=int, default=0,
        help="Erode kernel size. Default: None",
    )
    parser.add_argument(
        "--alpha_threshold", type=float, default=0.7,
        help="Threshold for alpha channel for inpainting. Default: 0.0",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use for inference. Default: cuda",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="/home/shenlong/Documents/maxhsu/gaussian-splatting/inpaint/lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default="/home/shenlong/Documents/maxhsu/gaussian-splatting/inpaint/ckpts/big-lama/",
        help="The path to the lama checkpoint.",
    )


def inpaint_img(img_path, text_prompt='', dilate_kernel_size=10, erode_kernel_size=0, alpha_threshold=0.7, device='cuda'):
    
    img = load_img_to_array(img_path)

    # masks: image with low alpha channel
    ALPHA_THRESHOLD = alpha_threshold
    masks = img[..., 3] < ALPHA_THRESHOLD * 255 # thresholding
    masks = masks.astype(np.uint8) * 255
    masks = masks[np.newaxis, ...]

    img = img[..., :3]

    # dilate or erode mask to avoid unmasked edge effect
    if erode_kernel_size is not None:
        masks = [erode_mask(mask, erode_kernel_size) for mask in masks]
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    assert len(masks) == 1, "Only one mask is supported for now."

    mask = masks[0]

    # visualize the segmentation results
    img_mask_p = img_path[:-4] + "_mask.png"
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
    plt.imshow(img)
    plt.axis('off')
    show_mask(plt.gca(), mask, random_color=False)
    plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
    plt.close()

    # inpaint the masked image
    img_inpainted_p = img_path[:-4] + "_inpaint.png"
    # Option 1: Inpaint with stable diffusion
    # img_inpainted = fill_img_with_sd(
    #     img, mask, text_prompt, device=device)
    # Option 2: Inpaint with lama
    lama_config = "/home/shenlong/Documents/maxhsu/gaussian-splatting/inpaint/lama/configs/prediction/default.yaml"
    lama_ckpt = "/home/shenlong/Documents/maxhsu/gaussian-splatting/inpaint/ckpts/big-lama/"
    img_inpainted = inpaint_img_with_lama(
        img, mask, lama_config, lama_ckpt, device=device)
    
    save_array_to_img(img_inpainted, img_inpainted_p)

    return img_inpainted_p


if __name__ == "__main__":
    """Example usage:
    python inpaint_anything.py \
        --input_img FA_demo/FA1_dog.png \
        --text_prompt "a teddy bear on a bench" \
        --dilate_kernel_size 15 \
        --erode_kernel_size 15 \
        --output_dir ./results \
        --alpha_threshold 0.1
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inpaint_img(args.input_img, args.text_prompt, args.dilate_kernel_size, args.erode_kernel_size, args.alpha_threshold, device)

