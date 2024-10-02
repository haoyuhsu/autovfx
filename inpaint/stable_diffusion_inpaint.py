import os
import sys
import glob
import argparse
import torch
import numpy as np
import PIL.Image as Image
from pathlib import Path
from diffusers import AutoPipelineForInpainting
from .utils.mask_processing import crop_for_filling_pre, crop_for_filling_post


def fill_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        device="cuda"
):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    # text_prompt = text_prompt if text_prompt else "Fill the missing part."
    img_crop, mask_crop = crop_for_filling_pre(img, mask, 1024)
    img_crop_filled = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_crop),
        mask_image=Image.fromarray(mask_crop)
    ).images[0]
    img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled), 1024)
    return img_filled


