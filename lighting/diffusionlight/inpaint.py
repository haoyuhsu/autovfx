# inpaint the ball on an image
# this one is design for general image that does not require special location to place 


import torch
import argparse
import numpy as np
import torch.distributed as dist
import os
from PIL import Image
from tqdm.auto import tqdm
import json

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from relighting.inpainter import BallInpainter

from relighting.mask_utils import MaskGenerator
from relighting.ball_processor import (
    get_ideal_normal_ball,
    crop_ball
)
from relighting.dataset import GeneralLoader
from relighting.utils import name2hash
import relighting.dist_utils as dist_util
import time

from relighting.image_processor import pil_square_image, pil_center_crop_and_resize


# cross import from inpaint_multi-illum.py
from relighting.argument import (
    SD_MODELS, 
    CONTROLNET_MODELS,
    VAE_MODELS
)


class AttrDict:
    def __init__(self, data):
        self.__dict__['_data'] = data

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self._data[key] = value

    def __delattr__(self, key):
        try:
            del self._data[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __repr__(self):
        return repr(self._data)


def create_default_args():
    args = {}
    # dataset parameter
    args["dataset"] = ""
    args["ball_size"] = 256
    args["ball_dilate"] = 20
    args["prompt"] = "a perfect mirrored reflective chrome ball sphere"
    args["prompt_dark"] = "a perfect black dark mirrored reflective chrome ball sphere"
    args["negative_prompt"] = "matte, diffuse, flat, dull"
    args["model_option"] = "sdxl"
    args["output_dir"] = ""
    args["img_height"] = 1024
    args["img_width"] = 1024
    args["seed"] = "auto"
    args["denoising_step"] = 30
    args["control_scale"] = 0.5

    args["use_controlnet"] = True
    args["force_square"] = True
    args["random_loader"] = True
    args["is_cpu"] = False
    args["offload"] = False
    args["limit_input"] = 0

    # LoRA stuff
    args["use_lora"] = True
    # get absolute path of current file
    current_folder = os.path.dirname(os.path.abspath(__file__))
    args["lora_path"] = os.path.join(current_folder, "models/ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500")
    args["lora_scale"] = 0.75

    # speed optimization stuff
    args["use_torch_compile"] = True

    # algorithm + iterative stuff
    args["algorithm"] = "iterative"

    args["agg_mode"] = "median"
    args["strength"] = 0.8
    args["num_iteration"] = 2
    args["ball_per_iteration"] = 30
    args["save_intermediate"] = True
    args["cache_dir"] = "./temp_inpaint_iterative"

    # pararelle processing
    args["idx"] = 0
    args["total"] = 1

    # for HDR stuff
    args["max_negative_ev"] = -5
    args["ev"] = "0,-2.5,-5"

    # NOTE: newly added functionality to return average ball
    args["use_avg_ball"] = False

    return AttrDict(args)


def get_ball_location(image_data, args):
    if 'boundary' in image_data:
        # support predefined boundary if need
        x = image_data["boundary"]["x"]
        y = image_data["boundary"]["y"]
        r = image_data["boundary"]["size"]
        
        # support ball dilation
        half_dilate = args.ball_dilate // 2

        # check if not left out-of-bound
        if x - half_dilate < 0: x += half_dilate
        if y - half_dilate < 0: y += half_dilate

        # check if not right out-of-bound
        if x + r + half_dilate > args.img_width: x -= half_dilate
        if y + r + half_dilate > args.img_height: y -= half_dilate   
            
    else:
        # we use top-left corner notation
        x, y, r = ((args.img_width // 2) - (args.ball_size // 2), (args.img_height // 2) - (args.ball_size // 2), args.ball_size)
    return x, y, r


def interpolate_embedding(pipe, args):
    print("interpolate embedding...")

    # get list of all EVs
    ev_list = [float(x) for x in args.ev.split(",")]
    interpolants = [ev / args.max_negative_ev for ev in ev_list]

    print("EV : ", ev_list)
    print("EV : ", interpolants)

    # calculate prompt embeddings
    prompt_normal = args.prompt
    prompt_dark = args.prompt_dark
    prompt_embeds_normal, _, pooled_prompt_embeds_normal, _ = pipe.pipeline.encode_prompt(prompt_normal)
    prompt_embeds_dark, _, pooled_prompt_embeds_dark, _ = pipe.pipeline.encode_prompt(prompt_dark)

    # interpolate embeddings
    interpolate_embeds = []
    for t in interpolants:
        int_prompt_embeds = prompt_embeds_normal + t * (prompt_embeds_dark - prompt_embeds_normal)
        int_pooled_prompt_embeds = pooled_prompt_embeds_normal + t * (pooled_prompt_embeds_dark - pooled_prompt_embeds_normal)

        interpolate_embeds.append((int_prompt_embeds, int_pooled_prompt_embeds))

    return dict(zip(ev_list, interpolate_embeds))


def inpaint_chrome_ball(input_image_path, output_dir, IMG_SIZE=1024, BALL_SIZE=256):
    # load arguments
    args = create_default_args()
    args.img_height = IMG_SIZE
    args.img_width = IMG_SIZE
    args.ball_size = BALL_SIZE
    args.output_dir = output_dir
    args.cache_dir = os.path.join(output_dir, "temp_inpaint_iterative")
        
    # get local rank
    if args.is_cpu:
        device = torch.device("cpu")
        torch_dtype = torch.float32
    else:
        device = dist_util.dev()
        torch_dtype = torch.float16
    
    # so, we need ball_dilate >= 16 (2*vae_scale_factor) to make our mask shape = (272, 272)
    assert args.ball_dilate % 2 == 0 # ball dilation should be symmetric
    
    # create controlnet pipeline 
    if args.model_option in ["sdxl", "sdxl_fast"] and args.use_controlnet:
        model, controlnet = SD_MODELS[args.model_option], CONTROLNET_MODELS[args.model_option]
        pipe = BallInpainter.from_sdxl(
            model=model, 
            controlnet=controlnet, 
            device=device,
            torch_dtype = torch_dtype,
            offload = args.offload
        )
    elif args.model_option in ["sdxl", "sdxl_fast"] and not args.use_controlnet:
        model = SD_MODELS[args.model_option]
        pipe = BallInpainter.from_sdxl(
            model=model,
            controlnet=None,
            device=device,
            torch_dtype = torch_dtype,
            offload = args.offload
        )
    elif args.use_controlnet:
        model, controlnet = SD_MODELS[args.model_option], CONTROLNET_MODELS[args.model_option]
        pipe = BallInpainter.from_sd(
            model=model,
            controlnet=controlnet,
            device=device,
            torch_dtype = torch_dtype,
            offload = args.offload
        )
    else:
        model = SD_MODELS[args.model_option]
        pipe = BallInpainter.from_sd(
            model=model,
            controlnet=None,
            device=device,
            torch_dtype = torch_dtype,
            offload = args.offload
        )

    
    if args.lora_scale > 0 and args.lora_path is None:
        raise ValueError("lora scale is not 0 but lora path is not set")
    
    if (args.lora_path is not None) and (args.use_lora):
        print(f"using lora path {args.lora_path}")
        print(f"using lora scale {args.lora_scale}")
        pipe.pipeline.load_lora_weights(args.lora_path)
        pipe.pipeline.fuse_lora(lora_scale=args.lora_scale) # fuse lora weight w' = w + \alpha \Delta w
        enabled_lora = True
    else:
        enabled_lora = False

    if args.use_torch_compile:
        try:
            print("compiling unet model")
            start_time = time.time()
            pipe.pipeline.unet = torch.compile(pipe.pipeline.unet, mode="reduce-overhead", fullgraph=True)
            print("Model compilation time: ", time.time() - start_time)
        except:
            pass
                
    # default height for sdxl is 1024, if not set, we set default height.
    if args.model_option == "sdxl" and args.img_height == 0 and args.img_width == 0:
        args.img_height = 1024
        args.img_width = 1024

    # interpolate embedding
    embedding_dict = interpolate_embedding(pipe, args)
    
    # prepare mask and normal ball
    mask_generator = MaskGenerator()
    normal_ball, mask_ball = get_ideal_normal_ball(size=args.ball_size+args.ball_dilate)
    _, mask_ball_for_crop = get_ideal_normal_ball(size=args.ball_size)
    
    
    # make output directory if not exist
    raw_output_dir = os.path.join(args.output_dir, "raw")
    control_output_dir = os.path.join(args.output_dir, "control")
    square_output_dir = os.path.join(args.output_dir, "square")
    os.makedirs(args.output_dir, exist_ok=True)    
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(control_output_dir, exist_ok=True)
    os.makedirs(square_output_dir, exist_ok=True)
    
    # create split seed
    # please DO NOT manual replace this line, use --seed option instead
    seeds = args.seed.split(",")

    resolution = (args.img_width, args.img_height)
    image = Image.open(input_image_path)
    if args.force_square:
        # image = pil_square_image(image, self.resolution)
        image = pil_center_crop_and_resize(image, resolution)
    else:
        image = image.resize(resolution)
    image_data = {
        "image": image,
        "path": input_image_path,
    }

    input_image = image_data["image"] 
    image_path = image_data["path"]

    input_image = input_image.convert("RGB")
    
    for ev, (prompt_embeds, pooled_prompt_embeds) in embedding_dict.items():
        # create output file name (we always use png to prevent quality loss)
        ev_str = str(ev).replace(".", "") if ev != 0 else "-00"
        outname = os.path.basename(image_path).split(".")[0] + f"_ev{ev_str}"

        # we use top-left corner notation (which is different from aj.aek's center point notation)
        x, y, r = get_ball_location(image_data, args)
        
        # create inpaint mask
        mask = mask_generator.generate_single(
            input_image, mask_ball, 
            x - (args.ball_dilate // 2),
            y - (args.ball_dilate // 2),
            r + args.ball_dilate
        )
            
        seeds = tqdm(seeds, desc="seeds") if len(seeds) > 10 else seeds   
            
        #replacely create image with differnt seed
        for seed in seeds:
            start_time = time.time()
            # set seed, if seed auto we use file name as seed
            if seed == "auto":
                filename = os.path.basename(image_path).split(".")[0]
                seed = name2hash(filename) 
                outpng = f"{outname}.png"
                cache_name = f"{outname}"
            else:
                seed = int(seed)
                outpng = f"{outname}_seed{seed}.png"
                cache_name = f"{outname}_seed{seed}"
            # skip if file exist, useful for resuming
            if os.path.exists(os.path.join(square_output_dir, outpng)):
                continue
            generator = torch.Generator().manual_seed(seed)
            kwargs = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                'negative_prompt': args.negative_prompt,
                'num_inference_steps': args.denoising_step,
                'generator': generator,
                'image': input_image,
                'mask_image': mask,
                'strength': 1.0,
                'current_seed': seed, # we still need seed in the pipeline!
                'controlnet_conditioning_scale': args.control_scale,
                'height': args.img_height,
                'width': args.img_width,
                'normal_ball': normal_ball,
                'mask_ball': mask_ball,
                'x': x,
                'y': y,
                'r': r,
            }
            
            if enabled_lora:
                kwargs["cross_attention_kwargs"] = {"scale": args.lora_scale}
            
            if args.algorithm == "normal":
                output_image = pipe.inpaint(**kwargs).images[0]
            elif args.algorithm == "iterative":
                # This is still buggy
                print("using inpainting iterative, this is going to take a while...")
                kwargs.update({
                    "strength": args.strength,
                    "num_iteration": args.num_iteration,
                    "ball_per_iteration": args.ball_per_iteration,
                    "agg_mode": args.agg_mode,
                    "save_intermediate": args.save_intermediate,
                    "cache_dir": os.path.join(args.cache_dir, cache_name),
                    "use_avg_ball": args.use_avg_ball,
                })
                output_image = pipe.inpaint_iterative(**kwargs)
            else:
                raise NotImplementedError(f"Unknown algorithm {args.algorithm}")
            
            
            square_image = output_image.crop((x, y, x+r, y+r))

            # return the most recent control_image for sanity check
            control_image = pipe.get_cache_control_image()
            if control_image is not None:
                control_image.save(os.path.join(control_output_dir, outpng))
            
            # save image 
            output_image.save(os.path.join(raw_output_dir, outpng))
            square_image.save(os.path.join(square_output_dir, outpng))

                          
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    inpaint_chrome_ball(args.input_image_path, args.output_dir)