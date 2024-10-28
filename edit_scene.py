#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# import bpy
from edit_utils import *
from gpt.code_gen import setup_LMP
import os
from opt import ROOT_DIR


def run_scene_editing(args):

    # Generate the code and execute
    lmps = setup_LMP(args.waymo_scene, args.reference_image_path)
    edit_lmp = lmps['plan_ui']
    edit_lmp(args.edit_text)


if __name__ == "__main__":
    # Set up command line argument parser
    from opt import get_opts
    hparams = get_opts()
    # print(hparams)
    # FOR DEBUGGING: write the edit text to a global .txt file (without overwriting)
    with open(os.path.join(ROOT_DIR, 'logs_lmp_code_gen.txt'), "a") as f:
        f.write(f"===== {hparams.edit_text} =====\n\n")
    run_scene_editing(hparams)