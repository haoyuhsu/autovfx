import os
from os import path
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_config, get_model
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import process_frame_with_text as process_frame

from tqdm import tqdm
import json

from demo_utils import merge_instances

def run_deva(img_path: str, output_path: str, text: str, DINO_THRESHOLD: float = 0.7):
    """
    Run DEVA on a video or a directory of images

    Inputs:
    img_path: path to the video/images directory
    output_path: path to the output directory
    text: the text to be prompted

    Outputs:
    None
    """
    parser = ArgumentParser()

    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)

    # for action in parser._actions:
    #     print(f"Argument: {action.dest}, Type: {action.type}, Default: {action.default}")

    cfg, args = get_config(parser)

    running_dir = os.path.dirname(os.path.realpath(__file__))
    cfg['model'] = os.path.join(running_dir, 'saves/DEVA-propagation.pth')
    cfg['GROUNDING_DINO_CONFIG_PATH'] = os.path.join(running_dir, 'saves/GroundingDINO_SwinT_OGC.py')
    cfg['GROUNDING_DINO_CHECKPOINT_PATH'] = os.path.join(running_dir, 'saves/groundingdino_swint_ogc.pth')
    cfg['SAM_CHECKPOINT_PATH'] = os.path.join(running_dir, 'saves/sam_vit_h_4b8939.pth')
    cfg['img_path'] = img_path
    cfg['output'] = os.path.join(output_path, '_'.join(text.split(' ')))
    cfg['chunk_size'] = 4
    cfg['amp'] = True
    cfg['temporal_setting'] = 'semionline'
    cfg['DINO_THRESHOLD'] = DINO_THRESHOLD   # 0.7 works best, 0.45 for some hard cases
    cfg['prompt'] = text

    deva_model = get_model(cfg)
    gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')

    # get data
    video_reader = SimpleVideoReader(cfg['img_path'])
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
    out_path = cfg['output']

    # Start eval
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    cfg['enable_long_term_count_usage'] = (
        cfg['enable_long_term']
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
             cfg['num_prototypes']) >= cfg['max_long_term_elements'])

    print('Configuration:', cfg)

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)

    with torch.cuda.amp.autocast(enabled=cfg['amp']):
        for ti, (frame, im_path) in enumerate(tqdm(loader)):
            process_frame(deva, gd_model, sam_model, im_path, result_saver, ti, image_np=frame)
        flush_buffer(deva, result_saver)
    result_saver.end()

    # save this as a video-level json
    with open(path.join(out_path, 'pred.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)  # prettier json

    # merge instances
    merge_instances(out_path)


if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)

    # for id2rgb
    np.random.seed(42)
    # """
    # Arguments loading
    # """
    # parser = ArgumentParser()

    # add_common_eval_args(parser)
    # add_ext_eval_args(parser)
    # add_text_default_args(parser)
    # deva_model, cfg, args = get_model_and_config(parser)
    # gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')
    # """
    # Temporal setting
    # """
    # cfg['temporal_setting'] = args.temporal_setting.lower()
    # assert cfg['temporal_setting'] in ['semionline', 'online']

    # # get data
    # video_reader = SimpleVideoReader(cfg['img_path'])
    # loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
    # out_path = cfg['output']

    # # Start eval
    # vid_length = len(loader)
    # # no need to count usage for LT if the video is not that long anyway
    # cfg['enable_long_term_count_usage'] = (
    #     cfg['enable_long_term']
    #     and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
    #          cfg['num_prototypes']) >= cfg['max_long_term_elements'])

    # print('Configuration:', cfg)

    # deva = DEVAInferenceCore(deva_model, config=cfg)
    # deva.next_voting_frame = cfg['num_voting_frames'] - 1
    # deva.enabled_long_id()
    # result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)

    # with torch.cuda.amp.autocast(enabled=cfg['amp']):
    #     for ti, (frame, im_path) in enumerate(tqdm(loader)):
    #         process_frame(deva, gd_model, sam_model, im_path, result_saver, ti, image_np=frame)
    #     flush_buffer(deva, result_saver)
    # result_saver.end()

    # # save this as a video-level json
    # with open(path.join(out_path, 'pred.json'), 'w') as f:
    #     json.dump(result_saver.video_json, f, indent=4)  # prettier json

    parser = ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--text_prompt', type=str)
    args = parser.parse_args()

    text_prompt = args.text_prompt

    run_deva( \
        img_path=args.img_path, \
        output_path=args.output_path, \
        text=text_prompt)