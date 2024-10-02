import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image

import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
import sys
sys.path.append(os.path.join(cur_dir, 'Tag2Text'))
from Tag2Text.models import tag2text
from Tag2Text import inference_ram
import torchvision.transforms as TS

# ChatGPT or nltk is required when using tags_chineses
# import openai
# import nltk

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'tags_chinese': tags_chinese,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f, indent=4)


def run_ram_segmentation(img_path, output_dir, device='cuda'):
    """
    Wrapper function to run the whole pipeline
    """
    running_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_path = os.path.join(running_dir, '../saves/GroundingDINO_SwinT_OGC.py')
    ram_ckpt_path = os.path.join(running_dir, '../saves/ram_swin_large_14m.pth')
    grounded_ckpt_path = os.path.join(running_dir, '../saves/groundingdino_swint_ogc.pth')
    sam_ckpt_path = os.path.join(running_dir, '../saves/sam_vit_h_4b8939.pth')

    box_threshold = 0.25
    text_threshold = 0.2
    iou_threshold = 0.5

    os.makedirs(output_dir, exist_ok=True)

    image_pil, image = load_image(img_path)
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # initialize Grounding DINO model
    model = load_model(cfg_path, grounded_ckpt_path, device=device)

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    ram_model = tag2text.ram(pretrained=ram_ckpt_path,
                                        image_size=384,
                                        vit='swin_l')
    
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()
    ram_model = ram_model.to(device)
    raw_image = image_pil.resize(
                    (384, 384))
    raw_image  = transform(raw_image).unsqueeze(0).to(device)
    res = inference_ram.inference(raw_image , ram_model)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    tags=res[0].replace(' |', ',')
    tags_chinese=res[1].replace(' |', ',')
    print("Image Tags: ", res[0])

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image, tags, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_ckpt_path).to(device))
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    
    # draw all output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "automatic_label_output.jpg"), 
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    # draw mask image
    save_mask_data(output_dir, tags_chinese, masks, boxes_filt, pred_phrases)

    ##### save mask and bbox images for each object instance #####
    for idx, mask in enumerate(masks):

        output_folder_name = os.path.join(output_dir, str(idx + 1))  # 0 for background
        os.makedirs(output_folder_name, exist_ok=True)

        # save binary mask image for each object
        mask_np = mask.cpu().numpy()[0]
        mask_img = (mask_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder_name, f"mask.jpg"), mask_img)

        # draw bounding box on the raw image
        demo_image = image.copy()
        random_color = np.random.random(3)
        demo_image[mask_np] = demo_image[mask_np] * 0.4 + random_color.reshape(1, 1, 3) * 255 * 0.6
        x0, y0, x1, y1 = boxes_filt[idx].numpy().astype(int)
        cv2.rectangle(demo_image, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_folder_name, f"bbox.jpg"), cv2.cvtColor(demo_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    image_path = args.input_image
    output_dir = args.output_dir

    run_ram_segmentation(image_path, output_dir, args.device)