from segment_anything import build_sam, SamPredictor 
import os
import cv2
import numpy as np
from collections import defaultdict

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'



predictor = SamPredictor(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
_ = predictor.model.to(device='cuda')
# image = cv2.imread('/home/hadoop-vacv/yanfeng/data/dancetrack/train/dancetrack0001/img1/00000109.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
# predictor.set_image(image)

# bbox = np.array([0,0,100,100], dtype=np.int32)

# masks, _, _ = predictor.predict(box=bbox)

# masks

input_path = '/home/hadoop-vacv/yanfeng/data/dancetrack/val/dancetrack0004'
targets = [f for f in os.listdir(os.path.join(input_path, 'img1')) if not os.path.isdir(os.path.join(input_path, 'img1', f))]
targets = [os.path.join(input_path, 'img1', f) for f in targets]
targets.sort()

bboxes_all = defaultdict(list)
gt_path = os.path.join(input_path, 'gt', 'gt.txt')
# gt_path = os.path.join('/home/hadoop-vacv/yanfeng/project/MOTRv2/MOTRv3/exps/motrv2ch_uni5cost6g/run2/tracker0', 'dancetrack0004.txt')
for l in open(gt_path):
    t, i, *xywh, mark, label = l.strip().split(',')[:8]
    t, i, mark, label = map(int, (t, i, mark, label))
    if mark == 0:
        continue
    if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
        continue
    else:
        crowd = False
    x, y, w, h = map(int, map(float, (xywh)))
    bboxes_all[t].append([x, y, x+w, y+h, i])

fps = 25
size = (1920, 1080) 
videowriter = cv2.VideoWriter('tmp.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)


for t in targets:
    print(f"Processing '{t}'...")
    image = cv2.imread(t)
    if image is None:
        print(f"Could not load '{t}' as an image, skipping...")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks_all = []
    bboxes = np.array(bboxes_all[int(os.path.basename(t)[:-4])])
    # predictor.set_image(image)
    # masks, _, _ = predictor.predict(box=bboxes[:, :4])
    predictor.set_image(image)
    
    for bbox in bboxes:
        masks, iou_predictions, low_res_masks = predictor.predict(box=bbox[:4])
        index_max = iou_predictions.argsort()[0]
        masks = np.concatenate([masks[index_max:(index_max+1)], masks[index_max:(index_max+1)], masks[index_max:(index_max+1)]], axis=0)
        masks = masks.astype(np.int32)*np.array(colors(bbox[4]))[:, None, None]
        masks_all.append(masks)
    
    predictor.reset_image()
    
    if len(masks_all):
        masks_sum = masks_all[0].copy()
        for m in masks_all[1:]:
            masks_sum += m
    else:
        masks_sum = np.zeros_like(img).transpose(2, 0, 1)

    img = image.copy()[..., ::-1]
    img = (img * 0.5 + (masks_sum.transpose(1,2,0) * 30) %128).astype(np.uint8)
    for bbox in bboxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), thickness=3)
    # cv2.imwrite('tmp.jpg', img)
    
    videowriter.write(img)
    
videowriter.release()
        
        
    
    
    
