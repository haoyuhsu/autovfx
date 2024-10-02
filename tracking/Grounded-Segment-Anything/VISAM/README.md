# MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors


This fork from https://github.com/megvii-research/MOTRv2 [MOTRv2](https://arxiv.org/abs/2211.09791), and after we will release our code CO-MOT.

## Main Results

### DanceTrack

| **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|   69.9   |   83.0   |   59.0   |   91.9   |   71.7   | [model](https://drive.google.com/file/d/1EA4lndu2yQcVgBKR09KfMe5efbf631Th/view?usp=share_link) |

### Visualization

<!-- |OC-SORT|MOTRv2| -->
|VISAM|
|![](https://raw.githubusercontent.com/BingfengYan/MOTSAM/main/visam.gif)|


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [MOTR](https://github.com/megvii-research/MOTR).

### Requirements
* Install pytorch using conda (optional)

    ```bash
    conda create -n motrv2 python=3.9
    conda activate motrv2
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
    ```
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Dataset preparation

1. Download YOLOX detection from [here](https://drive.google.com/file/d/1cdhtztG4dbj7vzWSVSehLL6s0oPalEJo/view?usp=share_link).
2. Please download [DanceTrack](https://dancetrack.github.io/) and [CrowdHuman](https://www.crowdhuman.org/) and unzip them as follows:

```
/data/Dataset/mot
├── crowdhuman
│   ├── annotation_train.odgt
│   ├── annotation_trainval.odgt
│   ├── annotation_val.odgt
│   └── Images
├── DanceTrack
│   ├── test
│   ├── train
│   └── val
├── det_db_motrv2.json
```

You may use the following command for generating crowdhuman trainval annotation:

```bash
cat annotation_train.odgt annotation_val.odgt > annotation_trainval.odgt
```

### Training

You may download the coco pretrained weight from [Deformable DETR (+ iterative bounding box refinement)](https://github.com/fundamentalvision/Deformable-DETR#:~:text=config%0Alog-,model,-%2B%2B%20two%2Dstage%20Deformable), and modify the `--pretrained` argument to the path of the weight. Then training MOTR on 8 GPUs as following:

```bash 
./tools/train.sh configs/motrv2.args
```

### Inference on DanceTrack Test Set

1. Download SAM weigth fro [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
2. run
```bash
# run a simple inference on our pretrained weights
./tools/simple_inference.sh ./motrv2_dancetrack.pth

# Or evaluate an experiment run
# ./tools/eval.sh exps/motrv2/run1

# then zip the results
zip motrv2.zip tracker/ -r
```

if you want run on yourself data, please get detection results from [ByteTrackInference](https://github.com/zyayoung/ByteTrackInference) firstly.


## Acknowledgements

- [MOTR](https://github.com/megvii-research/MOTR)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [OC-SORT](https://github.com/noahcao/OC_SORT)
- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [BDD100K](https://github.com/bdd100k/bdd100k)
