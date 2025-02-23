# AutoVFX

AutoVFX: Physically Realistic Video Editing from Natural Language Instructions.

### [Project Page](https://haoyuhsu.github.io/autovfx-website/) | [Paper](https://arxiv.org/abs/2411.02394)

[Hao-Yu Hsu<sup>1</sup>](https://haoyuhsu.github.io/), [Chih-Hao Lin<sup>1</sup>](https://chih-hao-lin.github.io/), [Albert J. Zhai<sup>1</sup>](https://ajzhai.github.io/), [Hongchi Xia<sup>1</sup>](https://xiahongchi.github.io/), [Shenlong Wang<sup>1</sup>](https://shenlong.web.illinois.edu/)

<sup>1</sup>University of Illinois at Urbana-Champaign

International Conference on 3D Vision (3DV), 2025

![teasor](docs/images/teasor.jpg)

## :dart: Progress
- [x] Environment Setup
- [x] Pretrained checkpoints, data, and software preparation
- [x] Simulation example on Garden scene
- [x] Details of pose extraction (SfM) and pose alignment
- [x] Details of training 3DGS
- [x] Details of surface reconstruction
- [x] Details of estimating relative scene scale
- [x] Code for sampling custom camera trajectory
- [ ] Local gradio app demo

## :clapper: Prerequisites
The code has been tested on:
- **OS**: Ubuntu 22.04.5 LTS
- **GPU**: NVIDIA GeForce RTX 4090
- **Driver Version**: 550
- **CUDA Version**: 12.4
- **nvcc**: 11.8

## :clapper: Environment Setup

- Create environment:
```bash
git clone https://github.com/haoyuhsu/autovfx.git
cd autovfx/
conda create -n autovfx python=3.10
conda activate autovfx
```

- Install PyTorch & cudatoolkit:
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# (Optional) To build the necessary CUDA extensions, cuda-toolkit is required.
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

- Install Gaussian Splatting submodules:
```bash
cd sugar/gaussian_splatting/
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

- Install segmentation & tracking modules:
```bash
# Tracking-with-DEVA
cd ../../tracking
pip install -e .

# Grounded-SAM
git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

# RAM & Tag2Text
git submodule init
git submodule update

git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

- Install inpainting modules:
```bash
# LaMa
cd ../../inpaint/lama
pip install -r requirements.txt
```

- Install lighting estimation modules:
```bash
# DiffusionLight
cd ../../lighting/diffusionlight
pip install -r requirements.txt
```

- Install other required packages:
```bash
# Other packages
pip install openai objaverse kornia wandb open3d plyfile imageio-ffmpeg einops e3nn pygltflib lpips scann geffnet open_clip_torch sentence-transformers==2.7.0 geffnet mmcv vedo

# PyTorch3D (try one of the commands)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
conda install pytorch3d -c pytorch3d

# Trimesh with speedup packages
pip install trimesh==4.3.2
pip install Rtree==1.2.0
conda install -c conda-forge embree=2.17.7
conda install -c conda-forge pyembree

# (Optional) COLMAP if not build from source
conda install conda-forge::colmap

cd ../..
```


## :clapper: Download pretrained checkpoints, required data and Blender

### Tracking modules
We use [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) for open-vocabulary video segmentation. 
```bash
cd tracking
bash download_models.sh
```

### Inpainting modules
We use [LaMa](https://github.com/advimman/lama) to inpaint the unseen region.
```bash
cd inpaint && mkdir ckpts
wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && unzip big-lama.zip -d ckpts
rm big-lama.zip
```

### Asset retrieval data
We use [CLIP](https://github.com/mlfoundations/open_clip) & [SBERT](https://sbert.net) features to annotate assets in [Objaverse](https://github.com/allenai/objaverse-xl), and we use SBERT features to annotate PBR materials in [PolyHaven](https://polyhaven.com). The preprocessed embeddings of both Objaverse 3D assets and PolyHaven PBR materials need to be downloaded. 

```bash
cd retrieval
# download processed embeddings
gdown --folder https://drive.google.com/drive/folders/1Lw87MstzbQgEX0iacTm9GpLYK2UE3gNm
# download PolyHaven PBR-materials
gdown https://drive.google.com/uc?id=1adZo_FPyLj7pFofNJfxSbnAv_EaJEV75
unzip polyhaven.zip && rm polyhaven.zip
```

### Download Blender 
We tested with [Blender 3.6.11](https://www.blender.org/download/release/Blender3.6/blender-3.6.11-linux-x64.tar.xz). Note that Blender 3+ requires Ubuntu version >= 20.04.

```bash
cd third_parties/Blender
wget https://download.blender.org/release//Blender3.6/blender-3.6.11-linux-x64.tar.xz
tar -xvf blender-3.6.11-linux-x64.tar.xz
rm blender-3.6.11-linux-x64.tar.xz
```

## :clapper: Dataset Preparation

Please download the preprocessed dataset of Garden scene from [here](https://drive.google.com/drive/folders/1eRdSAqDloGXk04JK60v3io6GHWdomy2N?usp=sharing) for quick demo.
The expected folder structure of the dataset will be:
```
├── datasets
│   | <your scene name>
│     ├── custom_camera_path   # optional for free-viewpoint rendering
│         ├── transforms_001.json
|         ├── ...
│     ├── images
|         ├── 00000.png
|         ├── 00001.png
|         ├── 00002.png
|         ├── ...
│     ├── mesh
|         ├── material_0.png
|         ├── mesh.mtl
|         ├── mesh.obj
│     ├── emitter_mesh.obj     # optional for indoor scenes
│     ├── normal
|         ├── 00000_normal.png
|         ├── 00001_normal.png
|         ├── 00002_normal.png
|         ├── ...
|     ├── sparse
|         | 0
|           ├── cameras.bin
|           ├── images.bin
|           ├── points3D.bin
|     ├── transforms.json
```
### Custom dataest
For your custom dataset, please follow these steps:
- Create a folder and put your images under `images`. The folder will be like this:
```
├── datasets
│   | <your scene name>
│     ├── images
|         ├── 00000.png
|         ├── 00001.png
|         ├── 00002.png
|         ├── ...
```
- Estimate normal maps for the usage of both pose alignment and normal regularization during 3DGS and BakedSDF training. Currently, we support three types of methods for monocular normal estimation, which are [Metric3D](https://github.com/YvanYin/Metric3D), [DSINE](https://github.com/baegwangbin/DSINE). and [Omnidata](https://github.com/EPFL-VILAB/omnidata). Empirically, the quality of normal estimation is ranked as Metric3D > DSINE > Omnidata.
```bash
python dataset_utils/get_mono_normal.py \
    --dataset_dir ./datasets/<your scene name> \
    --method metric3d     # 'metric3d', 'dsine', 'omnidata'
```
- Perform pose extraction using COLMAP, followed by pose alignment to set the up direction of the scene to `(0,0,1)`. Specify a text prompt for the most obvious flat surfaces in the scene, such as `ground`, `floor` or `table`.
```bash
python dataset_utils/colmap_runner.py \
    --dataset_dir ./datasets/<your scene name> \
    --text_prompt ground
```
- For details on surface mesh extraction, please refer to the ***Estimate Scene Properties*** section.
- All cameras are in camera-to-world coordinate with OpenCV format (x: right, y: down, z: front). Please refer to [this tutorial](https://github.com/google-research/multinerf?tab=readme-ov-file#making-your-own-loader-by-implementing-_load_renderings) on conversion between OpenCV and OpenGL camera format.
- We support sampling custom camera poses along a circular trajectory, please adjust the sampled parameters in `dataset_utils/sample_custom_traj.py` and run:
```bash
python dataset_utils/sample_custom_traj.py \
    --dataset_dir ./datasets/<your scene name> \
    --traj_name <your trajectory name> \
    --vis_traj
```

## :clapper:  Estimate Scene Properties

### Surface mesh extraction
We use [BakedSDF](https://bakedsdf.github.io/) implemented in [SDFStudio](https://github.com/zhihao-lin/sdfstudio) for surface reconstruction. Please make sure to use [our custom SDFStudio](https://github.com/zhihao-lin/sdfstudio) for reproducibility. We recommend to create an extra environemnt for this part since CUDA 11.3 has been tested on this repo.

#### BakedSDF training
```bash
# Example command
ns-train bakedsdf-mlp --vis wandb \
    --output-dir outputs/<scene name> --experiment-name <experiment name> \
    --trainer.steps-per-save 1000 \
    --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 5000 \
    --pipeline.datamanager.train-num-rays-per-batch 2048 \
    --pipeline.datamanager.eval-num-rays-per-batch 512 \
    --pipeline.model.sdf-field.inside-outside False \
    --pipeline.model.background-model none \
    --pipeline.model.near-plane 0.001 --pipeline.model.far-plane 6.0 \
    --machine.num-gpus 1 \
    --pipeline.model.mono-normal-loss-mult 0.1 \
    panoptic-data \
    --data <path to your dataset> \
    --panoptic_data False --mono_normal_data True --panoptic_segment False \
    --orientation-method none --center-poses False --auto-scale-poses False \
```
Generally, a decent surface mesh can be obtained with the command above. However, there are several hyperparameters that you should be careful to set appropriately.
- For fully captured indoor scenes, such as those in [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), set `--pipeline.model.sdf-field.inside-outside` to `True`.
- For outdoor scenes with distant backgrounds, such as those in the [Tanks & Temples](https://www.tanksandtemples.org/), set `--pipeline.model.background-model` to `mlp`.
- Adjust `--pipeline.datamanager.train-num-rays-per-batch`, `--pipeline.datamanager.eval-num-rays-per-batch`, and `--pipeline.model.num-neus-samples-per-ray` if you encounter OOM (out-of-memory) errors during training.

#### Mesh extraction & Texture baking
```bash
scene=outputs/<scene name>/<experiment name>/bakedsdf-mlp/<timestamp>
# Extract mesh
python scripts/extract_mesh.py --load-config $scene/config.yml \
    --output-path $scene/mesh.ply \
    --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
    --resolution 2048 --marching_cube_threshold 0.001 --create_visibility_mask True --simplify-mesh True

mkdir $scene/textured
# Bake texture
python scripts/texture.py \
    --load-config $scene/config.yml \
    --input-mesh-filename $scene/mesh-simplify.ply \
    --output-dir $scene/textured \
    --target_num_faces None
```
It is better not changing `bounding-box-min` and `bounding-box-max` since camera poses are already normalized within a unit cube in the pose alignment step.

### Training 3DGS & SuGaR
You could start training 3D gaussian splatting with one command.
```bash
bash train_3dgs.sh <your scene name>
```
Explanation of several hyperparameters used in `train_3dgs.sh`:
- Optimization parameters:
    - `lambda_normal`: loss between rendered normal and monocular normal prediction
    - `lambda_pseudo_normal`: loss between rendered normal and pseudo normal derived from rendered depth
    - `lambda_anisotropic`: regularize 3D gaussians shape to be isotropic
- Densification parameters:
    - consider adjust `size_threshold`and `min_opacity` if the Gaussians are floating excessively.
- Gaussians initialization parameters `--init_strategy`:
    - `colmap`: use a point cloud extracted from COLMAP for initialization
    - `ray_mesh`: use intersection points between camera rays from all training views and the scene mesh for initialization.
    - `hybrid`: combine both `colmap` and `ray_mesh` for initialization
    - **Ensure that `--scene_sdf_mesh_path` is specified when using `ray_mesh` or `hybrid`**

### Estimating relative scene scale
Use the following script to determine the relative scale between the current scene and a real-world scenario. Then, set the `--scene_scale` parameter to the estimated value during simulation.
```bash
python dataset_utils/estimate_scene_scale.py \ 
    --dataset_dir ./datasets/<your scene name> \
    --scene_mesh_path ./datasets/<your scene name>/mesh/mesh.obj \
    --anchor_frame_idx 0
```


## :clapper: Start Simulation

### Example demo for Garden scene
Please download the preprocessed Garden scene from [here](https://drive.google.com/drive/folders/1eRdSAqDloGXk04JK60v3io6GHWdomy2N?usp=sharing), and the pretrained 3DGS checkpoints and estimated scene properties from [here](https://drive.google.com/drive/folders/1KE8LSA_r-3f2LVlTLJ5k4SHENvbwdAfN?usp=sharing).

<!-- ***Need to update with .zip file for gdown*** -->
```bash
# If you encounter an error with gdown, please use the Google Drive link above to download the files.
mkdir datasets && cd datasets
gdown --folder https://drive.google.com/drive/folders/1eRdSAqDloGXk04JK60v3io6GHWdomy2N
cd ../
mkdir output && cd output
gdown --folder https://drive.google.com/drive/folders/1KE8LSA_r-3f2LVlTLJ5k4SHENvbwdAfN
```

- Text Prompt: *"Drop 5 basketballs on the table."*
```bash
export OPENAI_API_KEY=/your/openai_api_key/
export MESHY_API_KEY=/your/meshy_api_key/   # if you want to retrieve generated 3D assets

SCENE_NAME=garden_large
CUSTOM_TRAJ_NAME=transforms_001
SCENE_SCALE=2.67
BLENDER_CONFIG_NAME=blender_cfg_rigid_body_simulation

python edit_scene.py \
    --source_path datasets/${SCENE_NAME} \
    --model_path output/${SCENE_NAME}/ \
    --gaussians_ckpt_path output/${SCENE_NAME}/coarse/sugarcoarse_3Dgs15000_densityestim02_sdfnorm02/22000.pt \
    --custom_traj_name ${CUSTOM_TRAJ_NAME} \
    --anchor_frame_idx 0 \
    --scene_scale ${SCENE_SCALE} \
    --edit_text "Drop 5 basketballs on the table." \
    --scene_mesh_path datasets/${SCENE_NAME}/mesh/mesh.obj \
    --blender_config_name ${BLENDER_CONFIG_NAME}.json \
    --blender_output_dir_name ${BLENDER_CONFIG_NAME} \
    --render_type MULTI_VIEW \
    --deva_dino_threshold 0.45 \
    --is_uv_mesh
```
All the parameters are listed in the `opt.py`.

<details>
<summary><span style="font-weight: bold;">Arguments used in opt.py</span></summary>

  ##### --source_path
  Path to the dataset directory.
  ##### --model_path
  Path to the output directory.
  ##### --gaussians_ckpt_path
  Path to the Gaussian model checkpoint (.pt for SuGaR, .ply for vanilla 3DGS).
  ##### --scene_mesh_path
  Path to the reconstructed scene mesh (.ply or .obj).
  ##### --emitter_mesh_path
  Path to the emitter mesh for indoor lighting (.obj) (only used for indoor scenes).

  ##### --edit_text
  Editing instructions.
  ##### --custom_traj_name
  Filename of custom trajectory (default: training cameras).
  ##### --anchor_frame_idx
  Index of the frame used for single-view simulation (default: 0).
  ##### --scene_scale
  Relative scale of the scene. If an object in the scene is 1 unit tall but is known to be 0.7 meters in the real world, the scene_scale is 0.7. This parameter is crucial for ensuring accurate size correspondence and realistic simulation or rendering.
  ##### --blender_output_dir_name and --blender_config_name
  Name of Blender output folder and Blender .json config.
  ##### --render_type
  Choose 'MULTI_VIEW' to render frames along the entire camera trajectory, or 'SINGLE_VIEW' for static rendering from a single camera position (i.e., anchor_frame_idx).
  ##### --num_frames
  Specifies the number of frames to simulate and render (only used when '--render_type=SINGLE_VIEW').

  ##### --is_uv_mesh
  Enable this option if the scene mesh have UV textures.
  ##### --is_indoor_scene
  Enable this option if the scene is an indoor scene.
  ##### --waymo_scene
  Enable this option to simulate on Waymo road scenes. A different prompt for GPT-4 is used to fulfill road scene simulations, similar to ChatSim.

  ##### --deva_dino_threshold
  Increase this threshold to reduce excessive object detection. (0.7 is optimal, but lower to 0.45 for hard-to-detect cases).

</details>


## :clapper: Citation
If you find this paper and repository useful for your research, please consider citing: 
```bibtex
@article{hsu2024autovfx,
    title={AutoVFX: Physically Realistic Video Editing from Natural Language Instructions},
    author={Hsu, Hao-Yu and Lin, Zhi-Hao and Zhai, Albert and Xia, Hongchi and Wang, Shenlong},
    journal={arXiv preprint arXiv:2411.02394},
    year={2024}
}
```

## :clapper: Acknowledgement
This project is supported by the Intel AI SRS gift, Meta research grant, the IBM IIDAI Grant and NSF Awards #2331878, #2340254, #2312102, #2414227, and #2404385. Hao-Yu Hsu is supported by Siebel Scholarship. We greatly appreciate the NCSA for providing computing resources. We thank Derek Hoiem, Sarita Adve, Benjamin Ummenhofer, Kai Yuan, Micheal Paulitsch, Katelyn Gao, Quentin Leboutet for helpful discussions.

Our codebase are built based on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), [SuGaR](https://github.com/Anttwo/SuGaR), [SDFStudio](https://github.com/autonomousvision/sdfstudio), [DiffusionLight](https://github.com/DiffusionLight/DiffusionLight), [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA), [Objaverse](https://github.com/allenai/objaverse-xl), and the most important [Blender](https://github.com/blender/blender). Thanks for open-sourcing!.
