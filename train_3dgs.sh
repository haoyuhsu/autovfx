##### Example Usage #####

SCENE_NAME=$1

if [ -z "$2" ]; then
  LAMBDA_NORMAL=0.0
else
  LAMBDA_NORMAL=$2
fi

if [ -z "$3" ]; then
  LAMBDA_ANISOTROPIC=0.1
else
  LAMBDA_ANISOTROPIC=$3
fi

if [ -z "$4" ]; then
  LAMBDA_PSEUDO_NORMAL=0.01
else
  LAMBDA_PSEUDO_NORMAL=$4
fi

LAMBDA_ALPHA=0.0

if [ -z "$5" ]; then
  SIZE_THRESHOLD=20
else
  SIZE_THRESHOLD=$5
fi

if [ -z "$6" ]; then
  OPACITY=0.005
else
  OPACITY=$6
fi


# Step 1: training original gaussian splatting model
python sugar/gaussian_splatting/train.py \
    -s ./datasets/${SCENE_NAME} \
    -m ./output/${SCENE_NAME}/ \
    --iterations 15000 \
    --max_img_size 1920 \
    --lambda_normal ${LAMBDA_NORMAL} \
    --lambda_pseudo_normal ${LAMBDA_PSEUDO_NORMAL} \
    --lambda_alpha ${LAMBDA_ALPHA} \
    --lambda_anisotropic ${LAMBDA_ANISOTROPIC} \
    --data_device cpu \
    --scene_sdf_mesh_path ./datasets/${SCENE_NAME}/mesh/mesh.obj \
    --init_strategy hybrid \
    --size_threshold ${SIZE_THRESHOLD} \
    --min_opacity ${OPACITY}


# Step 2: training coarse sugar model
python sugar/train_coarse_density.py \
    -s ./datasets/${SCENE_NAME} \
    -c ./output/${SCENE_NAME}/ \
    -o ./output/${SCENE_NAME}/coarse/ \
    -i 15000 \
    --max_img_size 1920 \
    --lambda_normal ${LAMBDA_NORMAL} \
    --lambda_pseudo_normal ${LAMBDA_PSEUDO_NORMAL} \
    --lambda_alpha ${LAMBDA_ALPHA} \
    --lambda_anisotropic ${LAMBDA_ANISOTROPIC} \
    --iterations 22000
