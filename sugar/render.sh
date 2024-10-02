# Note: render.py not completed yet

SCENE_NAME=$1
LAMBDA_NORMAL=$2
POISSON_DEPTH=$3
VERTICES_DENSITY_QUANTILE=$4
NORMAL_CONSISTENCY_FACTOR=$5

FOLDER_ROOT_NAME=${SCENE_NAME}_lnorm${LAMBDA_NORMAL}_poisson${POISSON_DEPTH}_quantile${VERTICES_DENSITY_QUANTILE}_normalconsistency${NORMAL_CONSISTENCY_FACTOR}

python render.py \
    -s ../datasets/colmap/${SCENE_NAME} \
    -c ./output/${FOLDER_ROOT_NAME}/ \
    -i 7000 \
    --coarse_model_path ./output/${FOLDER_ROOT_NAME}/coarse/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
    --coarse_mesh_path ./output/${FOLDER_ROOT_NAME}/coarse-mesh/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000.ply \
    --refined_model_path ./output/${FOLDER_ROOT_NAME}/refined/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1/15000.pt \
    -o ./output/${FOLDER_ROOT_NAME}/rendering/