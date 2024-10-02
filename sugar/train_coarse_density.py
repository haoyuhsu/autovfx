import argparse
from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_density import coarse_training_with_density_regularization


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to optimize a coarse SuGaR model, i.e. a 3D Gaussian Splatting model with surface regularization losses in density space.')
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='path to the scene data to use.')
    parser.add_argument('-o', '--output_dir',
                        type=str, default=None, 
                        help='path to the output directory.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    
    parser.add_argument('-e', '--estimation_factor', type=float, default=0.2, help='factor to multiply the estimation loss by.')
    parser.add_argument('-n', '--normal_factor', type=float, default=0.2, help='factor to multiply the normal loss by.')
    
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')

    parser.add_argument('--lambda_depth', type=float, default=0.0, help='Weight for depth loss.')
    parser.add_argument('--lambda_normal', type=float, default=0.0, help='Weight for normal loss.')
    parser.add_argument('--lambda_pseudo_normal', type=float, default=0.0, help='Weight for pseudo normal loss. (0.01 in GaussianShader)')
    parser.add_argument('--lambda_alpha', type=float, default=0.0, help='Weight for opacity loss. (0.001 in GaussianShader)')
    parser.add_argument('--lambda_anisotropic', type=float, default=0.0, help='Weight for anisotropic loss.')

    parser.add_argument('--max_img_size', type=int, default=1920, help='Maximum image size for training. (Default: 1920)')
    parser.add_argument('--iterations', type=int, default=15_000, help='Number of iterations to train. (Default: 15_000)')

    args = parser.parse_args()
    
    # Call function
    coarse_training_with_density_regularization(args)
    