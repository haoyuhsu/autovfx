# Utility function to retrieve 3D models from Objaverse (https://objaverse.allenai.org/objaverse-1.0/)
import objaverse
import random
import urllib.request
import os
import torch
from PIL import Image
import open_clip
from sentence_transformers import SentenceTransformer
import pickle
import scann
import time
import json
from multiprocessing import Pool
import numpy as np
from pygltflib import GLTF2


ALL_OBJS_CLIP_TEXT_EMBEDS_PATH = './retrieval/embeddings/all_data_clip_text_embeddings.pkl'
# ALL_OBJS_CLIP_IMAGE_EMBEDS_PATH = './retrieval/embeddings/all_data_clip_image_embeddings.pkl'
ALL_OBJS_SBERT_TEXT_EMBEDS_PATH = './retrieval/embeddings/all_data_sbert_embeddings.pkl'

ANIMATED_OBJS_SBERT_TEXT_EMBEDS_PATH = './retrieval/embeddings/animated_data_sbert_embeddings.pkl'

ID2IDX_DICT_PATH = './retrieval/embeddings/all_data_obj2idx.json'


def check_glb_animations(file_path):
    gltf = GLTF2().load(file_path)
    if gltf.animations:
        animation_name = [animation.name for animation in gltf.animations]
        print("{} animations found: ".format(len(animation_name)), animation_name)
        return True
    else:
        return False
    

def get_searcher(database):
    '''
    `num_leaves` should be set to the square root of number of total objects (rule of thumb)
    More details could be found in the SCANN documentation: https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
    '''
    print("searcher loading")
    start_time = time.time()
    searcher = scann.scann_ops_pybind.builder(database, 10, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()
    print("searcher loaded by --- %s seconds ---" % (time.time() - start_time))

    return searcher


def search(feat, searcher, final_num_neighbors=5):
    print("start searching")
    start_time = time.time()
    neighbors, distances = searcher.search(feat, final_num_neighbors=final_num_neighbors, leaves_to_search=150, pre_reorder_num_neighbors=250)
    print("searching finished by --- %s seconds ---" % (time.time() - start_time))
    return neighbors, distances


def init_model():
    # initialize CLIP
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
    clip_model = clip_model.cuda()

    # initialize sentence transformer
    sbert_model = SentenceTransformer('all-mpnet-base-v2')

    return clip_model, clip_tokenizer, clip_preprocess, sbert_model


def load_database(embeds_path):
    with open(embeds_path, 'rb') as f:
        embeds = pickle.load(f)
        database = embeds['database']
        uids = embeds['uids']
    return database, uids


def download_asset_from_objaverse(obj_ids, save_dir):
    obj_paths = objaverse.load_objects(obj_ids)
    # move the data to the save_dir
    local_paths = []
    for obj_path in obj_paths.values():
        local_path = os.path.join(save_dir, obj_path.split('/')[-1])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        os.system('mv {} {}'.format(obj_path, local_path))
        local_paths.append(local_path)
    return local_paths


def download_rendered_images_from_gobjaverse(args):
    obj_index, obj_id = args
    assets_imgs_save_dir = './_cache/assets_rendering_gobjaverse'
    end = 40 # hard-coded
    copy_items = ['.png'] # hard-coded
    oss_base_dir = os.path.join("https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/objaverse", str(obj_index), "campos_512_v4")
    local_path = os.path.join(assets_imgs_save_dir, obj_id)
    if os.path.exists(local_path):
        print("existing, skipping")
        return
    os.system("mkdir -p {}".format(local_path))
    for index in range(end):
        index = "{:05d}".format(index)
        for copy_item in copy_items:
            postfix = index + "/" + index + copy_item
            oss_full_dir = os.path.join(oss_base_dir, postfix)
            basename = os.path.basename(oss_full_dir)
            os.system("curl -o {} -C - {}".format(os.path.join(local_path, basename + '.tmp'), oss_full_dir))
            os.system("mv {} {}".format(os.path.join(local_path, basename + '.tmp'), os.path.join(local_path, basename)))


def compute_clip_image_features(clip_model, clip_preprocess, image_folder):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    with torch.no_grad():
        image = torch.stack([clip_preprocess(Image.open(image_path)) for image_path in image_paths]).cuda()
        image_feature_clip = clip_model.encode_image(image)
        image_feature_clip /= image_feature_clip.norm(dim=-1, keepdim=True)
        image_feature_clip = image_feature_clip.cpu().numpy()
    return image_feature_clip


def compute_clip_text_features(clip_model, clip_tokenizer, text):
    with torch.no_grad():
        text_feature_clip = clip_model.encode_text(clip_tokenizer(text).cuda())
        text_feature_clip /= text_feature_clip.norm(dim=-1, keepdim=True)
        text_feature_clip = text_feature_clip.cpu().numpy()[0]
    return text_feature_clip


def compute_sbert_text_features(sbert_model, text):
    with torch.no_grad():
        text_feature_sbert = sbert_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        text_feature_sbert /= text_feature_sbert.norm(dim=-1, keepdim=True)
        text_feature_sbert = text_feature_sbert.cpu().numpy()
    return text_feature_sbert


def retrieve_asset_from_objaverse(obj_name, is_animated=False, random_pick=True):
    clip_model, clip_tokenizer, clip_preprocess, sbert_model = init_model()

    # load the database
    if is_animated:
        database, uids = load_database(ANIMATED_OBJS_SBERT_TEXT_EMBEDS_PATH)
    else:
        database, uids = load_database(ALL_OBJS_SBERT_TEXT_EMBEDS_PATH)
    searcher = get_searcher(database)

    # load the id2idx dictionary
    with open(ID2IDX_DICT_PATH, 'r') as f:
        id2idx = json.load(f)

    # compute embeddings for the query
    query_text = obj_name
    sbert_text_feature = compute_sbert_text_features(sbert_model, query_text)

    SEARCH_TOP_K = 10
    DOWNLOAD_TOP_K = 5
    assert DOWNLOAD_TOP_K <= SEARCH_TOP_K, 'DOWNLOAD_TOP_K should be less than or equal to SEARCH_TOP_K'

    # search for the nearest neighbors
    COSINE_THRESHOLD = 0.6
    neighbors, dists = search(sbert_text_feature, searcher, SEARCH_TOP_K)
    picked_uids = [uids[idx] for idx, dist in zip(neighbors, dists) if dist >= COSINE_THRESHOLD]
    picked_dists = [dist for dist in dists if dist >= COSINE_THRESHOLD]

    if not picked_uids:
        # TODO: resort to generative 3D option
        raise ValueError('No corresponding objects found for the query:', obj_name)

    # if DOWNLOAD_TOP_K < SEARCH_TOP_K and DOWNLOAD_TOP_K < len(picked_uids):

    # download the pre-rendered images from GObjaverse
    assets_imgs_save_dir = './_cache/assets_rendering_gobjaverse'
    obj2idx = {obj_id: id2idx[obj_id] for obj_id in picked_uids}  # Assuming obj_id_list is defined somewhere
    data = [(obj_index, obj_id) for obj_id, obj_index in obj2idx.items()]
    N_THREADS = 5
    p = Pool(N_THREADS)
    p.map(download_rendered_images_from_gobjaverse, data)

    # compute CLIP embeddings of the rendered images for each object ID and rank them
    score_list = []
    clip_text_feature = compute_clip_text_features(clip_model, clip_tokenizer, query_text)
    for obj_id, obj_dist in zip(picked_uids, picked_dists):
        image_folder = os.path.join(assets_imgs_save_dir, obj_id)
        clip_image_feature = compute_clip_image_features(clip_model, clip_preprocess, image_folder)
        clip_score = np.mean(np.dot(clip_text_feature, clip_image_feature.T))
        # clip_score = np.max(np.dot(clip_text_feature, clip_image_feature.T))
        print('======= ID: {} ======='.format(obj_id))
        print('SBERT similarity score:', obj_dist)
        print('CLIP similarity score:', clip_score)
        score = obj_dist + clip_score
        score_list.append(score)

    sorted_idx = np.argsort(score_list)[::-1]
    picked_uids = [picked_uids[idx] for idx in sorted_idx]
    picked_uids = picked_uids[:DOWNLOAD_TOP_K]
    picked_dists = [score_list[idx] for idx in sorted_idx]  # update as total score
    picked_dists = picked_dists[:DOWNLOAD_TOP_K]

    # download the assets from Objaverse
    save_dir = './_cache/assets'
    obj_paths = download_asset_from_objaverse(picked_uids, save_dir)

    for obj_id, obj_dist in zip(picked_uids, picked_dists):
        print('Picked object ID:', obj_id, 'with score =', obj_dist)

    obj_info = {}
    if random_pick:
        # return a random object path
        idx = random.randint(0, len(picked_uids) - 1)
        obj_info['object_name'] = obj_name
        obj_info['object_id'] = picked_uids[idx]
        obj_info['object_path'] = obj_paths[idx]
    else:
        # return object path with highest score
        obj_info['object_name'] = obj_name
        obj_info['object_id'] = picked_uids[0]
        obj_info['object_path'] = obj_paths[0]

    return obj_info


# def load_lvis_annotations():
#     lvis_annotations = objaverse.load_lvis_annotations()
#     lvis_annotations = {k.lower(): v for k, v in lvis_annotations.items()}
#     return lvis_annotations


# def try_existing_obj_assets(obj_name):
#     ASSETS_DIR = './assets'
#     obj_name = obj_name.lower().replace(' ', '_')
#     obj_path = os.path.join(ASSETS_DIR, obj_name + '.glb')
#     if os.path.exists(obj_path):
#         print('Found existing object asset:', obj_name)
#         return obj_path
#     else:
#         print('Object asset not found:', obj_name)
#         return None


if __name__ == '__main__':
    # test object retrieval from Objaverse
    # obj_name = 'pikachu'
    # obj_info = retrieve_asset_from_objaverse(obj_name)
    # print(obj_info)

    # temporary code for downloading Objaverse assets
    # picked_uids = ['37c740f674cd4719a1d1d2970bbe8c30']
    # save_dir = './assets'
    # obj_paths = download_asset_from_objaverse(picked_uids, save_dir)

    # temporary code for testing retrieval of animated objects
    obj_name = 'dragon'
    obj_info = retrieve_asset_from_objaverse(obj_name, is_animated=True)
    print(obj_info)