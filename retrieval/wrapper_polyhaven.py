# Utility function to retrieve textures from PolyHaven (https://polyhaven.com/)
import os
import torch
import numpy as np
import pickle
import random
from sentence_transformers import SentenceTransformer


POLYHAVEN_MATERIALS_ROOT_DIR = './retrieval/polyhaven'
SBERT_TEXT_EMBEDS_PATH = './retrieval/embeddings/polyhaven_materials_sbert_embeddings.pkl'


def compute_sbert_text_features(sbert_model, text):
    with torch.no_grad():
        text_feature_sbert = sbert_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        text_feature_sbert /= text_feature_sbert.norm(dim=-1, keepdim=True)
        text_feature_sbert = text_feature_sbert.cpu().numpy()
    return text_feature_sbert


def init_model():
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    return sbert_model


def retrieve_materials_from_polyhaven(target_material_name):
    sbert_model = init_model()

    ##### import the embeddings #####
    with open(os.path.join(SBERT_TEXT_EMBEDS_PATH), 'rb') as f:
        data = pickle.load(f)
    embedding_sbert_np = data['database']
    uids = data['uids']

    ##### test the similarity #####
    target_material_name_feature = compute_sbert_text_features(sbert_model, target_material_name)
    similarity = np.dot(embedding_sbert_np, target_material_name_feature.T)
    similarity = similarity.squeeze()

    ##### get the top K similar materials #####
    top_k = 5
    top_k_indices = similarity.argsort()[::-1][:top_k]

    # print the top K similar materials
    for idx in top_k_indices:
        print(uids[idx], ": ", similarity[idx])

    material_folder = os.path.join(POLYHAVEN_MATERIALS_ROOT_DIR, uids[top_k_indices[random.randint(0, top_k - 1)]])
    return material_folder


if __name__ == '__main__':
    target_material_name = 'yellow brick'
    material_folder = retrieve_materials_from_polyhaven(target_material_name)
    print(material_folder)