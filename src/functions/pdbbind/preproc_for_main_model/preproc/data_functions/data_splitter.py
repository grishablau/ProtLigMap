
from logging import config
from typing import Dict, Any
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from loguru import logger

from src.utils.file_system import check_directory_exists


def batch_cosine_similarity(embeddings: np.ndarray, batch_size=1000) -> np.ndarray:
    n = len(embeddings)
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for start_i in range(0, n, batch_size):
        end_i = min(start_i + batch_size, n)
        batch_i = embeddings[start_i:end_i]

        for start_j in range(0, n, batch_size):
            end_j = min(start_j + batch_size, n)
            batch_j = embeddings[start_j:end_j]

            dot_products = np.dot(batch_i, batch_j.T)
            norms_i = np.linalg.norm(batch_i, axis=1)[:, None]
            norms_j = np.linalg.norm(batch_j, axis=1)[None, :]
            denom = norms_i * norms_j

            with np.errstate(divide='ignore', invalid='ignore'):
                sims = np.divide(dot_products, denom)
                sims[denom == 0] = 0.0

            sim_matrix[start_i:end_i, start_j:end_j] = sims
    return sim_matrix

def get_protein_clusters(data: Dict[str, Any], distance_threshold: float = 0.5) -> np.ndarray:
    protein_vecs = data['protein_vecs']
    unique_prots, inverse_indices = np.unique(protein_vecs, axis=0, return_inverse=True)

    logger.info(f"Computing cosine similarity on {len(unique_prots)} unique protein vectors")
    similarity_matrix = batch_cosine_similarity(unique_prots)
    distance_matrix = 1.0 - similarity_matrix

    logger.info("Clustering proteins with AgglomerativeClustering")
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='complete',
        distance_threshold=distance_threshold,
        n_clusters=None
    )

    cluster_labels_unique = clustering.fit_predict(distance_matrix)
    cluster_labels_all = cluster_labels_unique[inverse_indices]
    return cluster_labels_all

def split_by_protein_cluster(data: Dict[str, Any], held_out_clusters: list[int]) -> Dict[str, Dict[str, Any]]:
    distance_threshold = config["distance_threshold"]
    cluster_labels_all = get_protein_clusters(data, distance_threshold)
    data['protein_class'] = cluster_labels_all

    logger.info(f"Splitting data. Holding out clusters: {held_out_clusters}")
    train_idx = np.where(~np.isin(cluster_labels_all, held_out_clusters))[0]
    val_idx = np.where(np.isin(cluster_labels_all, held_out_clusters))[0]

    split = {}
    for key in ['embedding_matrix', 'pic50_vals', 'protein_vecs', 'valid_keys', 'protein_class']:
        arr = data[key]
        split[key] = {
            'train': arr[train_idx],
            'val': arr[val_idx]
        }
    return split
