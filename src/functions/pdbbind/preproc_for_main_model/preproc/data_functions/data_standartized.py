from typing import Dict, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from loguru import logger


def standardize_and_transform_embeddings(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    key: str = "embedding_matrix",
    apply_pca: bool = True,
    n_components: int = 51
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Standardize and optionally apply PCA to the embeddings in train and val data.
    
    Args:
        train_data: Dictionary containing 'embedding_matrix' for training set.
        val_data: Dictionary containing 'embedding_matrix' for validation set.
        key: Which key to standardize in the dicts (default: 'embedding_matrix').
        apply_pca: Whether to apply PCA after standardization.
        n_components: Number of PCA components if PCA is applied.

    Returns:
        Tuple of updated (train_data, val_data) with transformed embeddings.
    """
    logger.info("Standardizing embedding_matrix using only train data stats")

    scaler = StandardScaler()
    train_emb = train_data[key]
    val_emb = val_data[key]

    if train_emb.ndim == 1:
        train_emb = train_emb.reshape(-1, 1)
    if val_emb.ndim == 1:
        val_emb = val_emb.reshape(-1, 1)

    # Fit only on training data
    scaler.fit(train_emb)
    train_scaled = scaler.transform(train_emb)
    val_scaled = scaler.transform(val_emb)

    # Update dicts
    train_data[key] = train_scaled
    val_data[key] = val_scaled

    if apply_pca:
        logger.info(f"Applying PCA to standardized embeddings with {n_components} components")
        pca = PCA(n_components=n_components)
        train_pca = pca.fit_transform(train_scaled)
        val_pca = pca.transform(val_scaled)

        train_data[key] = train_pca
        val_data[key] = val_pca

    return train_data, val_data
