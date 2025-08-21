import numpy as np
from typing import Dict, Tuple, List
from sklearn.decomposition import PCA
from loguru import logger   


def find_n_components_for_variance_threshold(
    data: np.ndarray,
    variance_threshold: float = 0.95
) -> int:
    """
    Determine how many PCA components are needed to explain the given amount of variance.

    Args:
        data: Input data array (e.g., embedding_matrix) to analyze.
        variance_threshold: Target cumulative variance to explain (default: 0.95).

    Returns:
        Number of components required to reach the variance threshold.
    """
    logger.info(f"Computing number of PCA components needed to explain {variance_threshold * 100:.1f}% variance")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    pca = PCA()
    pca.fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1

    logger.info(f"{n_components} components are needed to explain {variance_threshold * 100:.1f}% of the variance")

    return n_components
