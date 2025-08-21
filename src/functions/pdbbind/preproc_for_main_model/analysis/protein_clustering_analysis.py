import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from typing import Dict, List, Tuple, Union
from joblib import Parallel, delayed, parallel_backend
from loguru import logger
import logging
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np

# === Preprocessing ===
def standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)

def reduce_pca(X: np.ndarray, n_components: float = 0.95) -> np.ndarray:
    return PCA(n_components=n_components).fit_transform(X)

# === Clustering ===
def analyse_protein_clusters_kmeans(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, Dict[int, List[float]]]:
    kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1000,
            n_init=1,
            max_iter=20,
            random_state=42,
        )
    labels = kmeans.fit_predict(X)
    cluster_pic50 = defaultdict(list)
    for label, val in zip(labels, y):
        cluster_pic50[label].append(val)
    return labels, cluster_pic50

def calculate_stat(cluster_pic50: Dict[int, List[float]], stat: str = "intra_variance") -> float:
    if stat == "intra_variance":
        variances = [np.var(vals) for vals in cluster_pic50.values() if len(vals) > 1]
        return np.mean(variances) if variances else 0.0
    raise ValueError(f"Unknown stat type: {stat}")

def compute_stat_for_cluster(X, y, n_clusters, stat, std, pca):
    labels, cluster_pic50 = analyse_protein_clusters_kmeans(X, y, n_clusters)
    stat_val = calculate_stat(cluster_pic50, stat)
    logger.info(f"[{stat}] Std={std}, PCA={pca}, Clusters={n_clusters} → Stat={stat_val:.4f}")
    return stat_val

# === Smart Search ===
def binary_stat_search(X, y, stat, std, pca, low=200, high=1500, tol=0.02, max_iter=15):
    best_n = None
    best_stat = float("inf")
    for _ in range(max_iter):
        mid = (low + high) // 2
        current_stat = compute_stat_for_cluster(X, y, mid, stat, std, pca)
        if abs(current_stat - 1.0) < tol:
            return mid, current_stat
        if current_stat > 1:
            low = mid + 1
        else:
            high = mid - 1
        if abs(current_stat - 1.0) < abs(best_stat - 1.0):
            best_n, best_stat = mid, current_stat
    return best_n, best_stat

# === Main ===
def plot_stats_vs_threshold_multi(
    data: Dict[str, np.ndarray],
    thresholds: Union[np.ndarray, None],
    config_options: List[Tuple[bool, bool]],
    stat_options: List[str],
    n_jobs: int = 96,
    search_method: str = "exhaustive",  # or "smart_search",
    low_input: int = 200,
    high_input: int = 1500,
    tol_input: int = 0.02,
    max_iter_input: int = 15
):
    y = data["pic50_vals"]
    preprocessed_data = {}

    # Preprocess with each config
    for std, pca in config_options:
        X = data["protein_vecs"]
        if std:
            X = standardize(X)
        if pca:
            X = reduce_pca(X, n_components=0.95)
            logger.info(f"[Std={std}, PCA={pca}] → PCA reduced to {X.shape[1]} dims")
        preprocessed_data[(std, pca)] = X

    # Compute stats
    for stat in stat_options:
        plt.figure(figsize=(10, 6))
        for std, pca in config_options:
            X = preprocessed_data[(std, pca)]

            if search_method == "smart_search":
                best_n, best_val = binary_stat_search(X, y, stat, std, pca, low=low_input, high=high_input, tol=tol_input, max_iter=max_iter_input)
                logger.info(f"Best Cluster={best_n}, {stat}={best_val:.4f} for config std={std}, pca={pca}")
                plt.scatter([best_n], [best_val], marker='x', color='red', label=f"Best: std={std}, pca={pca}")
                continue

            # Default: exhaustive
            cluster_range = thresholds.astype(int)
            with parallel_backend("loky", inner_max_num_threads=1):
                values = Parallel(n_jobs=min(n_jobs, 32))(
                    delayed(compute_stat_for_cluster)(X, y, n_clusters, stat, std, pca)
                    for n_clusters in cluster_range
                )
            plt.plot(thresholds, values, marker='o', label=f"std={std}, pca={pca}")

        plt.xlabel("# Clusters")
        plt.ylabel(f"{stat} over clusters")
        plt.title(f"{stat} vs # clusters")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def aggregate_duplicates_with_ids(
    X: np.ndarray,
    pic50_vals: np.ndarray,
    valid_keys: np.ndarray,
    decimal_round=4
):
    # Round embeddings to reduce floating noise (optional)
    X_rounded = np.round(X, decimals=decimal_round)

    # Convert each embedding vector to a tuple (hashable)
    keys = [tuple(row) for row in X_rounded]

    # Group embeddings by rounded vector; collect all pic50s and all protein IDs per group
    grouped = defaultdict(lambda: {"pic50s": [], "protein_ids": []})

    for key, pic50, valid_key in zip(keys, pic50_vals, valid_keys):
        grouped[key]["pic50s"].append(pic50)
        grouped[key]["protein_ids"].append(valid_key.split('_')[0])  # protein ID from valid_key

    unique_embeddings = []
    aggregated_pic50 = []
    unique_protein_id_lists = []  # now store list of protein IDs per group

    for key, group in grouped.items():
        unique_embeddings.append(np.array(key))
        aggregated_pic50.append(np.max(group["pic50s"]))
        unique_protein_id_lists.append(group["protein_ids"])  # keep all protein IDs per group

    unique_embeddings = np.stack(unique_embeddings)
    aggregated_pic50 = np.array(aggregated_pic50)
    # Keep as list of lists, NOT np.array because uneven lengths
    # unique_protein_id_lists = np.array(unique_protein_id_lists)

    return unique_embeddings, aggregated_pic50, unique_protein_id_lists

import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import numpy as np

def plot_and_compare_pic50_distributions(train_pic50, val_pic50, title):
    import matplotlib.pyplot as plt
    from scipy.stats import wasserstein_distance
    import numpy as np

    plt.figure(figsize=(8, 4))
    bins = np.linspace(min(train_pic50.min(), val_pic50.min()), max(train_pic50.max(), val_pic50.max()), 50)

    plt.hist(train_pic50, bins=bins, alpha=0.6, label='Train', color='blue', density=True)
    plt.hist(val_pic50, bins=bins, alpha=0.6, label='Val', color='orange', density=True)
    plt.title(title)
    plt.xlabel('pIC50')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Wasserstein distance (lower = better match)
    distance = wasserstein_distance(train_pic50, val_pic50)
    print(f"[{title}] Wasserstein Distance: {distance:.4f}")
    return distance




def cluster_and_split_data(data, n_clusters=1338, test_size=0.3, random_state=42):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(console_handler)

    logger.info(f"Clustering {data['protein_vecs'].shape[0]} proteins into {n_clusters} classes...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        init="k-means++",
        random_state=random_state
    )
    cluster_labels = kmeans.fit_predict(data['protein_vecs'])

    logger.info("Clustering done.")

    # data["unique_protein_ids"] is now list of lists of protein IDs per cluster
    unique_id_lists = data["unique_protein_ids"]  # list of lists

    # Build reverse map: protein_id -> cluster index
    protein_to_cluster = {}
    for cluster_idx, protein_id_list in enumerate(unique_id_lists):
        for pid in protein_id_list:
            protein_to_cluster[pid] = cluster_idx

    protein_names = np.array([key.split('_')[0] for key in data['valid_keys']])

    # Check missing IDs
    unique_ids_set = set(protein_to_cluster.keys())
    missing_ids = set(protein_names) - unique_ids_set
    if missing_ids:
        logger.warning(f"Number of protein IDs in valid_keys not found in any cluster: {len(missing_ids)}")
        logger.warning(f"Some missing IDs: {list(missing_ids)[:10]}")

    # Assign cluster to each original data point (protein vector)
    # For missing IDs, assign -1 cluster
    protein_classes_full = np.array([
        protein_to_cluster.get(pid, -1) for pid in protein_names
    ])

    # Filter out missing IDs from train/val split
    valid_mask = protein_classes_full != -1
    logger.info(f"Total valid protein vectors after filtering missing IDs: {valid_mask.sum()}")

    unique_classes = np.unique(protein_classes_full[valid_mask])
    train_classes, val_classes = train_test_split(
        unique_classes,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Number of clusters in TRAIN: {len(train_classes)}")
    print(f"Number of clusters in VAL: {len(val_classes)}")


    train_mask = np.isin(protein_classes_full, train_classes) & valid_mask
    val_mask = np.isin(protein_classes_full, val_classes) & valid_mask

    def subset_data(mask):
        return {
            'valid_keys': data['valid_keys'][mask],
            'protein_class': protein_classes_full[mask],
            'pic50_vals': data['pic50_vals_raw'][mask],
            'protein_vecs': data['protein_vecs_raw'][mask],
        }

    data_processed = {
        'train': subset_data(train_mask),
        'val': subset_data(val_mask),
        'log': {
            'n_total': len(data['valid_keys']),
            'n_train': train_mask.sum(),
            'n_val': val_mask.sum(),
            'n_clusters': n_clusters,
            'mean_pic50_train': data['pic50_vals_raw'][train_mask].mean(),
            'mean_pic50_val': data['pic50_vals_raw'][val_mask].mean(),
        }
    }

    passthrough_keys = ['embedding_matrix', 'model_outputs', 'results', 'log_file', 'mid_outputs']
    for key in passthrough_keys:
        if key in data and isinstance(data[key], np.ndarray):
            data_processed[key] = data[key]

    logger.info(f"Split complete. Train={data_processed['log']['n_train']}, Val={data_processed['log']['n_val']}")
    logger.info(f"Avg pIC50s: Train={data_processed['log']['mean_pic50_train']:.4f}, Val={data_processed['log']['mean_pic50_val']:.4f}")



    return data_processed


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def analyze_pic50_variances_and_cluster_sizes(X, y, n_clusters=1338):
    # Step 1: KMeans clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        init="k-means++",
        random_state=42
    )
    labels = kmeans.fit_predict(X)

    # Step 2: Group pIC50s and count sizes
    cluster_to_pic50s = defaultdict(list)
    cluster_sizes = np.zeros(n_clusters, dtype=int)

    for label, val in zip(labels, y):
        cluster_to_pic50s[label].append(val)
        cluster_sizes[label] += 1

    # Step 3: Compute variances for clusters with >1 point
    variances = [np.var(v) for v in cluster_to_pic50s.values() if len(v) > 1]

    # Step 4: Summary of cluster sizes
    unique, counts = np.unique(cluster_sizes, return_counts=True)
    size_summary = dict(zip(unique, counts))

    print("=== Cluster Size Summary ===")
    for i in range(6):
        count = size_summary.get(i, 0)
        print(f"Clusters with {i} points: {count}")
    print(f"Clusters with >5 points: {sum(v for k, v in size_summary.items() if k > 5)}")
    print()

    # Step 5: Print variance stats
    variances = np.array(variances)
    print("=== pIC50 Variance per Cluster (clusters with >1 point) ===")
    print(f"Total clusters used: {len(variances)}")
    print(f"Min variance:   {variances.min():.4f}")
    print(f"Max variance:   {variances.max():.4f}")
    print(f"Mean variance:  {variances.mean():.4f}")
    print(f"Median variance:{np.median(variances):.4f}")
    print(f"Std of variances: {np.std(variances):.4f}")

    # === Plot 1: Histogram of variances ===
    plt.figure(figsize=(8, 4))
    plt.hist(variances, bins=40, color='skyblue', edgecolor='black')
    plt.xlabel("Variance of pIC50 in Cluster")
    plt.ylabel("Number of Clusters")
    plt.title(f"pIC50 Variance Across Clusters ({n_clusters} clusters)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot 2: Histogram of cluster sizes ===
    plt.figure(figsize=(8, 4))
    plt.hist(cluster_sizes, bins=range(0, cluster_sizes.max()+2), color='salmon', edgecolor='black', align='left')
    plt.xlabel("Number of Points in Cluster")
    plt.ylabel("Number of Clusters")
    plt.title(f"Cluster Size Distribution ({n_clusters} clusters)")
    plt.xticks(range(0, cluster_sizes.max()+1, max(1, cluster_sizes.max() // 20)))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return variances, cluster_sizes

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def cluster_proteins_kmeans(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    use_cosine: bool = False,
    random_state: int = 42,
    batch_size: int = 1000,
):
    """
    Cluster protein embeddings with KMeans.
    If use_cosine=True, normalize embeddings to unit length so
    Euclidean distance approximates cosine similarity.
    Returns:
        - cluster labels (np.ndarray)
        - dict: cluster index -> list of pIC50 values
        - silhouette score (float, cosine or euclidean depending on use_cosine)
    """
    if use_cosine:
        X_norm = normalize(X, norm='l2', axis=1)
    else:
        X_norm = X

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=10,
        max_iter=300,
        random_state=random_state,
        init="k-means++",
    )
    labels = kmeans.fit_predict(X_norm)

    cluster_pic50 = defaultdict(list)
    for label, val in zip(labels, y):
        cluster_pic50[label].append(val)

    # Silhouette score for cluster quality (if n_clusters>1)
    try:
        metric = 'cosine' if use_cosine else 'euclidean'
        sil_score = silhouette_score(X_norm, labels, metric=metric)
    except Exception as e:
        sil_score = None

    return labels, cluster_pic50, sil_score


def analyze_clusters_and_plot(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    use_cosine: bool = False
):
    labels, cluster_pic50, sil_score = cluster_proteins_kmeans(X, y, n_clusters, use_cosine)

    cluster_sizes = np.array([len(cluster_pic50[i]) for i in range(n_clusters)])
    variances = np.array([np.var(cluster_pic50[i]) if len(cluster_pic50[i]) > 1 else 0 for i in range(n_clusters)])

    print(f"=== Cluster Size Summary ===")
    for i in range(6):
        print(f"Clusters with {i} points: {(cluster_sizes == i).sum()}")
    print(f"Clusters with >5 points: {(cluster_sizes > 5).sum()}\n")

    filtered_vars = variances[cluster_sizes > 1]

    print(f"=== pIC50 Variance per Cluster (clusters with >1 point) ===")
    print(f"Total clusters used: {len(filtered_vars)}")
    print(f"Min variance: {filtered_vars.min():.4f}")
    print(f"Max variance: {filtered_vars.max():.4f}")
    print(f"Mean variance: {filtered_vars.mean():.4f}")
    print(f"Median variance: {np.median(filtered_vars):.4f}")
    print(f"Std of variances: {filtered_vars.std():.4f}")

    print(f"\nSilhouette Score (metric={'cosine' if use_cosine else 'euclidean'}): {sil_score:.4f}\n")

    # Plot variance histogram
    plt.figure(figsize=(8, 4))
    plt.hist(filtered_vars, bins=40, color='skyblue', edgecolor='black')
    plt.xlabel("Variance of pIC50 in Cluster")
    plt.ylabel("Number of Clusters")
    plt.title(f"pIC50 Variance Across Clusters ({n_clusters} clusters)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot cluster size histogram
    plt.figure(figsize=(8, 4))
    plt.hist(cluster_sizes, bins=range(0, cluster_sizes.max() + 2), color='salmon', edgecolor='black', align='left')
    plt.xlabel("Number of Points in Cluster")
    plt.ylabel("Number of Clusters")
    plt.title(f"Cluster Size Distribution ({n_clusters} clusters)")
    plt.xticks(range(0, cluster_sizes.max() + 1, max(1, cluster_sizes.max() // 20)))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return labels, cluster_pic50, sil_score


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from joblib import Parallel, delayed
from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt

def cluster_proteins_by_embeddings_parallel(
    protein_vecs: np.ndarray,
    min_k: int = 2,
    max_k: int = 100,
    k_step: int = 1,
    plot: bool = True,
    n_jobs: int = -1
):
    # Standardize protein vectors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(protein_vecs)

    ks = list(range(min_k, max_k + 1, k_step))

    def evaluate_k(k):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        inertia = kmeans.inertia_
        return sil, db, inertia

    print(f"Running KMeans in parallel for k from {min_k} to {max_k} (step={k_step})...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_k)(k) for k in ks
    )

    sil_scores, db_scores, inertias = zip(*results)

    # Use KneeLocator to find optimal k
    sil_knee = KneeLocator(ks, sil_scores, curve="convex", direction="increasing")
    db_knee = KneeLocator(ks, db_scores, curve="convex", direction="decreasing")
    inertia_knee = KneeLocator(ks, inertias, curve="convex", direction="decreasing")

    best_k = {
        "silhouette": sil_knee.knee,
        "davies_bouldin": db_knee.knee,
        "inertia": inertia_knee.knee,
    }

    if plot:
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        ax[0].plot(ks, sil_scores, label='Silhouette Score')
        if sil_knee.knee:
            ax[0].axvline(sil_knee.knee, color='r', linestyle='--', label=f"Knee: {sil_knee.knee}")
        ax[0].set_ylabel('Silhouette')
        ax[0].legend()

        ax[1].plot(ks, db_scores, label='Davies-Bouldin Score', color='orange')
        if db_knee.knee:
            ax[1].axvline(db_knee.knee, color='r', linestyle='--', label=f"Knee: {db_knee.knee}")
        ax[1].set_ylabel('DB Score')
        ax[1].legend()

        ax[2].plot(ks, inertias, label='Inertia', color='green')
        if inertia_knee.knee:
            ax[2].axvline(inertia_knee.knee, color='r', linestyle='--', label=f"Knee: {inertia_knee.knee}")
        ax[2].set_ylabel('Inertia')
        ax[2].set_xlabel('Number of Clusters')
        ax[2].legend()

        plt.suptitle("Clustering Quality vs. Number of Clusters")
        plt.tight_layout()
        plt.show()

    return {
        "silhouette_scores": sil_scores,
        "davies_bouldin_scores": db_scores,
        "inertias": inertias,
        "cluster_range": ks,
        "X_scaled": X_scaled,
        "best_k": best_k
    }



import os
import pickle
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# === Logger Setup ===
def setup_logger(log_path):
    logger = logging.getLogger("split_logger")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def split_data_by_protein_class(data, n_clusters=700, save_dir="cluster_split_outputs"):
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(os.path.join(save_dir, "split_log.txt"))

    # Extract arrays
    protein_vecs_raw = data['protein_vecs_raw']
    protein_vecs = data['protein_vecs']
    pic50_vals = data['pic50_vals']
    valid_keys = data['valid_keys']
    
    # Run KMeans clustering on full protein data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    all_cluster_labels = kmeans.fit_predict(protein_vecs_raw)

    # Assign classes to selected points only (5279 points)
    selected_classes = all_cluster_labels[:len(protein_vecs)]

    # Stratified split to avoid shared classes between train/val
    unique_classes = np.unique(selected_classes)
    train_classes, val_classes = train_test_split(unique_classes, test_size=0.3, random_state=42)

    train_indices = [i for i, c in enumerate(selected_classes) if c in train_classes]
    val_indices = [i for i, c in enumerate(selected_classes) if c in val_classes]

    def build_subset(indices):
        return {
            'valid_keys': valid_keys[indices],
            'protein_class': selected_classes[indices],
            'pic50_vals': pic50_vals[indices],
            'protein_vecs': protein_vecs[indices],
        }

    data_processed = {
        'train': build_subset(train_indices),
        'val': build_subset(val_indices)
    }

    # Add relevant keys from the original `data`
    for key, val in data.items():
        if key in ['embedding_matrix', 'pic50_vals', 'protein_vecs', 'valid_keys', 'protein_vecs_raw', 'pic50_vals_raw']:
            continue  # Already included or unnecessary
        data_processed[key] = val

    # Save processed data
    output_file = os.path.join(save_dir, "data_processed.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(data_processed, f)

    # Log information
    logger.info("Finished data split.")
    logger.info(f"Train size: {len(train_indices)} | Val size: {len(val_indices)}")
    logger.info(f"Train classes: {len(train_classes)} | Val classes: {len(val_classes)}")
    logger.info(f"Data saved to: {output_file}")

    return data_processed
