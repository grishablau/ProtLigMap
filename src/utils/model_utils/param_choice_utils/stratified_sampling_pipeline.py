import copy
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

from src.utils.model_utils.param_choice_utils.plot_sim_dist import get_bins, plot_components_bins

import seaborn as sns
import pandas as pd
from scipy.stats import entropy

def representativity_report(original_data, reduced_data, pic50_bins=20):
    """
    Generates a full representativity report: plots + coverage scores.
    Adapted for `train_data_components` structure:
      - 'local_protein_idx' instead of 'protein'
      - 'input_feats' instead of 'input_features'
      - 'pic50' kept as-is
    """
    scores = {}
    
    # Protein coverage
    orig_proteins = set(original_data['local_protein_idx'].cpu().numpy())
    red_proteins = set(reduced_data['local_protein_idx'].cpu().numpy())
    scores['protein_coverage'] = len(red_proteins) / len(orig_proteins)

    # pIC50 KL divergence
    orig_hist, bin_edges = np.histogram(original_data['pic50'], bins=pic50_bins, density=True)
    red_hist, _ = np.histogram(reduced_data['pic50'], bins=bin_edges, density=True)
    orig_hist += 1e-8
    red_hist += 1e-8
    scores['pic50_kl_divergence'] = entropy(orig_hist, red_hist)

    # Input feature coverage via PCA variance ratio
    features_orig = np.vstack(original_data['input_feats'])
    features_red = np.vstack(reduced_data['input_feats'])

    pca = PCA(n_components=2)
    pca.fit(features_orig)
    orig_proj = pca.transform(features_orig)
    orig_var = np.sum(np.var(orig_proj, axis=0))

    red_proj = pca.transform(features_red)
    red_var = np.sum(np.var(red_proj, axis=0))

    scores['input_feature_variance_ratio'] = red_var / orig_var

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Protein distribution
    orig_counts = pd.Series(original_data['local_protein_idx'].cpu().numpy()).value_counts()
    red_counts = pd.Series(reduced_data['local_protein_idx'].cpu().numpy()).value_counts()
    
    sns.barplot(x=orig_counts.index, y=orig_counts.values, ax=axes[0], color="skyblue", alpha=0.5)
    sns.barplot(x=red_counts.index, y=red_counts.values, ax=axes[0], color="red", alpha=0.6)
    axes[0].set_title(f"Protein distribution\nProtein coverage: {scores['protein_coverage']:.2f}")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

    # pIC50 histogram
    sns.histplot(original_data['pic50'], bins=pic50_bins, color="skyblue", alpha=0.5, ax=axes[1], label="Original")
    sns.histplot(reduced_data['pic50'], bins=bin_edges, color="red", alpha=0.6, ax=axes[1], label="Selected")
    axes[1].set_title(f"pIC50 distribution\nKL divergence: {scores['pic50_kl_divergence']:.4f}")
    axes[1].legend()

    # Input feature PCA
    axes[2].scatter(orig_proj[:,0], orig_proj[:,1], alpha=0.3, color="skyblue", label="Original")
    axes[2].scatter(red_proj[:,0], red_proj[:,1], alpha=0.8, color="red", label="Selected")
    axes[2].set_title(f"Input feature space (PCA 2D)\nVariance ratio: {scores['input_feature_variance_ratio']:.2f}")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    
    return scores


def full_reduce_pipeline(
    data_components,
    points_num=1000,
    input_feat_clusters=50,
    plot_bins=True,
    plot_pca=True,
    report_scores=False,
    best_protein=False,       # NEW: toggle best protein mode
    random_state=42
):
    """
    Full pipeline: cluster, stratified sample, return reduced dataset.
    If best_protein=True, selects only the richest protein cluster.
    """
    np.random.seed(random_state)

    # -------------------------------
    # Helper functions
    # -------------------------------
    def cluster_by(data, by="protein", subcluster=False, parent_clusters=None):
        clusters = {}

        if not subcluster:
            if by == "protein":
                items = data["local_protein_idx"].cpu().numpy()
                for idx, protein in enumerate(items):
                    clusters.setdefault(int(protein), []).append(idx)
            elif by == "pic50":
                pic50_vals = np.array(data["pic50"])
                bins = np.digitize(pic50_vals, bins=[5, 7])
                for idx, bin_id in enumerate(bins):
                    clusters.setdefault(int(bin_id), []).append(idx)
            elif by == "input_features":
                features = np.vstack(data["input_feats"])
                n_clusters = min(input_feat_clusters, len(features))
                kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
                labels = kmeans.fit_predict(features)
                for idx, label in enumerate(labels):
                    clusters.setdefault(int(label), []).append(idx)
            else:
                raise ValueError(f"Unknown clustering key: {by}")
        else:
            for parent_id, idx_list in parent_clusters.items():
                pic50_vals = np.array(data["pic50"])[idx_list]
                bins = np.digitize(pic50_vals, bins=[5, 7])
                for idx, bin_id in zip(idx_list, bins):
                    clusters.setdefault((int(parent_id), int(bin_id)), []).append(idx)

        return clusters

    def stratified_clustering(protein_pic50_clusters, input_feature_clusters):
        stratified = defaultdict(list)
        for prot_pic_id, prot_pic_idxs in protein_pic50_clusters.items():
            for inp_id, inp_idxs in input_feature_clusters.items():
                intersection = list(set(prot_pic_idxs) & set(inp_idxs))
                if intersection:
                    stratified[(prot_pic_id, inp_id)] = intersection
        return dict(stratified)

    def pick_up_points_from_clusters(stratified_clusters, points_num=1000):
        """
        Pick exactly `points_num` points from stratified clusters while maintaining
        approximate proportional representation.
        """
        all_indices = []
        cluster_sizes = {k: len(v) for k, v in stratified_clusters.items()}
        total_points = sum(cluster_sizes.values())

        # Step 1: Initial proportional pick (floored)
        picks_per_cluster = {}
        for cluster_id, size in cluster_sizes.items():
            pick = int(size / total_points * points_num)
            picks_per_cluster[cluster_id] = min(pick, size)

        # Step 2: Adjust to reach exactly points_num
        picked_so_far = sum(picks_per_cluster.values())
        remainder = points_num - picked_so_far

        if remainder > 0:
            leftover = {k: cluster_sizes[k] - picks_per_cluster[k] for k in cluster_sizes}
            sorted_clusters = sorted(leftover.items(), key=lambda x: -x[1])
            for cluster_id, available in sorted_clusters:
                if remainder <= 0:
                    break
                add = min(available, remainder)
                picks_per_cluster[cluster_id] += add
                remainder -= add

        # Step 3: Actually select points
        for cluster_id, num_pick in picks_per_cluster.items():
            if num_pick > 0:
                idx_list = stratified_clusters[cluster_id]
                selected = np.random.choice(idx_list, num_pick, replace=False)
                all_indices.extend(selected)

        np.random.shuffle(all_indices)
        return all_indices

    def copy_data(data):
        return copy.deepcopy(data)

    def subset_data(data, selected_indices):
        subset = {}
        for key, values in data.items():
            if key in ["pic50", "input_feats", "local_protein_idx"]:
                if isinstance(values, torch.Tensor):
                    subset[key] = values[selected_indices]
                else:
                    subset[key] = [values[i] for i in selected_indices]
            else:
                subset[key] = values
        return subset

    def plot_pca_coverage(data, selected_indices):
        features = np.vstack(data["input_feats"])
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        plt.figure(figsize=(8, 6))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.3, label="All points")
        plt.scatter(features_2d[selected_indices, 0], features_2d[selected_indices, 1],
                    color='red', alpha=0.8, label="Selected points")
        plt.title("PCA coverage of selected points")
        plt.legend()
        plt.show()

    # -------------------------------
    # Step 0: optionally pick the best protein
    # -------------------------------
    if best_protein:
        clustered_by_protein = cluster_by(data_components, by="protein")
        best_prot_id = max(clustered_by_protein.items(), key=lambda x: len(x[1]))[0]
        indices = clustered_by_protein[best_prot_id]
        data_components = subset_data(copy_data(data_components), indices)

    # -------------------------------
    # Step 1: Cluster by protein and pIC50
    # -------------------------------
    clustered_by_protein = cluster_by(data_components, by="protein")
    sub_clustered_by_pic50 = cluster_by(data_components, by="pic50", subcluster=True, parent_clusters=clustered_by_protein)

    # -------------------------------
    # Step 2: Cluster input features
    # -------------------------------
    clustered_by_input_features = cluster_by(data_components, by="input_features")

    # -------------------------------
    # Step 3: Stratified clustering
    # -------------------------------
    stratified_clusters = stratified_clustering(sub_clustered_by_pic50, clustered_by_input_features)

    # -------------------------------
    # Step 4: Pick representative points
    # -------------------------------
    selected_indices = pick_up_points_from_clusters(stratified_clusters, points_num=points_num)

    # -------------------------------
    # Step 5: Subset dataset
    # -------------------------------
    reduced_data = subset_data(copy_data(data_components), selected_indices)

    # -------------------------------
    # Step 6: Optional plots
    # -------------------------------
    if plot_bins:
        bins = get_bins(reduced_data)
        plot_components_bins(bins)
    
    if plot_pca:
        plot_pca_coverage(data_components, selected_indices)

    # -------------------------------
    # Step 7: Optional representativity scores
    # -------------------------------
    scores = None
    if report_scores:
        scores = representativity_report(data_components, reduced_data)

    return reduced_data, scores, bins



def save_reduced_dataset(final_train_data, original_dataset_path, new_dataset_path):
    """
    Save reduced dataset to a new file, keeping the structure compatible
    with the model code.
    
    final_train_data: dict
        Output of your full_reduce_pipeline, containing:
        'pic50', 'input_feats', 'prot_sim', 'agg_to_local', 'idx_to_agg', 'local_protein_idx'
    original_dataset_path: str
        Path to the original dataset (used to copy the full structure if needed)
    new_dataset_path: str
        Path to save the new dataset
    """
    # Load original dataset
    original_data = torch.load(original_dataset_path, weights_only=False)

    # Create a new dataset object by copying the original and replacing the train data
    new_data = original_data.copy()
    
    # Assuming original_data['train'] is an object with attributes .pic50, .embeddings, .prot_sim_mat
    train_obj = original_data['train']

    # Replace attributes with reduced dataset
    train_obj.pic50 = final_train_data['pic50']
    train_obj.embeddings = final_train_data['input_feats']
    train_obj.prot_sim_mat = final_train_data['prot_sim']

    # Keep the agg_to_local and idx_to_agg the same
    train_obj.agg_to_local = final_train_data['agg_to_local']
    train_obj.idx_to_agg = final_train_data['idx_to_agg']

    # Save the new dataset
    torch.save(new_data, new_dataset_path)
    print(f"Reduced dataset saved to {new_dataset_path}")
