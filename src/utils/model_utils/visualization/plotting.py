import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for headless environments

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize, KBinsDiscretizer
from sklearn.decomposition import PCA
import torch
import os
import logging
from tqdm import tqdm
import umap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ==== Helper functions ====

def cluster_agg_vectors(prot_agg_proc, n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=42).fit(prot_agg_proc).labels_

def map_agg_labels_to_train_val(protein_clusters_agg, prot_idx_train, prot_idx_val):
    return (
        np.array([protein_clusters_agg[idx] for idx in prot_idx_train]),
        np.array([protein_clusters_agg[idx] for idx in prot_idx_val])
    )

def compute_umap(X, random_state=42):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.5, n_components=2, random_state=random_state)
    return reducer.fit_transform(X)

def save_umap_plot(E2d, labels, title, filename, cmap_name='tab20', continuous=False):
    plt.figure(figsize=(6, 5))
    if continuous:
        cmap = plt.get_cmap(cmap_name)
        normed = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
        colors = cmap(normed)
    else:
        n_classes = len(np.unique(labels))
        cmap = plt.get_cmap('tab20' if n_classes <= 20 else 'tab20b')
        colors = cmap((labels % cmap.N) / cmap.N)
    plt.scatter(E2d[:, 0], E2d[:, 1], c=colors, s=10, alpha=0.7)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_pic50_binning(n_bins, pic50_raw_train, pic50_raw_val):
    binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    train_bins = binner.fit_transform(pic50_raw_train.reshape(-1, 1)).flatten().astype(int)
    val_bins = binner.transform(pic50_raw_val.reshape(-1, 1)).flatten().astype(int)
    return train_bins, val_bins

# ==== Manifold Visualizer Class ====

class ManifoldVisualizer:
    def __init__(self, data, k_means_prot_cluster_list=[5], n_pic50_bins=4):
        self.k_means_prot_cluster_list = k_means_prot_cluster_list

        self.prot_idx_train = data['train'].prot_idx
        self.prot_idx_val = data['val'].prot_idx

        self.pic50_raw_train = data['train'].pic50_raw
        self.pic50_raw_val = data['val'].pic50_raw

        self.X_train_orig = data['train'].embeddings
        self.X_val_orig = data['val'].embeddings

        # Preprocess protein vectors
        all_prots = data['train'].prot_raw_agg_filtered_into_model
        prot_std = StandardScaler().fit_transform(all_prots)
        prot_pca = PCA(n_components=0.95).fit_transform(prot_std)
        self.prot_agg_proc_filtered = normalize(prot_pca, norm='l2')

        # Cluster proteins
        self.cluster_labels_agg = {
            k: cluster_agg_vectors(self.prot_agg_proc_filtered, k)
            for k in k_means_prot_cluster_list
        }

        # Bin pIC50
        self.pic50_train_bins, self.pic50_val_bins = run_pic50_binning(
            n_pic50_bins,
            self.pic50_raw_train,
            self.pic50_raw_val
        )

    def get_cluster_labels(self, k):
        return map_agg_labels_to_train_val(
            self.cluster_labels_agg[k],
            self.prot_idx_train,
            self.prot_idx_val
        )

    def visualize(self, model=None, device=None, embedding_type="orig", tag="", output_dir="/pca"):
        """
        embedding_type options:
            - "orig"      → raw dataset embeddings
            - "manifold"  → model.manifold_emb
            - "proj"      → model.proj_emb
        """
        os.makedirs(output_dir, exist_ok=True)

        def get_embeddings(X):
            if embedding_type == "orig" or model is None:
                return X.numpy()

            model.eval()
            with torch.no_grad():
                outputs = model(X.to(device))

            # Case 1: model returns a tuple
            if isinstance(outputs, tuple):
                if embedding_type == "manifold":
                    return outputs[0].cpu().numpy()
                elif embedding_type == "proj":
                    return outputs[1].cpu().numpy()

            # Case 2: model returns an object with attributes
            if embedding_type == "manifold":
                return outputs.manifold_emb.cpu().numpy()
            elif embedding_type == "proj":
                return outputs.proj_emb.cpu().numpy()

            raise ValueError(f"Unknown embedding_type: {embedding_type}")


        logging.info("Extracting embeddings")
        X_train = get_embeddings(self.X_train_orig)
        X_val = get_embeddings(self.X_val_orig)

        data_splits = {
            'train': (X_train, self.pic50_train_bins),
            'val': (X_val, self.pic50_val_bins)
        }

        # Plot protein clusters
        logging.info("Plotting UMAP for protein clusters")
        for k in self.k_means_prot_cluster_list:
            train_labels, val_labels = self.get_cluster_labels(k)
            for name, (X, _) in tqdm(data_splits.items(), desc=f"Protein clusters (k={k})"):
                labels = train_labels if name == 'train' else val_labels
                E2d = compute_umap(X)
                filename = os.path.join(output_dir, f"umap_prot_k{k}_{name}_{tag}.png")
                save_umap_plot(E2d, labels,
                               f"umap ({name}) - protein clusters (k={k}) {tag}",
                               filename)
                logging.info(f"Saved: {filename}")

        # Plot pIC50 bins
        logging.info("Plotting umap for pIC50 bins")
        for name, (X, pic50_bins) in tqdm(data_splits.items(), desc="pIC50 bins"):
            E2d = compute_umap(X)
            filename = os.path.join(output_dir, f"umap_pic50_{name}_{tag}.png")
            save_umap_plot(E2d, pic50_bins,
                           f"umap ({name}) - pIC50 bins {tag}",
                           filename,
                           cmap_name='viridis',
                           continuous=True)
            logging.info(f"Saved: {filename}")

        # Plot combined clusters + pIC50
        logging.info("Plotting umap for combined protein clusters + pIC50")
        k = self.k_means_prot_cluster_list[0]
        train_labels, val_labels = self.get_cluster_labels(k)
        combined = {
            'train': train_labels * 100 + self.pic50_train_bins,
            'val': val_labels * 100 + self.pic50_val_bins
        }

        for name, (X, _) in tqdm(data_splits.items(), desc=f"Combined k={k}"):
            E2d = compute_umap(X)
            filename = os.path.join(output_dir, f"umap_combined_k{k}_{name}_{tag}.png")
            save_umap_plot(E2d, combined[name],
                           f"umap ({name}) - combined protein/pIC50 (k={k}) {tag}",
                           filename)
            logging.info(f"Saved: {filename}")
