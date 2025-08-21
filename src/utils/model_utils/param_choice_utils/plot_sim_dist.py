import numpy as np
import matplotlib.pyplot as plt
import torch

def compute_histogram(data, num_bins=50):
    """
    Compute histogram for 1D array using linear bins.
    """
    data = np.asarray(data).flatten()
    bins = np.linspace(data.min(), data.max(), num_bins)
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges


def compute_pairwise_similarity_hist(data, batch_size=1000, metric="L2"):
    """
    Compute histogram of pairwise similarities in batches for 1D or 2D tensor.
    Uses linear bins only.
    """
    data = data.float().cpu()
    N = data.shape[0]
    hist = None
    bin_edges = None

    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        Xi = data[i:end_i]

        for j in range(i, N, batch_size):
            end_j = min(j + batch_size, N)
            Xj = data[j:end_j]

            if metric == "L2":
                dist = torch.cdist(Xi, Xj, p=2)
                sim = 1 / (1 + dist)
            elif metric == "abs":
                p_i = Xi.view(-1, 1)
                p_j = Xj.view(1, -1)
                diff = torch.abs(p_i - p_j)
                sim = 1 / (1 + diff)
            else:
                raise ValueError(f"Unknown metric {metric}")

            sim = sim.cpu().numpy()
            if i == j:
                triu_idx = np.triu_indices(end_i - i, k=1)
                sim_vals = sim[triu_idx]
            else:
                sim_vals = sim.flatten()

            batch_hist, batch_bins = compute_histogram(sim_vals)
            if hist is None:
                hist = batch_hist
                bin_edges = batch_bins
            else:
                hist += batch_hist

    return hist, bin_edges


def get_bins(components, batch_size=1000):
    """
    Compute histogram bins for pic50, input_feats, and prot_sim using linear bins.
    """
    bins_components = {}

    # pic50 similarity
    hist, bin_edges = compute_pairwise_similarity_hist(
        components["pic50"], metric="abs", batch_size=batch_size
    )
    bins_components["pic50"] = {"hist": hist, "bin_edges": bin_edges}

    # input_feats similarity
    hist, bin_edges = compute_pairwise_similarity_hist(
        components["input_feats"], metric="L2", batch_size=batch_size
    )
    bins_components["input_feats"] = {"hist": hist, "bin_edges": bin_edges}

    # prot_sim weighted by local_protein_idx
    prot_sim = components["prot_sim"].cpu().numpy()
    local_idx = components["local_protein_idx"].cpu().numpy()
    N = len(local_idx)
    hist = None
    bin_edges = None

    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        idx_i = local_idx[i:end_i]
        for j in range(i, N, batch_size):
            end_j = min(j + batch_size, N)
            idx_j = local_idx[j:end_j]
            sim_vals = prot_sim[np.ix_(idx_i, idx_j)]
            if i == j:
                triu_idx = np.triu_indices(end_i - i, k=1)
                sim_vals = sim_vals[triu_idx]
            else:
                sim_vals = sim_vals.flatten()
            batch_hist, batch_bins = compute_histogram(sim_vals)
            if hist is None:
                hist = batch_hist
                bin_edges = batch_bins
            else:
                hist += batch_hist
    bins_components["prot_sim"] = {"hist": hist, "bin_edges": bin_edges}

    return bins_components


def plot_components_bins(
    bins_components,
    title="Components Similarity Distributions",
    logarithmic=False,
    exclude_logarithmic=None
):
    """
    Plot histograms for all components.
    logarithmic affects only the y-axis scale.
    """
    if exclude_logarithmic is None:
        exclude_logarithmic = []

    n = len(bins_components)
    ncols = 2
    nrows = (n + 1) // ncols

    plt.figure(figsize=(ncols * 6, nrows * 4))
    plt.suptitle(title, fontsize=16)

    for i, (key, data) in enumerate(bins_components.items(), 1):
        hist = data["hist"]
        bin_edges = data["bin_edges"]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)

        ax = plt.subplot(nrows, ncols, i)
        ax.bar(bin_centers, hist, width=bin_widths, align='center', edgecolor='black')
        ax.set_title(key)
        ax.set_xlabel(key)
        ax.set_ylabel("Frequency")

        # Apply log scale to y-axis if requested
        if logarithmic and (key not in exclude_logarithmic):
            ax.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



