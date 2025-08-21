import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_similarity(final_train_data, w_affinity=0.976, w_input=0.024, affinity_mode='exp_abs', affinity_scale=1.0):
    """
    Compute combined similarity (affinity + input features) ahead of training,
    plot its distribution, and return histogram values.
    
    Returns:
        combined_sim (torch.Tensor): flattened similarity matrix
        hist (np.ndarray): histogram counts
        bin_edges (np.ndarray): histogram bin edges
    """
    pic50 = final_train_data['pic50']
    input_feats = final_train_data['input_feats']

    # ----- Affinity similarity -----
    pic50_i = pic50.view(-1, 1)
    pic50_j = pic50.view(1, -1)
    diff = torch.abs(pic50_i - pic50_j)
    
    if affinity_mode == 'exp_abs':
        affinity_sim = torch.exp(-diff / affinity_scale)
    elif affinity_mode == 'exp_squared':
        affinity_sim = torch.exp(-(diff**2) / affinity_scale)
    else:
        raise ValueError("Unknown affinity_mode")

    # ----- Input feature similarity -----
    dist_matrix = torch.cdist(input_feats, input_feats, p=2)
    input_sim = 1 / (1 + dist_matrix)

    # ----- Weighted combined similarity -----
    combined_sim = (affinity_sim**w_affinity) * (input_sim**w_input)

    # ----- Flatten and plot histogram -----
    combined_sim_np = combined_sim.flatten().cpu().numpy()
    hist, bin_edges = np.histogram(combined_sim_np, bins=50)

    plt.figure(figsize=(6,4))
    plt.hist(combined_sim_np, bins=50, color='skyblue', edgecolor='k')
    plt.xlabel("Combined similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of precomputed combined similarity")
    plt.show()

    return combined_sim, hist, bin_edges
