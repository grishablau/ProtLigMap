import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_combined_similarity(final_train_data, 
                                w_protein=0.051, w_affinity=0.949, w_input=0.0,
                                affinity_mode='exp_abs', affinity_scale=1.0,
                                batch_size=1000, plot=True):
    """
    Compute combined similarity (protein + affinity + input features) in batches.
    
    Returns:
        combined_sim_flat (torch.Tensor): 1D tensor of upper-triangle similarities
        hist (np.ndarray): histogram counts
        bin_edges (np.ndarray): histogram bin edges
    """
    pic50 = final_train_data['pic50']
    input_feats = final_train_data['input_feats']
    prot_sim_mat = final_train_data['prot_sim']
    local_idx = final_train_data['local_protein_idx'].cpu().numpy()
    
    N = pic50.shape[0]
    combined_sim_vals = []

    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        pic50_i = pic50[i:end_i]
        input_i = input_feats[i:end_i]
        idx_i = local_idx[i:end_i]

        for j in range(i, N, batch_size):
            end_j = min(j + batch_size, N)
            pic50_j = pic50[j:end_j]
            input_j = input_feats[j:end_j]
            idx_j = local_idx[j:end_j]

            # ----- Affinity similarity -----
            pic50_i_exp = pic50_i.view(-1,1)
            pic50_j_exp = pic50_j.view(1,-1)
            diff = torch.abs(pic50_i_exp - pic50_j_exp)
            if affinity_mode == 'exp_abs':
                affinity_sim = torch.exp(-diff / affinity_scale)
            elif affinity_mode == 'exp_squared':
                affinity_sim = torch.exp(-(diff**2) / affinity_scale)
            else:
                raise ValueError("Unknown affinity_mode")

            # ----- Input feature similarity -----
            dist_matrix = torch.cdist(input_i, input_j, p=2)
            input_sim = 1 / (1 + dist_matrix)

            # ----- Protein similarity -----
            prot_sim_vals = prot_sim_mat[np.ix_(idx_i, idx_j)]
            prot_sim = torch.tensor(prot_sim_vals, dtype=torch.float32)

            # ----- Weighted combined similarity -----
            combined_sim = (prot_sim**w_protein) * (affinity_sim**w_affinity) * (input_sim**w_input)

            # ----- Extract upper-triangle for same-batch or flatten if cross-batch -----
            if i == j:
                triu_idx = torch.triu_indices(end_i - i, end_j - j, offset=1)
                combined_sim = combined_sim[triu_idx[0], triu_idx[1]]
            else:
                combined_sim = combined_sim.flatten()

            combined_sim_vals.append(combined_sim)

    combined_sim_flat = torch.cat(combined_sim_vals)

    # ----- Histogram -----
    combined_sim_np = combined_sim_flat.cpu().numpy()
    hist, bin_edges = np.histogram(combined_sim_np, bins=50)

    # ----- Optional plot -----
    if plot:
        plt.figure(figsize=(6,4))
        plt.hist(combined_sim_np, bins=50, color='skyblue', edgecolor='k')
        plt.xlabel("Combined similarity")
        plt.ylabel("Frequency")
        plt.title("Distribution of precomputed combined similarity")
        plt.show()

    return combined_sim_flat, hist, bin_edges
