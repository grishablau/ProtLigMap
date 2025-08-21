import numpy as np
import torch


def compute_pos_neg_weights_batched(final_train_data, 
                                    w_protein=0.051, w_affinity=0.949, w_input=0.0,
                                    affinity_mode='exp_abs', affinity_scale=1.0,
                                    batch_size=1000, num_bins=50, pos_threshold=None):
    pic50 = final_train_data['pic50']
    input_feats = final_train_data['input_feats']
    prot_sim_mat = final_train_data['prot_sim']
    local_idx = final_train_data['local_protein_idx'].cpu().numpy()
    N = pic50.shape[0]

    # Initialize histogram accumulator
    hist = np.zeros(num_bins)
    bin_edges = np.linspace(0,1,num_bins+1)  # similarity is always between 0 and 1

    # Go over batches
    for i in range(0, N, batch_size):
        end_i = min(i+batch_size, N)
        pic50_i = pic50[i:end_i]
        input_i = input_feats[i:end_i]
        idx_i = local_idx[i:end_i]

        for j in range(i, N, batch_size):
            end_j = min(j+batch_size, N)
            pic50_j = pic50[j:end_j]
            input_j = input_feats[j:end_j]
            idx_j = local_idx[j:end_j]

            # Affinity similarity
            diff = torch.abs(pic50_i.view(-1,1) - pic50_j.view(1,-1))
            if affinity_mode=='exp_abs':
                affinity_sim = torch.exp(-diff/affinity_scale)
            elif affinity_mode=='exp_squared':
                affinity_sim = torch.exp(-(diff**2)/affinity_scale)

            # Input similarity
            input_sim = 1/(1+torch.cdist(input_i,input_j,p=2))

            # Protein similarity
            prot_sim_vals = prot_sim_mat[np.ix_(idx_i, idx_j)]
            prot_sim = torch.tensor(prot_sim_vals,dtype=torch.float32)

            # Combined similarity
            combined_sim = (prot_sim**w_protein) * (affinity_sim**w_affinity) * (input_sim**w_input)
            
            # Upper triangle if same batch, else flatten
            if i==j:
                triu_idx = torch.triu_indices(end_i-i,end_j-j,offset=1)
                combined_sim = combined_sim[triu_idx[0],triu_idx[1]]
            else:
                combined_sim = combined_sim.flatten()

            # Update histogram
            h, _ = np.histogram(combined_sim.cpu().numpy(), bins=bin_edges)
            hist += h

    # Determine threshold if not given
    if pos_threshold is None:
        cumulative_counts = np.cumsum(hist)
        total_counts = cumulative_counts[-1]
        idx = np.searchsorted(cumulative_counts, total_counts//2)
        pos_threshold = bin_edges[idx]

    # Estimate pos/neg weights
    total = hist.sum()
    n_pos = hist[bin_edges[:-1] >= pos_threshold].sum()
    n_neg = hist[bin_edges[:-1] < pos_threshold].sum()
    pos_weight = total / (2*n_pos) if n_pos>0 else 1.0
    neg_weight = total / (2*n_neg) if n_neg>0 else 1.0

    return pos_weight, neg_weight, hist, bin_edges
