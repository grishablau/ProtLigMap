import torch
import numpy as np

def compute_pos_neg_weights_stream(final_train_data, 
                                   w_protein=0.051, w_affinity=0.949, w_input=0.0,
                                   affinity_mode='exp_abs', affinity_scale=1.0,
                                   batch_size=1000, pos_threshold=0.5, verbose=True):
    """
    Compute positive and negative pair weights in a memory-efficient streaming way.
    
    Returns:
        pos_weight, neg_weight: weights for positives and negatives
        approx_pos_threshold: the threshold used to separate pos/neg
    """
    pic50 = final_train_data['pic50']
    input_feats = final_train_data['input_feats']
    prot_sim_mat = final_train_data['prot_sim']
    local_idx = final_train_data['local_protein_idx'].cpu().numpy()
    
    N = pic50.shape[0]
    
    # If pos_threshold is None, estimate it using a small random sample
    if pos_threshold is None:
        sample_idx = np.random.choice(N, min(2000, N), replace=False)
        combined_samp = []

        for i in sample_idx:
            for j in sample_idx:
                if i >= j:
                    continue
                # Protein similarity
                prot_sim = prot_sim_mat[local_idx[i], local_idx[j]].item()
                # Affinity similarity
                diff = abs((pic50[i]-pic50[j]).item())
                if affinity_mode == 'exp_abs':
                    affinity_sim = 1/(1+diff)
                else:
                    affinity_sim = np.exp(-(diff**2)/affinity_scale)
                # Input similarity
                input_dist = torch.norm(input_feats[i]-input_feats[j]).item()
                input_sim = 1/(1+input_dist)
                # Weighted combination
                combined = (prot_sim**w_protein)*(affinity_sim**w_affinity)*(input_sim**w_input)
                combined_samp.append(combined)
        pos_threshold = float(np.median(combined_samp))
        if verbose:
            print(f"Estimated pos_threshold (median of sample): {pos_threshold:.4f}")
    
    # Streaming counts
    n_pos = 0
    n_neg = 0
    total_pairs = 0
    all_neg_sims = []
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

            # Protein similarity
            prot_sim_vals = prot_sim_mat[np.ix_(idx_i, idx_j)]
            prot_sim = torch.tensor(prot_sim_vals, dtype=torch.float32)

            # Affinity similarity
            pic50_i_exp = pic50_i.view(-1,1)
            pic50_j_exp = pic50_j.view(1,-1)
            diff = torch.abs(pic50_i_exp - pic50_j_exp)
            if affinity_mode=='exp_abs':
                affinity_sim = 1/(1+diff)
            else:
                affinity_sim = torch.exp(-(diff**2)/affinity_scale)

            # Input similarity
            dist_matrix = torch.cdist(input_i, input_j, p=2)
            input_sim = 1/(1+dist_matrix)

            # Weighted combined similarity
            combined_sim = (prot_sim**w_protein)*(affinity_sim**w_affinity)*(input_sim**w_input)

            # Only upper-triangle for same-batch
            if i==j:
                triu_idx = torch.triu_indices(end_i-i, end_j-j, offset=1)
                combined_sim = combined_sim[triu_idx[0], triu_idx[1]]

            # Count pos/neg
            n_pos += torch.sum(combined_sim >= pos_threshold).item()
            n_neg += torch.sum(combined_sim < pos_threshold).item()
            total_pairs += combined_sim.numel()

            neg_sims_batch = combined_sim[combined_sim < pos_threshold]
            all_neg_sims.append(neg_sims_batch)


    pos_weight = total_pairs/(2*n_pos) if n_pos>0 else 1.0
    neg_weight = total_pairs/(2*n_neg) if n_neg>0 else 1.0


    all_neg_sims = torch.cat(all_neg_sims)
    neg_median = all_neg_sims.median().item()

    if verbose:
        print(f"Streaming pos_weight: {pos_weight:.3f}, neg_weight: {neg_weight:.3f}")
        print(f"Median of negative pairs: {neg_median:.4f}")

    return pos_weight, neg_weight, pos_threshold, neg_median
