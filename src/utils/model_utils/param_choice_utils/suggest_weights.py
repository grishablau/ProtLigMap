import torch
import numpy as np

def suggest_weights(components, batch_size=1000, verbose=True, exclude=[]):
    """
    Suggest relative weights for protein, affinity (pic50), and input features
    based on variance of pairwise similarities in the current dataset.
    
    This version processes pairwise similarities in batches (O(N²) total work, 
    but never loads the full NxN matrix into memory).
    
    Args:
        components: dict with keys:
            'pic50', 'input_feats', 'prot_sim', 'local_protein_idx', 'idx_to_agg', 'agg_to_local'
        batch_size: number of samples to process per block (controls memory usage)
        exclude: list of components to exclude, options: 'protein', 'affinity', 'input'
    
    Returns:
        dict: {'w_protein': ..., 'w_affinity': ..., 'w_input': ...}
    """
    vars_dict = {}

    # --- Protein ---
    if 'protein' not in exclude:
        prot_sim = components['prot_sim'].cpu().numpy()
        local_idx = components['local_protein_idx'].cpu().numpy()
        N = len(local_idx)
        mean_val, mean_sq_val, count = 0.0, 0.0, 0

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
                mean_val += sim_vals.sum()
                mean_sq_val += np.square(sim_vals).sum()
                count += sim_vals.size

        mean_val /= count
        mean_sq_val /= count
        vars_dict['protein'] = mean_sq_val - mean_val**2
    else:
        vars_dict['protein'] = 0.0

    # --- Affinity ---
    if 'affinity' not in exclude:
        pic50 = components['pic50'].float().cpu()
        N = pic50.shape[0]
        mean_val, mean_sq_val, count = 0.0, 0.0, 0

        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            Xi = pic50[i:end_i]
            for j in range(i, N, batch_size):
                end_j = min(j + batch_size, N)
                Xj = pic50[j:end_j]
                diff = torch.abs(Xi.view(-1, 1) - Xj.view(1, -1))
                sim = (1 / (1 + diff)).numpy()
                if i == j:
                    triu_idx = np.triu_indices(end_i - i, k=1)
                    sim = sim[triu_idx]
                else:
                    sim = sim.flatten()
                mean_val += sim.sum()
                mean_sq_val += np.square(sim).sum()
                count += sim.size

        mean_val /= count
        mean_sq_val /= count
        vars_dict['affinity'] = mean_sq_val - mean_val**2
    else:
        vars_dict['affinity'] = 0.0

    # --- Input features ---
    if 'input' not in exclude:
        feats = components['input_feats'].float().cpu()
        N = feats.shape[0]
        mean_val, mean_sq_val, count = 0.0, 0.0, 0

        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            Xi = feats[i:end_i]
            for j in range(i, N, batch_size):
                end_j = min(j + batch_size, N)
                Xj = feats[j:end_j]
                dist = torch.cdist(Xi, Xj, p=2)
                sim = (1 / (1 + dist)).numpy()
                if i == j:
                    triu_idx = np.triu_indices(end_i - i, k=1)
                    sim = sim[triu_idx]
                else:
                    sim = sim.flatten()
                mean_val += sim.sum()
                mean_sq_val += np.square(sim).sum()
                count += sim.size

        mean_val /= count
        mean_sq_val /= count
        vars_dict['input'] = mean_sq_val - mean_val**2
    else:
        vars_dict['input'] = 0.0

    # Normalize to sum to 1
    total_var = sum(vars_dict.values())
    if total_var > 0:
        w_protein = vars_dict['protein'] / total_var
        w_affinity = vars_dict['affinity'] / total_var
        w_input = vars_dict['input'] / total_var
    else:
        w_protein = w_affinity = w_input = 1/3

    if verbose:
        print(f"Variance-based suggested weights (exclude={exclude}):")
        print(f"  Protein:  {vars_dict['protein']:.6f} -> w_protein={w_protein:.3f}")
        print(f"  Affinity: {vars_dict['affinity']:.6f} -> w_affinity={w_affinity:.3f}")
        print(f"  Input:    {vars_dict['input']:.6f} -> w_input={w_input:.3f}")

    return {'w_protein': w_protein, 'w_affinity': w_affinity, 'w_input': w_input}
