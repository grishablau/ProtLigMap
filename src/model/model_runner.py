import csv
import logging
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from src.utils.model_utils.initialise import initialise_proj, ProteinLigandDataset
from src.utils.model_utils.visualization.plotting import ManifoldVisualizer


# ===== Logging Setup =====
log_file = "logs/model/training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
)
logging.info(f"Logging started. Logs saved to {log_file}")

EPS = 1e-6


# # optional
# # storage for all stats
# batch_stats = []
# csv_file = Path("embed_stats.csv")

# if not csv_file.exists():
#     with open(csv_file, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["epoch", "batch", "min", "max", "mean", "std", "frac_pos"])
#         writer.writeheader()

# def log_embed_stats(embed_sim, epoch, batch_idx, pos_threshold=0.73):
#     stats = {
#         "epoch": epoch,
#         "batch": batch_idx,
#         "min": embed_sim.min().item(),
#         "max": embed_sim.max().item(),
#         "mean": embed_sim.mean().item(),
#         "std": embed_sim.std().item(),
#         "frac_pos": (embed_sim >= pos_threshold).float().mean().item(),
#     }
#     with open(csv_file, "a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=stats.keys())
#         writer.writerow(stats)




# ===== Contrastive Network =====
class ContrastiveNet(nn.Module):
    def __init__(self, input_dim=155, manifold_dim=128,
                  contrastive_dim=64):
        super().__init__()
         # Single encoder for joint protein+ligand embeddings 
        self.encoder = nn.Sequential( 
            nn.Linear(input_dim, 256), 
            nn.ReLU(), 
            nn.LayerNorm(256), 
            nn.Dropout(0.3), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.LayerNorm(128), 
            nn.Dropout(0.1), 
            nn.Linear(128, manifold_dim), 
            nn.ReLU(), 
            nn.LayerNorm(manifold_dim), )
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(manifold_dim, contrastive_dim),
            nn.ReLU(),
            nn.Linear(contrastive_dim, contrastive_dim),
            nn.LayerNorm(contrastive_dim),
        )

    def forward(self, x):
        manifold_emb = self.encoder(x)
        manifold_emb = F.normalize(manifold_emb, dim=-1)  # stabilize manifold embeddings

        proj_emb = self.projection_head(manifold_emb)
        proj_emb = F.normalize(proj_emb, dim=-1)  # stabilize contrastive space

        return manifold_emb, proj_emb



# ===== Similarities =====
def compute_similarities(
    pic50_std, idx_to_agg, prot_sim_mat, input_features,
    w_protein, w_affinity, w_input, agg_to_local_map
):
# Affinity similarity (pic50)
    pic50_i = pic50_std.view(-1, 1)
    pic50_j = pic50_std.view(1, -1)
    diff = torch.abs(pic50_i - pic50_j)
    affinity_sim = 1.0 / (1.0 + diff)


    # Protein similarity (if used)
    idx = torch.tensor([agg_to_local_map[int(a)] for a in idx_to_agg], device=prot_sim_mat.device, dtype=torch.long)
    protein_sim = prot_sim_mat[idx[:, None], idx[None, :]]


    # Input feature similarity (L2 → similarity)
    dist_matrix = torch.cdist(input_features, input_features, p=2)
    input_sim = 1.0 / (1.0 + dist_matrix)


    # Log-space weighted geometric mean
    denom = max(w_protein + w_affinity + w_input, EPS)
    log_sim = (
        w_protein * torch.log(protein_sim)
        + w_affinity * torch.log(affinity_sim)
        + w_input * torch.log(input_sim)
    ) / denom
    combined_sim = torch.exp(log_sim)

    return combined_sim

# ===== Loss =====
def calc_loss(
    embeddings, pic50_std, idx_to_agg, prot_sim_mat, input_features,
    w_protein, w_affinity, w_input, agg_to_local_map, batch_idx, epoch,
    margin_pos=0.95, margin_neg=0.99,
    pos_weight=1.0, neg_weight=1.0,
    pos_threshold=0.5, gamma=1.0,
    top_k_negatives=None
):
    
    """
    Contrastive loss with optional hard-negative mining.
    """
    n = embeddings.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=embeddings.device), None

    # Compute pairwise embedding similarities
    # Normalize embeddings (optional, but helps numerical stability)

    # Compute pairwise Euclidean distances
    dist = torch.cdist(embeddings, embeddings, p=2)

    # Convert distance to similarity
    embed_sim = 1.0 / (1.0 + dist)  # now similarity in (0,1]

    
    # log_embed_stats(embed_sim, epoch, batch_idx)
    batch_idx+=1
    # Compute target similarities
    combined_sim = compute_similarities(
        pic50_std, idx_to_agg, prot_sim_mat, input_features,
        w_protein, w_affinity, w_input, agg_to_local_map
    )
    if gamma != 1.0:
        combined_sim = combined_sim.pow(gamma)




    # Masks for positive and negative pairs
    pos_mask = (combined_sim >= pos_threshold).float()
    neg_mask = 1.0 - pos_mask

    # Hard-negative mining
    if top_k_negatives is not None:
        hardest_mask = torch.zeros_like(neg_mask)
        for i in range(n):
            # Only consider valid negatives for this anchor
            valid_neg_idx = (neg_mask[i] > 0).nonzero(as_tuple=True)[0]
            if len(valid_neg_idx) == 0:
                continue

            # Select top-k hardest negatives based on embed similarity
            neg_sims = embed_sim[i, valid_neg_idx]
            k = min(top_k_negatives, len(valid_neg_idx))
            topk_idx_in_valid = torch.topk(neg_sims, k=k).indices
            topk_idx = valid_neg_idx[topk_idx_in_valid]
            hardest_mask[i, topk_idx] = 1.0

        # Update negative mask to include only hardest negatives
        neg_mask = neg_mask * hardest_mask

    # Compute contrastive loss
    pos_term = pos_weight * pos_mask * combined_sim * F.relu(margin_pos - embed_sim)
    neg_term = neg_weight * neg_mask * (1.0 - combined_sim) * F.relu(embed_sim - margin_neg)

    # Ignore diagonal elements
    mask_offdiag = 1.0 - torch.eye(n, device=embeddings.device)
    loss_matrix = (pos_term + neg_term) * mask_offdiag
    loss = loss_matrix.sum() / float(max(n * (n - 1), 1))

    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=embeddings.device)

    return loss, combined_sim


# ===== Logging =====
def log_similarity_stats(embed_sim, combined_sim, epoch, tag, margin_pos, margin_neg, pos_threshold):
    with torch.no_grad():
        # Flatten off-diagonal elements if square
        if combined_sim.dim() == 2 and combined_sim.size(0) == combined_sim.size(1):
            n = combined_sim.size(0)
            mask_offdiag = 1.0 - torch.eye(n, device=combined_sim.device)
            combined_vals = (combined_sim * mask_offdiag).flatten()
        else:
            combined_vals = combined_sim.flatten()

        if embed_sim.dim() == 2 and embed_sim.size(0) == embed_sim.size(1):
            n = embed_sim.size(0)
            mask_offdiag = 1.0 - torch.eye(n, device=embed_sim.device)
            embed_vals = (embed_sim * mask_offdiag).flatten()
        else:
            embed_vals = embed_sim.flatten()

        # Log both margins
        logging.info(
            f"[{tag}] Epoch {epoch} | embed μ={embed_vals.mean():.4f} σ={embed_vals.std():.4f} "
            f"| combined μ={combined_vals.mean():.4f} σ={combined_vals.std():.4f} "
            f"| frac_pos(target)={(combined_vals>=pos_threshold).float().mean():.3f} "
            f"frac_pos(embed)={(embed_vals>=pos_threshold).float().mean():.3f} "
            f"| margin_pos={margin_pos:.4f} margin_neg={margin_neg:.4f}"
        )



def run_epoch(
    dataloader, model, optimizer, device, epoch, total_epochs, tag,
    margin_pos, margin_neg,  # <- changed
    w_protein, w_affinity, w_input, prot_sim_mat, agg_to_local_map,
    pos_weight, neg_weight, pos_threshold, gamma,
    max_sample=5000, reg_embed_std=1e-4
):

    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    all_proj, all_manifold = [], []
    all_pic50, all_idx = [], []
    batch_losses = []
    for batch in dataloader:
        emb, pic50, idx_to_agg = batch
        emb, pic50, idx_to_agg = emb.to(device), pic50.to(device), idx_to_agg.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            manifold_emb, proj_emb = model(emb)
            batch_idx = 0
            loss, _ = calc_loss(
                proj_emb, pic50, idx_to_agg, prot_sim_mat, emb,
                w_protein, w_affinity, w_input, agg_to_local_map, batch_idx, epoch,
                margin_pos=margin_pos, margin_neg=margin_neg,
                pos_weight=pos_weight, neg_weight=neg_weight,
                pos_threshold=pos_threshold, gamma=gamma,
                top_k_negatives=500  # <-- only hardest 5 negatives per anchor
            )
        

            # small regularization to avoid embedding collapse
            if reg_embed_std > 0:
                std_loss = reg_embed_std * (1.0 - torch.std(proj_emb, dim=0)).mean()
                loss = loss + std_loss


            if is_train and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        batch_losses.append(loss.detach().cpu())
        all_proj.append(proj_emb.detach().cpu())
        all_manifold.append(manifold_emb.detach().cpu())
        all_pic50.append(pic50.detach().cpu())
        all_idx.append(idx_to_agg.detach().cpu())

    # Concatenate full epoch embeddings
    proj_emb_epoch = torch.cat(all_proj)
    manifold_emb_epoch = torch.cat(all_manifold)
    pic50_epoch = torch.cat(all_pic50)
    idx_epoch = torch.cat(all_idx)

    # Sample subset for epoch-level stats
    n_total = proj_emb_epoch.size(0)
    if n_total > max_sample:
        sample_idx = torch.randperm(n_total)[:max_sample]
        proj_emb_sample = proj_emb_epoch[sample_idx]
        manifold_emb_sample = manifold_emb_epoch[sample_idx]
        pic50_sample = pic50_epoch[sample_idx]
        idx_sample = idx_epoch[sample_idx]
    else:
        proj_emb_sample = proj_emb_epoch
        manifold_emb_sample = manifold_emb_epoch
        pic50_sample = pic50_epoch
        idx_sample = idx_epoch

    # Compute target similarity for sampled subset
    target_sim_sample = compute_similarities(
        pic50_sample, idx_sample, prot_sim_mat, proj_emb_sample,
        w_protein, w_affinity, w_input, agg_to_local_map
    )
    if gamma != 1.0:
        target_sim_sample = target_sim_sample.pow(gamma)


    # Embedding similarities
    dist_raw = torch.cdist(proj_emb_sample, proj_emb_sample)
    embed_sim = 1.0 / (1.0 + dist_raw)


    dist_raw_m = torch.cdist(manifold_emb_sample, manifold_emb_sample)
    embed_sim_m = 1.0 / (1.0 + dist_raw_m)



    # Log stats
    log_similarity_stats(embed_sim, target_sim_sample, epoch, tag, margin_pos, margin_neg, pos_threshold)
    log_similarity_stats(embed_sim_m, target_sim_sample, epoch, f"{tag}_MANIFOLD", margin_pos, margin_neg, pos_threshold)


    # Spearman correlation (off-diagonal only)
    spearman_corr = np.nan
    # ===== Robust Spearman correlation =====
    try:
        proj_emb_np = proj_emb_sample.numpy()
        n = proj_emb_np.shape[0]

        # Compute condensed Euclidean distances
        dist_vec = pdist(proj_emb_np, metric="euclidean")

        # Extract upper triangle of target similarity to match pdist ordering
        target_sim_mat = target_sim_sample.numpy()
        target_sim_condensed = target_sim_mat[np.triu_indices(n, k=1)]

        # Optional logging
        logging.info(f"dist_vec std={np.std(dist_vec):.6f}, target_sim std={np.std(target_sim_condensed):.6f}")
        logging.info(f"min/max dist_vec: {dist_vec.min()}/{dist_vec.max()}, min/max target_sim_vec: {target_sim_condensed.min()}/{target_sim_condensed.max()}")

        # Keep only finite entries
        mask = np.isfinite(dist_vec) & np.isfinite(target_sim_condensed)

        # Minimum variance threshold
        min_var = 1e-6
        if mask.sum() > 0 and np.std(dist_vec[mask]) > min_var and np.std(target_sim_condensed[mask]) > min_var:
            spearman_corr = spearmanr(-dist_vec[mask], target_sim_condensed[mask]).correlation
        else:
            spearman_corr = np.nan

    except Exception as e:
        logging.warning(f"Spearman computation failed: {e}")
        spearman_corr = np.nan


        # Compute mean loss over all batches
    mean_loss = float(torch.stack(batch_losses).mean())

    # Log a concise summary per epoch
    logging.info(
        f"[{tag}] Epoch {epoch}/{total_epochs} | "
        f"Loss: {mean_loss:.4f} | "
        f"Spearman: {spearman_corr:.4f} | "
        f"Embed μ={proj_emb_sample.mean():.4f} σ={proj_emb_sample.std():.4f} | "
        f"Frac pos(embed)={(embed_sim > pos_threshold).float().mean():.3f}"
    )


    return mean_loss, spearman_corr, margin_pos, margin_neg


# ===== Training Procedure =====
def run_model(
    data, results_folder, outputs_folder, load_visualizer,
    num_epochs, save_every, w_protein, w_affinity, w_input, margin_pos, margin_neg,
    pos_weight, neg_weight, pos_threshold, gamma
):
    results_folder, results_folder_pca = Path(results_folder), Path(results_folder)/"pca"
    outputs_folder = Path(outputs_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ContrastiveNet(data["train"].embeddings.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

    train_loader = DataLoader(data["train"], batch_size=1000, shuffle=True, num_workers=0)
    val_loader = DataLoader(data["val"], batch_size=1000, shuffle=False, num_workers=0)

    prot_sim_train = data["train"].prot_sim_mat.to(device)
    prot_sim_val = data["val"].prot_sim_mat.to(device)

    # Visualizer
    if load_visualizer:
        with open(outputs_folder / "intermidiate_objects/model/visualization/saved_visualizer.pkl", "rb") as f:
            visualizer = pickle.load(f)
    else:
        visualizer = ManifoldVisualizer(data, k_means_prot_cluster_list=[5], n_pic50_bins=4)
        with open(outputs_folder / "intermidiate_objects/model/visualization/saved_visualizer.pkl", "wb") as f:
            pickle.dump(visualizer, f)

    for epoch in range(1, num_epochs+1):
        # optional, if you want to visualize the first epoch
        if epoch == 1:
            visualizer.visualize(model=model, device=device, embedding_type="orig", tag=f"epoch{epoch}", output_dir=results_folder_pca)
        mean_loss, spearman_corr, _, _ = run_epoch(
            train_loader, model, optimizer, device, epoch, num_epochs, "TRAIN",
            margin_pos, margin_neg,
            w_protein, w_affinity, w_input, prot_sim_train, data["train"].agg_to_local,
            pos_weight, neg_weight, pos_threshold, gamma
        )

        if epoch < 5:
            scheduler.step()
            logging.info(f"Epoch {epoch} | LR = {scheduler.get_last_lr()[0]:.6f}")

        # Validation
        if epoch % save_every == 0 or epoch in [1, num_epochs]:
            _ , spearman_corr , _, _ = run_epoch(
                val_loader, model, None, device, epoch, num_epochs, "VAL",
                margin_pos, margin_neg,
                w_protein, w_affinity, w_input, prot_sim_val, data["val"].agg_to_local,
                pos_weight, neg_weight, pos_threshold, gamma
            )      
            
            
            if spearman_corr>0.38:
                visualizer.visualize(model=model, device=device, embedding_type="proj", tag=f"epoch{epoch}", output_dir=results_folder_pca)
                results_folder.mkdir(parents=True, exist_ok=True)
                ckpt = {"epoch": epoch, "model_state_dict": model.state_dict()}
                torch.save(ckpt, results_folder/f"checkpoints/checkpoint_epoch_{epoch}_spearman_corr_{spearman_corr}.pt")

            # if spearman_corr>0.95:
                # visualizer.visualize(model=model, device=device, embedding_type="orig", tag=f"epoch{epoch}", output_dir=results_folder_pca)
                # visualizer.visualize(model=model, device=device, embedding_type="manifold", tag=f"epoch{epoch}", output_dir=results_folder_pca)
      
    logging.info("Training complete.")

# ===== Execution =====
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.getcwd(), ""))
    results_folder = os.path.join(project_root, "results", "manifold_model")
    outputs_folder = os.path.join(project_root, "outputs")
    dataset_path = os.path.join(outputs_folder, "datasets.pt")
    
    
    initialise_proj()
    data = torch.load(dataset_path, weights_only=False)

    run_model(
        data,
        results_folder=results_folder,
        outputs_folder=outputs_folder,
        load_visualizer=False,
        num_epochs=2000,
        save_every=1,
        w_protein=0.051,
        w_affinity=0.949,
        # w_protein=0.6,
        # w_affinity=0.4,
        w_input=0,
        pos_weight=1.224,
        neg_weight=0.845,
        pos_threshold=0.55,
        gamma=1,
        margin_pos = 0.6,
        margin_neg = 0.5
    )
