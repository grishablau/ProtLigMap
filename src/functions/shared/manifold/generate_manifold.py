import os
import torch
from pathlib import Path
import pickle
from torch.utils.data import ConcatDataset, DataLoader
# Ensure your src folder is on path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from src.utils.model_utils.initialise import initialise_proj, ProteinLigandDataset

# ===== Model Definition (must match your training code) =====
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveNet(nn.Module):
    def __init__(self, input_dim=155, manifold_dim=128, contrastive_dim=64):
        super().__init__()
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
            nn.LayerNorm(manifold_dim)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(manifold_dim, contrastive_dim),
            nn.ReLU(),
            nn.Linear(contrastive_dim, contrastive_dim),
            nn.LayerNorm(contrastive_dim)
        )

    def forward(self, x):
        manifold_emb = self.encoder(x)
        manifold_emb = F.normalize(manifold_emb, dim=-1)
        proj_emb = self.projection_head(manifold_emb)
        proj_emb = F.normalize(proj_emb, dim=-1)
        return manifold_emb, proj_emb

# ===== Paths =====
project_root = os.path.abspath(os.getcwd())
results_folder = Path(project_root) / "results" / "manifold_model"
outputs_folder = Path(project_root) / "outputs"
dataset_path = outputs_folder / "datasets.pt"
checkpoint_path = results_folder / "checkpoints" / "checkpoint_epoch_28_spearman_corr_0.3859080021352886.pt"  # <- replace with your desired checkpoint

# ===== Load dataset =====
initialise_proj()
data = torch.load(dataset_path, weights_only=False)

# Combine all datasets into one list
all_data = ConcatDataset([data["train"], data["val"]])
all_loader = DataLoader(all_data, batch_size=512, shuffle=False, num_workers=0)

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load model and checkpoint =====
input_dim = data["train"].embeddings.shape[1]
model = ContrastiveNet(input_dim=input_dim).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ===== Extract manifold embeddings =====
all_manifold = []
all_proj = []
with torch.no_grad():
    for batch in all_loader:
        emb, pic50, idx = batch
        emb = emb.to(device)
        manifold_emb, proj_emb = model(emb)
        all_manifold.append(manifold_emb.cpu())
        all_proj.append(proj_emb.cpu())

all_manifold = torch.cat(all_manifold)
all_proj = torch.cat(all_proj)

# ===== Save embeddings =====
save_dir = results_folder / "full_manifold"
save_dir.mkdir(parents=True, exist_ok=True)
torch.save(all_manifold, save_dir / "manifold_embeddings.pt")
torch.save(all_proj, save_dir / "projection_embeddings.pt")

print(f"Manifold saved at {save_dir}")
