from torch.utils.data import Dataset
from pathlib import Path
import torch
import sys
import os


class ProteinLigandDataset(Dataset):
    def __init__(self, embeddings, pic50, prot_sim_mat, valid_keys, idx_to_agg, agg_to_local, pic50_raw, prot_idx, prot_raw_agg_filtered_into_model):

        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.pic50 = torch.tensor(pic50, dtype=torch.float32)
        self.prot_sim_mat = torch.tensor(prot_sim_mat, dtype=torch.float32)
        self.valid_keys = valid_keys
        self.idx_to_agg = torch.tensor(idx_to_agg, dtype=torch.float32)
        self.agg_to_local = agg_to_local
        self.pic50_raw = torch.tensor(pic50_raw, dtype=torch.float32)
        self.prot_idx = torch.tensor(prot_idx, dtype=torch.int64)
        self.prot_raw_agg_filtered_into_model = torch.tensor(prot_raw_agg_filtered_into_model, dtype=torch.float32)

    def __len__(self):
        return len(self.pic50)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.pic50[idx], self.idx_to_agg[idx]
    

def initialise_proj():


    torch.serialization.add_safe_globals([ProteinLigandDataset])

    project_root = Path(__file__).resolve().parent.parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.chdir(project_root)
    print(f"[Initialise] Project root set to: {project_root}")
    print(f"[Initialise] Current working dir now: {Path.cwd()}")


