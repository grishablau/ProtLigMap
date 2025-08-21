
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any
from loguru import logger


class ProteinLigandDataset(Dataset):
    def __init__(
        self,
        embedding_matrix,
        protein_vecs,
        pic50_vals,
        protein_class,
        valid_keys
    ):
        self.embeddings = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.proteins = torch.tensor(protein_vecs, dtype=torch.float32)
        self.pic50 = torch.tensor(pic50_vals, dtype=torch.float32)
        self.cluster = torch.tensor(protein_class, dtype=torch.float32)
        self.pdbid = list(valid_keys)

    def __len__(self):
        return len(self.pic50)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.proteins[idx], self.pic50[idx], self.cluster[idx], self.pdbid[idx]

def place_holding(data: Dict[str, Dict[str, Any]]) -> Dict[str, ProteinLigandDataset]:
    logger.info("Preparing DataLoaders for splits")
    splits = list(next(iter(data.values())).keys())

    datasets = {}
    for split in splits:
        logger.info(f"Processing split: {split}")
        split_data = {key: val[split] for key, val in data.items()}
        datasets[split] = ProteinLigandDataset(**split_data)

    logger.info("All splits processed into ProteinLigandDataset")
    return datasets
