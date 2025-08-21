import os
from ..initialise import initialise_proj, ProteinLigandDataset
import torch


def load_train_data_components(dataset_path):
    initialise_proj()
    data = torch.load(os.path.abspath(dataset_path), weights_only=False)

    train_data = data['train']

    pic50 = train_data.pic50
    input_feats = train_data.embeddings.cpu()
    prot_sim = train_data.prot_sim_mat.cpu()
    agg_to_local = train_data.agg_to_local
    idx_to_agg = train_data.idx_to_agg

    # idx maps each aggregated protein index to the local index in prot_sim
    local_protein_idx = torch.tensor([agg_to_local[int(a)] for a in idx_to_agg]).cpu()

    return {
        "pic50": pic50,
        "input_feats": input_feats,
        "prot_sim": prot_sim,
        "agg_to_local": agg_to_local,
        "idx_to_agg": idx_to_agg,
        "local_protein_idx": local_protein_idx
    }



