import os
import torch
import dgl
import pandas as pd
import numpy as np
from rdkit import Chem
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader
from models.backbone.IGModel.scripts.ligand_features import LigandFeature
from models.backbone.IGModel.scripts.load_receptor import ReceptorFile
from models.backbone.IGModel.scripts.pocket_features import PocketFeatures
from models.backbone.IGModel.scripts.cplx_graph import ComplexGraph
from models.backbone.IGModel.scripts.model import *
from models.backbone.IGModel.scripts.utils import run_an_eval_epoch, sdf_split
import sys
sys.path.append('/media/racah/2b2b05ab-497e-47ab-a698-6e77a3b775c4/grisha/for_ProtLigMap/models/backbone/IGModel/scripts')


class ComplexDataset(Dataset):
    def __init__(self, rec_gs, cplx_gs):
        self.rec_gs = rec_gs
        self.cplx_gs = cplx_gs

    def __getitem__(self, idx):
        return idx, self.rec_gs[idx], self.cplx_gs[idx]

    def __len__(self):
        return len(self.rec_gs)


def load_model(model_path=None, device="cpu"):
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    return model


def collect_all_inputs(base_dir):
    """Collect inputs from a directory structure."""
    input_lists = []
    for subfolder in sorted(os.listdir(base_dir)):
        prefix = subfolder
        full_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(full_path):
            continue
        rec_fpath = os.path.join(full_path, f"{prefix}_protein_prepared.pdb")
        ref_lig_fpath = os.path.join(full_path, f"{prefix}_ligand.sdf")
        pose_fpath = os.path.join(full_path, f"{prefix}_docked.sdf")
        if os.path.exists(rec_fpath) and os.path.exists(ref_lig_fpath) and os.path.exists(pose_fpath):
            input_lists.append([prefix, rec_fpath, ref_lig_fpath, pose_fpath])
    return input_lists


def load_model_outputs(input_list, model=None, device="cpu"):
    """Core function to load inputs and return outputs from the model."""
    prefix, rec_fpath, ref_lig_fpath, pose_fpath = input_list

    if model is None:
        model = load_model(device=device)

    if pose_fpath.endswith("sdf"):
        _, poses_content = sdf_split(pose_fpath)
    else:
        raise ValueError("Pose file must be .sdf format")

    # Create pocket graph
    rec = ReceptorFile(rec_fpath=rec_fpath, ref_lig_fpath=ref_lig_fpath)
    rec.clip_rec()
    rec.define_pocket()
    pock_feat = PocketFeatures(rec, pock_center=rec.pock_center)
    pock_g = pock_feat.pock_to_graph()
    pock_g = dgl.add_self_loop(pock_g)

    keys, cplx_graphs = [], []
    for idx, p in enumerate(poses_content):
        mol = Chem.MolFromMolBlock(p)
        try:
            lig = LigandFeature(mol=mol)
            lig.lig_to_graph()
            cplx = ComplexGraph(rec, lig)
            cplx_graph = cplx.get_cplx_graph()
            cplx_graphs.append(cplx_graph)
            keys.append(f"{prefix}-{idx}")
        except Exception as e:
            print(f"Pose {idx} skipped due to error: {e}")

    if len(keys) == 0:
        raise RuntimeError("No valid complexes found")

    dataset = ComplexDataset([pock_g]*len(keys), cplx_graphs)
    loader = GraphDataLoader(dataset, batch_size=32, shuffle=False)

    pred_rmsd, pred_pkd = run_an_eval_epoch(model, loader, device=device)

    # Run forward pass on first batch for deeper features
    with torch.no_grad():
        idx, rec_g, cplx_g = next(iter(loader))
        rec_g, cplx_g = rec_g.to(device), cplx_g.to(device)
        output = model(rec_g, cplx_g)

    pred_hrmsd, pred_pkd, cplx_lig_feats, feats, W, *_ = output

    return {
        "pred_hrmsd": pred_hrmsd,
        "pred_pkd": pred_pkd,
        "cplx_lig_feats": cplx_lig_feats,
        "feats": feats,
        "W": W,
        "keys": keys
    }

