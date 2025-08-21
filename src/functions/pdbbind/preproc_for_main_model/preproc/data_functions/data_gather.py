
from pathlib import Path
import numpy as np
from typing import Dict
from tqdm import tqdm

from src.utils.file_system import abs_path, check_directory_exists, check_file_exists
from loguru import logger

def gather_project_data(config: Dict[str, str]) -> None:
    pic50_values_file = check_file_exists(config['pic50_values_file'])
    logger.info(pic50_values_file)
    protein_fingerprint_file = check_file_exists(config['protein_fingerprint_file'])
    embedding_dir = check_directory_exists(config['embedding_dir'])
    mid_outputs = abs_path(config['mid_outputs_dir'])
    mid_outputs.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading pIC50 values from {pic50_values_file}")
    pic50_npz = np.load(pic50_values_file)

    logger.info(f"Loading protein fingerprints from {protein_fingerprint_file}")
    protein_npz = np.load(protein_fingerprint_file)
    fingerprints = protein_npz["fingerprints"]
    pdb_codes = protein_npz["pdb_codes"]
    if isinstance(pdb_codes[0], bytes):
        pdb_codes = np.char.decode(pdb_codes, encoding="utf-8")
    pdb_code_to_vec = {code: vec for code, vec in zip(pdb_codes, fingerprints)}

    embedding_files = sorted(embedding_dir.glob("*/*.npz"))

    embedding_list = []
    pic50_vals = []
    protein_vecs = []
    valid_keys = []

    logger.info(f"Validating {len(embedding_files)} embeddings...")
    for file in tqdm(embedding_files, desc="Validating embeddings"):
        key = file.stem
        pdb_id = key[:4]

        try:
            data = np.load(file)
            embedding = data["feats"]
            if embedding.ndim > 1:
                embedding = embedding.flatten()
        except Exception as e:
            logger.warning(f"Failed to load embedding {file}: {e}")
            continue

        if key not in pic50_npz:
            logger.error(f"Missing pIC50 value for {key}. Skipping.")
            continue
        val = pic50_npz[key]
        if isinstance(val, np.ndarray) and val.size == 1:
            val = float(val.item())
        elif np.isscalar(val):
            val = float(val)
        else:
            logger.error(f"Invalid pIC50 format for {key}. Skipping.")
            continue

        vec = pdb_code_to_vec.get(pdb_id)
        if vec is None:
            logger.error(f"Missing protein fingerprint for pdb_id {pdb_id}. Skipping.")
            continue

        embedding_list.append(embedding)
        pic50_vals.append(val)
        protein_vecs.append(vec)
        valid_keys.append(key)

    logger.info(f"Total valid samples: {len(valid_keys)}")

    embedding_matrix = np.array(embedding_list)
    pic50_vals = np.array(pic50_vals, dtype=np.float32)
    protein_vecs = np.array(protein_vecs, dtype=np.float32)
    valid_keys = np.array(valid_keys)

    if not (len(embedding_matrix) == len(pic50_vals) == len(protein_vecs)):
        raise ValueError(f"Inconsistent data lengths: embeddings={len(embedding_matrix)}, pic50={len(pic50_vals)}, proteins={len(protein_vecs)}")

    if np.isnan(protein_vecs).any() or np.isinf(protein_vecs).any():
        raise ValueError("Protein vectors contain NaNs or Infs.")
    if np.isnan(embedding_matrix).any() or np.isinf(embedding_matrix).any():
        raise ValueError("Embeddings contain NaNs or Infs.")
    if np.isnan(pic50_vals).any() or np.isinf(pic50_vals).any():
        logger.warning("pIC50 values contain NaNs or Infs, filling with mean.")
        pic50_vals = np.where(np.isnan(pic50_vals) | np.isinf(pic50_vals), np.nanmean(pic50_vals), pic50_vals)

    out_path = mid_outputs / "data_raw.npz"
    np.savez(out_path,
             embeddings=embedding_matrix,
             pic50_vals=pic50_vals,
             protein_vecs=protein_vecs,
             valid_keys=valid_keys)
    logger.info(f"Saved gathered data to {out_path}")