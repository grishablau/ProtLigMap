from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser, PPBuilder
import torch
import esm

from src.utils.file_system import check_directory_exists
from loguru import logger

# === Load ESM-2 model ===
logger.info("Loading ESM-2 model...")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

# === Sequence to ESM embedding ===
def compute_esm_embedding(seq: str) -> np.ndarray:
    data = [("protein", seq)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
    token_representations = results["representations"][6]
    # Mean pooling (excluding [CLS] and [EOS])
    embedding = token_representations[0, 1:-1].mean(0)
    return embedding.cpu().numpy()

def generate_protein_fingerprints(config: Dict[str, str]) -> None:
    """
    Generates protein fingerprints by computing ESM-2 embeddings for sequences
    derived from cleaned/prepared PDB files.

    This function:
    - Reads a list of PDB codes from an index file.
    - For each valid PDB code, finds a corresponding protein structure file in `merge_exp_dir`.
    - Extracts the amino acid sequence from the structure using Biopython.
    - Computes an ESM-2 embedding (mean-pooled over residues) for the sequence.
    - Saves the resulting fingerprint array along with matching PDB codes to a compressed `.npz` file.
    - Logs any issues (missing files, parsing errors, empty sequences) to a separate log file.

    Args:
        config (Dict[str, str]): Dictionary containing required paths:
            - "index_file": Path to a file listing all candidate PDB codes (one per line).
            - "merge_exp_dir": Directory containing subfolders for each PDB code with protein PDB files.
            - "output_file": Path to the `.npz` file where fingerprints and PDB codes will be saved.
            - "log_file": Path to the log file where any errors or skipped PDBs will be recorded.

    Notes:
        - Expected PDB file names include: 
            `{pdb_code}_protein_cleaned.pdb`, `{pdb_code}_protein_prepared.pdb`, or `{pdb_code}_protein.pdb`.
        - If multiple files exist, the first match in that order is used.
        - Uses the ESM-2 model `esm2_t6_8M_UR50D` for embedding generation.
        - Embeddings are mean-pooled over all residues (excluding [CLS] and [EOS] tokens).
        - Errors such as missing files or invalid sequences are logged but do not interrupt execution.
    """

    index_file = Path(config["index_file"])
    merge_exp_dir = check_directory_exists(config["merge_exp_dir"])
    output_file = Path(config["output_file"])
    log_file = Path(config["log_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # === Get valid pdb codes ===
    pdb_codes = []
    with open(index_file, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            code = line.strip().split()[0].lower()
            if (merge_exp_dir / code).is_dir():
                pdb_codes.append(code)

    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()

    fingerprints = []
    valid_pdb_codes = []
    errors = []

    for pdb_code in tqdm(pdb_codes, desc="Processing proteins"):
        pdb_folder = merge_exp_dir / pdb_code
        for fname in [f"{pdb_code}_protein_cleaned.pdb", f"{pdb_code}_protein_prepared.pdb", f"{pdb_code}_protein.pdb"]:
            pdb_path = pdb_folder / fname
            if pdb_path.exists():
                break
        else:
            errors.append(f"[No PDB file] {pdb_code}")
            continue

        try:
            structure = parser.get_structure(pdb_code, pdb_path)
            sequence = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(structure))

            if not sequence:
                errors.append(f"[No sequence] {pdb_code}")
                continue

            embedding = compute_esm_embedding(sequence)
            fingerprints.append(embedding)
            valid_pdb_codes.append(pdb_code)

        except Exception as e:
            errors.append(f"[Parse error] {pdb_code}: {str(e)}")

    # === Save as .npz ===
    fingerprints_np = np.array(fingerprints, dtype=np.float32)
    np.savez_compressed(output_file, fingerprints=fingerprints_np, pdb_codes=np.array(valid_pdb_codes))

    # === Log errors ===
    with open(log_file, 'w') as f:
        for err in errors:
            f.write(err + '\n')

    logger.info(f"Saved {len(valid_pdb_codes)} protein ESM fingerprints to {output_file}")
    logger.info(f"Logged {len(errors)} issues to {log_file}")
