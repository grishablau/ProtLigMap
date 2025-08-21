
from pathlib import Path
from typing import Dict
import numpy as np
from loguru import logger
from src.utils.file_system import abs_path, check_directory_exists, skip_if_condition


def load_project_data(config: Dict[str, str]) -> Dict[str, object]:
    root = check_directory_exists(config["matrices_data_root"])
    mid_outputs = check_directory_exists(config["mid_outputs_dir"])

    pic50_values_file = skip_if_condition(config["pic50_values_file"], check="existence", check_folder_or_file="file")
    protein_fingerprint_file = skip_if_condition(config["protein_fingerprint_file"], check="existence", check_folder_or_file="file")
    embedding_dir = check_directory_exists(config["embedding_dir"])
    embedding_files = sorted(embedding_dir.glob("*/*.npz"))

    model_outputs = check_directory_exists(config["model_outputs"])
    results = check_directory_exists(config["results"])
    log_file = skip_if_condition(config["log_file"], check="existence", check_folder_or_file="file")


    # Load core model input data
    data_npz_path = abs_path(config["data_npz"])
    logger.info(data_npz_path)
    if not data_npz_path.exists():
        raise FileNotFoundError(f"Preprocessed data file not found: {data_npz_path}. Run `gather_project_data()` first.")

    logger.info(f"Loading preprocessed data from {data_npz_path}")
    data = np.load(data_npz_path)

    return {
        'root': root,
        'pic50_values_file': pic50_values_file,
        'protein_fingerprint_file': protein_fingerprint_file,
        'embedding_dir': embedding_dir,
        'embedding_files': embedding_files,
        'model_outputs': model_outputs,
        'results': results,
        'log_file': log_file,
        'mid_outputs': mid_outputs,
        'embedding_matrix': data['embeddings'],
        'pic50_vals': data['pic50_vals'],
        'protein_vecs': data['protein_vecs'],
        'valid_keys': data['valid_keys']
    }
