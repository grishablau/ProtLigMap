import pandas as pd
from src.preproc_data_for_ai.protein_cleaning import protein_proc
from src.preproc_data_for_ai.protein_to_vector import run_protein_parallel
from src.preproc_data_for_ai.standard_type_cleaning import standard_type_filtering
from src.preproc_data_for_ai.std_value import std_value_to_pic50
from src.preproc_data_for_ai.vectorise_smiles_data import run_smiles_parallel
from loguru import logger
import sys
from src.utils.file_system import python_load
import numpy as np

def preproc_data_for_ai(input_dict):
    df = pd.read_parquet(input_dict['data_path'])
    try:
        settings = input_dict['settings']
        settings = (python_load(settings)).processed_settings
    except:
        logger.error("No settings Python file is indicated in YAML pipeline file!")
        sys.exit(1)

    # Step 1: Standard type filter (IC50 / pIC50)
    df = standard_type_filtering(input_dict, df)

    # Step 2: Clean and validate protein sequences
    df = protein_proc(df, input_dict['debug_mode'])

    # Step 3: Convert units to pic50
    df = std_value_to_pic50(df, input_dict['debug_mode'])

    # Step 4: Convert SMILES to Morgan fingerprints (Parallel)
    smiles_vectors = list(run_smiles_parallel(df['canonical_smiles'].tolist(), settings['ligand_emb_model']))
    df['smiles_vector'] = smiles_vectors
# 
    # Step 5: Embed protein sequences using ProtBert (Parallel)
    protein_embeddings = list(run_protein_parallel(df['protein_sequence'].tolist(), settings['prot_emb_model']))
    df['protein_embedding'] = protein_embeddings

    # Step 6: Drop non-numeric columns for training
    df = df.drop(columns=['molecule_chembl_id', 'kinase_name', 'standard_units', 'standard_type', 'canonical_smiles', 'protein_sequence'])

    logger.info("Data preprocessing completed. Final preview:")

    if 'result_save' not in input_dict:
        # Convert numpy arrays to lists before saving
        df['smiles_vector'] = df['smiles_vector'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df['protein_embedding'] = df['protein_embedding'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        # Save the DataFrame to Parquet
        df.to_parquet(input_dict['output'], engine='pyarrow')
        logger.success(f"Data has been processed and saved to {input_dict['output']}")

    return df
