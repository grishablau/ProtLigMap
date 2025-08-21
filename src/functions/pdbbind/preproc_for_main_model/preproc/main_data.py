from . import data_loader, data_splitter, data_standartized, data_placeholder
from typing import Dict
from loguru import logger

def load_model_data(config: Dict[str, str], do_pca: bool = False):
    logger.info("Loading data...")
    data = data_loader.load_project_data(config)

    logger.info("Splitting data by protein clusters...")
    splitted_data = data_splitter.split_by_protein_cluster(data, held_out_clusters=[3, 4, 11, 13])

    logger.info("Standardizing data%s..." % (" with PCA" if do_pca else ""))
    stdized_data = data_standartized.standardize_embeddings(splitted_data, do_pca=do_pca)

    # Merge metadata back in
    data_for_plc = stdized_data.copy()
    for key in ['valid_keys', 'protein_class', 'pic50_vals', 'protein_vecs']:
        data_for_plc[key] = splitted_data[key]

    logger.info("Creating dataloaders...")
    datasets = data_placeholder.create_dataloaders(data_for_plc)

    return datasets
