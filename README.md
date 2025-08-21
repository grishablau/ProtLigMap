# ProtLigMap Pipeline

This repository implements a modular pipeline for processing protein-ligand data, embedding it using deep learning models, and running downstream machine learning tasks.

## Overview

The project is structured in multiple stages:

1. **Pipeline Execution**  
   The main entry point for the pipeline is:
   - `main.py`: Runs the overall pipeline logic.
   - `pipeline.yaml`: Defines the pipeline configuration, stages, and parameters.

2. **Data Preparation for Modeling**  
   After running the main pipeline, data is prepared for ML training through the following notebooks:
   - `for_ProtLigMap/notebooks/post_backbone_model_pre_ml_model_preproc/model_data_preparation.ipynb`: Prepares the dataset (e.g., data splitting, standartisation etc.).
   - `for_ProtLigMap/notebooks/post_backbone_model_pre_ml_model_preproc/data_analysis_and_run_model.ipynb`: Analyzes the processed data and runs contrastive ML models.

## Requirements

- Python 3.12.3  
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
