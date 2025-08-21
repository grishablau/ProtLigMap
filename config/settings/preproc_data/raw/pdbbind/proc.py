from pathlib import Path

def get_merge_and_clean_paths():
    # Define base path
    DATAROOT = Path("/data/pdbbind/core")

    # Define paths relative to DATAROOT
    config = {
        "data_root": DATAROOT,
        "paths": {
            "original_folder": DATAROOT / "raw/for_use",
            "merged_folder": DATAROOT / "stage_1_merged"
            # "general_index": DATAROOT / "INDEX_general_PL_data.2019",
            # "refined_index": DATAROOT / "INDEX_refined_data.2019",
            # "combined_index": DATAROOT / "INDEX_combined.2019",
            # "cleaned_index": DATAROOT / "INDEX_general_PL_data_cleaned.2019"
        }
    }

    return config
