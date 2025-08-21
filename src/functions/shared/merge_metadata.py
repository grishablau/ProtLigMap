import os
from typing import List, Dict
from loguru import logger


def read_index_file(file_path: str) -> Dict[str, str]:
    """
    Reads a PDB index file and returns a dictionary mapping PDB codes to lines.
    Skips comments and blank lines.

    Args:
        file_path (str): Path to the input index file.

    Returns:
        Dict[str, str]: Dictionary of PDB code → full line.
    """
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pdb_code = line.split()[0]
            data[pdb_code] = line
    logger.info(f"Loaded {len(data)} entries from {file_path}")
    return data


def combine_index_files(input_dict: Dict[str, str]) -> None:
    """
    Combines multiple PDB index files into one, merging by PDB code.

    Args:
        input_dict (Dict[str, str]): Dictionary with:
            - 'input_paths': list of index file paths, seperated by comma
            - 'output_path': destination path for the combined output
    """

    input_comma_seperated: List[str] = input_dict["input_paths"]
    input_paths = [s.strip() for s in input_comma_seperated.split(',')]

    output_path = input_dict["output_path"]

    if not input_paths:
        logger.warning("No input files provided.")
        return

    combined_data: Dict[str, str] = {}
    header_lines: List[str] = []

    for idx, path in enumerate(input_paths):
        if not os.path.exists(path):
            logger.warning(f"File not found, skipping: {path}")
            continue

        data = read_index_file(path)
        combined_data.update(data)

        if idx == 0:
            with open(path, 'r') as f:
                header_lines = [line for line in f if line.startswith('#')]

    if not combined_data:
        logger.warning("No data found in any input files. Nothing written.")
        return

    # Write output
    with open(output_path, 'w') as f_out:
        for line in header_lines:
            f_out.write(line)
        for pdb_code in sorted(combined_data):
            f_out.write(combined_data[pdb_code] + '\n')

    logger.info(f"Combined index written to: {output_path}")
    logger.info(f"Total unique entries: {len(combined_data)}")
