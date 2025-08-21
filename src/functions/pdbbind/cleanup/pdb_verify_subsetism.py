from loguru import logger
from typing import Set, Dict

def get_pdb_codes(file_path: str) -> Set[str]:
    """
    Parses a PDB index file and returns a set of PDB codes.

    Args:
        file_path (str): Path to the index file.

    Returns:
        Set[str]: Set of unique PDB codes.
    """
    codes = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pdb_code = line.split()[0]
            codes.add(pdb_code)
    logger.info(f"Loaded {len(codes)} entries from {file_path}")
    return codes


def check_refined_subset(config: Dict[str, str]) -> None:
    """
    Checks whether the refined index file is a subset of the general index file.
    Optionally prompts user to continue.

    Args:
        config (Dict[str, str]): Dictionary with keys:
            - 'general_index': path to the general index file
            - 'refined_index': path to the refined index file
            - 'require_confirmation': 'TRUE' or 'FALSE' (case-insensitive)
    """
    general_path = config["general_index"]
    refined_path = config["refined_index"]
    require_confirmation = config.get("require_confirmation", "FALSE").upper()

    general_codes = get_pdb_codes(general_path)
    refined_codes = get_pdb_codes(refined_path)

    not_in_general = refined_codes - general_codes

    if not not_in_general:
        logger.info(" All refined entries are present in the general data (refined is a subset).")
    else:
        logger.warning(" These refined entries are NOT in the general data:")
        for code in sorted(not_in_general):
            logger.warning(f"  - {code}")

    # Optional interactive prompt
    if require_confirmation == "TRUE":
        answer = input("\n Continue with next step? (y/n): ").strip().lower()
        if answer not in {"y", "yes"}:
            logger.info(" Aborted by user.")
            exit(1)
        logger.info(" User chose to continue.")
