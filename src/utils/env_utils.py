import os
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def setup_mgltools(mgtools_path: str) -> Tuple[str, str, str]:
    """
    Sets up MGLTools environment and returns key tool paths.
    """
    mgl_root = os.path.expanduser(mgtools_path)
    os.environ["LD_LIBRARY_PATH"] = f"{mgl_root}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    MGLPYTHON = f"{mgl_root}/bin/MGLpython2.7"
    PREPARE_RECEPTOR = f"{mgl_root}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
    PREPARE_LIGAND = f"{mgl_root}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
    logger.info(f"MGLTools configured from: {mgl_root}")
    return mgl_root, MGLPYTHON, PREPARE_RECEPTOR, PREPARE_LIGAND
