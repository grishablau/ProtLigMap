from loguru import logger
import subprocess
from typing import List, Tuple, Optional

def safe_run(
    cmd: List[str],
    desc: str = "",
    timeout: int = 300,
    shell: bool = False,
    cwd: Optional[str] = None,
    env: Optional[dict] = None
) -> Tuple[bool, str]:
    logger.info(f"Running: {desc}")
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=True
        )
        # If subprocess.run with check=True succeeds, return success and stdout
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        # CalledProcessError contains stdout and stderr, log both
        logger.error(f"Failed: {desc}")
        logger.error(f"STDOUT:\n{e.stdout}")
        logger.error(f"STDERR:\n{e.stderr}")
        return False, (e.stdout or "") + "\n" + (e.stderr or "")
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout expired during: {desc}")
        return False, "Timeout"
