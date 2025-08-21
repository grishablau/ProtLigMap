import importlib.util
from pathlib import Path

config_path = Path(__file__).resolve().parents[4] / "config/settings/data_loader" / "data_config.py"
spec = importlib.util.spec_from_file_location("data_config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

config = config_module.config

# Make all paths absolute based on project root (assumed notebooks folder is project root)
project_root = Path(__file__).resolve().parents[2]  # Adjust as needed to root dir

for key, val in config.items():
    if isinstance(val, str) and val and not Path(val).is_absolute():
        config[key] = str(project_root / val)

result = config




