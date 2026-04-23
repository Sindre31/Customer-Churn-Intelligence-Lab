from pathlib import Path
import json


def ensure_dirs():
    for folder in ["data/processed", "models", "outputs"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
