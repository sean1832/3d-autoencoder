import json
from pathlib import Path

import numpy as np


def load_json(path: str | Path):
    file_path: Path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")

    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_npy(path: str | Path) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")
    if Path(path).suffix != ".npy":
        raise ValueError(f"File is not a .npy file: {path}")
    return np.load(path)


def write_json(data: dict | list, path: str | Path):
    with open(path, "w") as f:
        json.dump(data, f)
