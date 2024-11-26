import json
from pathlib import Path


def load_data(path: str | Path):
    file_path: Path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")

    with open(path, "r") as f:
        data = json.load(f)
    return data
