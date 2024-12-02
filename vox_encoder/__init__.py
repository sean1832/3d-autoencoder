from pathlib import Path

DATA_DIR = Path("data")

# Training data
# 20x20x20
_DATA_TRAIN_20 = Path(f"{DATA_DIR}/train/20x20x20")
DATA_TRAIN_RAW_20 = Path(f"{_DATA_TRAIN_20}/raw")
DATA_TRAIN_PROCESSED_20 = Path(f"{_DATA_TRAIN_20}/processed")

# 22x20x20
_DATA_TRAIN_22 = Path(f"{DATA_DIR}/train/22x20x20")
DATA_TRAIN_RAW_22 = Path(f"{_DATA_TRAIN_22}/raw")
DATA_TRAIN_PROCESSED_22 = Path(f"{_DATA_TRAIN_22}/processed")

# 24x24x24
_DATA_TRAIN_24 = Path(f"{DATA_DIR}/train/24x24x24")
DATA_TRAIN_RAW_24 = Path(f"{_DATA_TRAIN_24}/raw")
DATA_TRAIN_PROCESSED_24 = Path(f"{_DATA_TRAIN_24}/processed")

# 50x50x50
_DATA_TRAIN_50 = Path(f"{DATA_DIR}/train/50x50x50")
DATA_TRAIN_RAW_50 = Path(f"{_DATA_TRAIN_50}/raw")
DATA_TRAIN_PROCESSED_50 = Path(f"{_DATA_TRAIN_50}/processed")

# Model
MODEL_DIR = Path(f"{DATA_DIR}/model")
MODEL_LATEST_DIR = Path(f"{MODEL_DIR}/latest")

INFERENCE_IN_DIR = Path(f"{DATA_DIR}/inference/input")
INFERENCE_OUT_DIR = Path(f"{DATA_DIR}/inference/output")


# construct folders
DATA_DIR.mkdir(exist_ok=True)
DATA_TRAIN_RAW_50.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
INFERENCE_IN_DIR.mkdir(exist_ok=True)
INFERENCE_OUT_DIR.mkdir(exist_ok=True)

MODEL_DIR.mkdir(exist_ok=True)
