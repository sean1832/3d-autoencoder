from pathlib import Path

DATA_DIR = Path("data")
DATA_TRAIN = Path(f"{DATA_DIR}/train")
MODEL_DIR = Path(f"{DATA_DIR}/model")
MODEL_LATEST_DIR = Path(f"{MODEL_DIR}/latest")

INFERENCE_IN_DIR = Path(f"{DATA_DIR}/inference/input")
INFERENCE_OUT_DIR = Path(f"{DATA_DIR}/inference/output")


# construct folders
DATA_DIR.mkdir(exist_ok=True)
DATA_TRAIN.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
INFERENCE_IN_DIR.mkdir(exist_ok=True)
INFERENCE_OUT_DIR.mkdir(exist_ok=True)

MODEL_DIR.mkdir(exist_ok=True)
