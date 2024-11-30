from pathlib import Path

DATA_DIR = Path("data")
DATA_TRAIN = Path(f"{DATA_DIR}/train")
MODEL_DIR = Path(f"{DATA_DIR}/model")
INFERENCE_IN_DIR = Path(f"{DATA_DIR}/inference/input")
INFERENCE_OUT_DIR = Path(f"{DATA_DIR}/inference/output")

MODEL_TORCH_DIR = Path(f"{MODEL_DIR}/torch")
MODEL_ONNX_DIR = Path(f"{MODEL_DIR}/onnx")

# construct folders
DATA_DIR.mkdir(exist_ok=True)
DATA_TRAIN.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
INFERENCE_IN_DIR.mkdir(exist_ok=True)
INFERENCE_OUT_DIR.mkdir(exist_ok=True)

MODEL_TORCH_DIR.mkdir(exist_ok=True)
MODEL_ONNX_DIR.mkdir(exist_ok=True)
