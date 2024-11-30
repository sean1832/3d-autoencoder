import json
from pathlib import Path

import numpy as np
import torch

from config import INPUT_DIM, LATENT_DIM
from vox_encoder import INFERENCE_IN_DIR, INFERENCE_OUT_DIR, MODEL_DIR
from vox_encoder.data_utils import insert_and_replace_2d
from vox_encoder.evaluate import Evaluate
from vox_encoder.file_io import load_data


def prepare_for_json(
    tensor: torch.Tensor, original_data: list[list[float]], column_idx: int
) -> list:
    # Convert tensor to numpy array and then to list
    flat_data = tensor.cpu().numpy().tolist()
    result = insert_and_replace_2d(original_data, column_idx, flat_data)
    return result


def main():
    # Parameters
    input_dim = INPUT_DIM
    latent_dim = LATENT_DIM

    # Data
    data_index = 6  # column of the data that is the state of the voxel

    # Paths
    checkpoint_path = Path(MODEL_DIR, "torch", "AE_checkpoint.pth")
    latent_path = Path(INFERENCE_OUT_DIR, "inference_output_latent_mod.json")

    original_data_path = Path(INFERENCE_IN_DIR, "inference_0")
    output_thresh_path = Path(INFERENCE_OUT_DIR, "inference_output_thresholded.json")

    # Initialize evaluator
    evaluator = Evaluate.load_linear(checkpoint_path, input_dim, latent_dim)

    # Load latent
    original_data = load_data(original_data_path)
    latent_data = load_data(latent_path)

    latent_tensor = torch.tensor(latent_data, dtype=torch.float32)

    # Run inference
    thresholded_output = evaluator.decode(latent_tensor)

    # Save outputs
    try:
        with open(output_thresh_path, "w") as f:
            json.dump(prepare_for_json(thresholded_output, original_data, data_index), f)

        print(f"Saved outputs to {INFERENCE_OUT_DIR}")

    except Exception as e:
        print(f"Error during saving: {e}")
        print(f"Original data type: {type(original_data)}")
        print(f"Original data shape: {np.array(original_data).shape}")
        raise


if __name__ == "__main__":
    main()
