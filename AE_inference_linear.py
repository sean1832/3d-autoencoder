import json
from pathlib import Path

import numpy as np
import torch

from config import LATENT_DIM
from vox_encoder import INFERENCE_IN_DIR, INFERENCE_OUT_DIR, MODEL_LATEST_DIR
from vox_encoder.data_utils import extract_2d, insert_and_replace_2d
from vox_encoder.evaluate import Evaluate
from vox_encoder.file_io import load_json


def prepare_for_json(tensor: torch.Tensor, original_data: dict, column_idx: int) -> dict:
    # Convert tensor to numpy array and then to list
    flat_data = tensor.view(-1).cpu().numpy().tolist()
    data = original_data["Data"]
    result = insert_and_replace_2d(data, column_idx, flat_data)
    original_data["Data"] = result
    return original_data


def main():
    # Parameters
    input_dim = 24 * 24 * 24
    latent_dim = LATENT_DIM

    # Data
    data_index = 3  # column of the data that is the state of the voxel

    # Paths
    checkpoint_path = Path(MODEL_LATEST_DIR, "AE_checkpoint_linear.pth")
    original_data_path = Path(INFERENCE_IN_DIR, "inference")
    output_raw_path = Path(INFERENCE_OUT_DIR, "inference_output_raw.json")
    output_thresh_path = Path(INFERENCE_OUT_DIR, "inference_output_thresholded.json")
    latent_path = Path(INFERENCE_OUT_DIR, "inference_output_latent.json")

    # Initialize evaluator
    evaluator = Evaluate.load_linear(checkpoint_path, input_dim, latent_dim)

    # Load inference data
    original_data = load_json(original_data_path)
    clean_data = extract_2d(original_data["Data"], data_index, float)

    # Print debug information
    print(f"Input data type: {type(clean_data)}")
    print(f"Input data shape: {np.array(clean_data).shape}")

    input_tensor = torch.tensor(clean_data, dtype=torch.float32)

    # Ensure input tensor has correct shape
    if input_tensor.shape[-1] != input_dim:
        raise ValueError(f"Input tensor must have last dimension of {input_dim}")

    # Run inference
    raw_output, thresholded_output, latent, accuracy = evaluator.inference_linear(input_tensor)

    print(f"Accuracy: {accuracy*100:.2f}%")

    # Save outputs
    try:
        with open(output_raw_path, "w") as f:
            json.dump(prepare_for_json(raw_output, original_data, data_index), f)

        with open(output_thresh_path, "w") as f:
            json.dump(prepare_for_json(thresholded_output, original_data, data_index), f)

        with open(latent_path, "w") as f:
            latent_flat = latent.cpu().numpy().tolist()
            json.dump(latent_flat, f)

        with open(Path(INFERENCE_OUT_DIR, "inference_accuracy.json"), "w") as f:
            json.dump({"accuracy": accuracy}, f)

        print(f"Saved outputs to {INFERENCE_OUT_DIR}")

    except Exception as e:
        print(f"Error during saving: {e}")
        print(f"Original data type: {type(original_data)}")
        print(f"Original data shape: {np.array(original_data).shape}")
        raise


if __name__ == "__main__":
    main()
