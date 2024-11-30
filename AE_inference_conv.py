import json
from pathlib import Path

import numpy as np
import torch

from config import LATENT_DIM
from vox_encoder import INFERENCE_IN_DIR, INFERENCE_OUT_DIR, MODEL_LATEST_DIR
from vox_encoder.data_utils import extract_2d, insert_and_replace_2d
from vox_encoder.evaluate import Evaluate
from vox_encoder.file_io import load_data


def prepare_for_json(
    tensor: torch.Tensor, original_data: list[list[float]], column_idx: int
) -> list:
    # Convert tensor to numpy array and then to list
    flat_data = tensor.view(-1).cpu().numpy().tolist()
    result = insert_and_replace_2d(original_data, column_idx, flat_data)
    return result


def main():
    # Parameters
    latent_dim = LATENT_DIM

    # Data
    data_index = 3  # column of the data that is the state of the voxel

    # Paths
    checkpoint_path = Path(MODEL_LATEST_DIR, "AE_checkpoint_conv.pth")
    original_data_path = Path(INFERENCE_IN_DIR, "inference")
    output_raw_path = Path(INFERENCE_OUT_DIR, "inference_output_raw.json")
    output_thresh_path = Path(INFERENCE_OUT_DIR, "inference_output_thresholded.json")
    latent_path = Path(INFERENCE_OUT_DIR, "inference_output_latent.json")

    # Initialize evaluator
    evaluator = Evaluate.load_conv(checkpoint_path, latent_dim)

    # Load inference data
    original_data = load_data(original_data_path)
    clean_data = extract_2d(original_data, data_index, float)

    clean_data = np.array(clean_data).reshape(22, 20, 20)  # Reshape to 3D grid

    # Print debug information
    print(f"Input data type: {type(clean_data)}")
    print(f"Input data shape: {np.array(clean_data).shape}")

    input_tensor = torch.tensor(clean_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Run inference
    raw_output, thresholded_output, latent = evaluator.inference(input_tensor)

    # Save outputs
    try:
        with open(output_raw_path, "w") as f:
            json.dump(prepare_for_json(raw_output, original_data, data_index), f)

        with open(output_thresh_path, "w") as f:
            json.dump(prepare_for_json(thresholded_output, original_data, data_index), f)

        with open(latent_path, "w") as f:
            latent_flat = latent.cpu().numpy().tolist()  # (128, 3, 3)
            json.dump(latent_flat, f)

        print(f"Saved outputs to {INFERENCE_OUT_DIR}")

    except Exception as e:
        print(f"Error during saving: {e}")
        print(f"Original data type: {type(original_data)}")
        print(f"Original data shape: {np.array(original_data).shape}")
        raise


if __name__ == "__main__":
    main()
