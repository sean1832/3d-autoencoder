import json
from pathlib import Path

import numpy as np
import torch

from vox_encoder import DATA_DIR, OUTPUT_DIR
from vox_encoder.autoencoder import VoxelAutoencoder
from vox_encoder.data_utils import extract_2d, insert_and_replace_2d
from vox_encoder.file_io import load_data


def load_model(ckpt_path: str | Path, input_dim: int, latent_dim: int) -> VoxelAutoencoder:
    model = VoxelAutoencoder(input_dim, latent_dim)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode (inference)
    model.eval()
    return model


def inference(model: VoxelAutoencoder, input_data: torch.Tensor, threshold: float = 0.5):
    with torch.no_grad():
        # Add batch dimension if it doesn't exist
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)  # Add batch dimension

        # Forward pass
        output = model(input_data)

        # Apply threshold
        thresholded_output = (output > threshold).float()

        # Remove batch dimension for single sample
        output = output.squeeze(0)
        thresholded_output = thresholded_output.squeeze(0)

        return output, thresholded_output


def prepare_for_json(tensor: torch.Tensor, original_data: list[list[float]]) -> list:
    # Convert tensor to numpy array and then to list
    flat_data = tensor.cpu().numpy().tolist()
    result = insert_and_replace_2d(original_data, 3, flat_data)
    return result


def main():
    data_root = Path(DATA_DIR, "test")
    output_dir = Path(OUTPUT_DIR, "inference")

    # Parameters
    input_dim = 40392
    latent_dim = 256

    checkpoint_path = Path(OUTPUT_DIR, "AE_checkpoint.pth")
    original_data_path = Path(data_root, "inference_0")
    output_raw_path = Path(output_dir, "inference_output_raw.json")
    output_thresh_path = Path(output_dir, "inference_output_thresholded.json")

    # Load the model
    model = load_model(checkpoint_path, input_dim, latent_dim)

    # Load inference data
    original_data = load_data(original_data_path)
    clean_data = extract_2d(original_data, 3, float)

    # Print debug information
    print(f"Input data type: {type(clean_data)}")
    print(f"Input data shape: {np.array(clean_data).shape}")

    input_tensor = torch.tensor(clean_data, dtype=torch.float32)

    # Ensure input tensor has correct shape
    if input_tensor.shape[-1] != input_dim:
        raise ValueError(f"Input tensor must have last dimension of {input_dim}")

    # Run inference
    raw_output, thresholded_output = inference(model, input_tensor)

    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_DIR, "inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    try:
        with open(output_raw_path, "w") as f:
            json.dump(prepare_for_json(raw_output, original_data), f)

        with open(output_thresh_path, "w") as f:
            json.dump(prepare_for_json(thresholded_output, original_data), f)

        print(f"Saved outputs to {output_dir}")

    except Exception as e:
        print(f"Error during saving: {e}")
        print(f"Original data type: {type(original_data)}")
        print(f"Original data shape: {np.array(original_data).shape}")
        raise


if __name__ == "__main__":
    main()
