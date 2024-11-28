from pathlib import Path
from typing import Tuple, Union

import torch
from torch import nn

from vox_encoder.autoencoder import VoxelAutoencoder_linear1


class Evaluate:
    def __init__(self, model: nn.Module):
        self.model: nn.Module = model

        # Set the model to evaluation mode (inference)
        self.model.eval()

    @classmethod
    def load_linear(
        cls, ckpt_path: Union[Path, str], input_dim: int, latent_dim: int
    ) -> "Evaluate":
        model = VoxelAutoencoder_linear1(input_dim, latent_dim)
        checkpoint: dict = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model)

    def inference(
        self, input_data: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Add batch dimension if it doesn't exist
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0)  # Add batch dimension

            # Forward pass
            output: torch.Tensor = self.model(input_data)

            # Apply threshold
            thresholded_output: torch.Tensor = (output > threshold).float()

            # Remove batch dimension for single sample
            output = output.squeeze(0)
            thresholded_output = thresholded_output.squeeze(0)

            return output, thresholded_output
