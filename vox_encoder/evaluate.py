from pathlib import Path
from typing import Tuple, Union

import torch
from torch import nn

from vox_encoder.autoencoder import (
    VoxelAutoencoder_CNN2,
    VoxelAutoencoder_linear1,
)


class Evaluate:
    def __init__(self, model: nn.Module) -> None:
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

    @classmethod
    def load_conv(cls, ckpt_path: Union[Path, str], latent_dim: int) -> "Evaluate":
        model = VoxelAutoencoder_CNN2(latent_dim)
        checkpoint: dict = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model)

    def threashold(self, output: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (output > threshold).float()

    def inference(
        self,
        input_data: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Add batch dimension if it doesn't exist
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0)  # Add batch dimension

            # Forward pass
            output: torch.Tensor = self.model(input_data)

            # Apply threshold
            thresholded_output: torch.Tensor = self.threashold(output, threshold)

            # latent representation
            latent: torch.Tensor = self.model.encoder(input_data)

            # Remove batch dimension for single sample
            output = output.squeeze(0)
            thresholded_output = thresholded_output.squeeze(0)
            latent = latent.squeeze(0)

            return output, thresholded_output, latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Add batch dimension if it doesn't exist
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)

            # Decode latent representation
            output: torch.Tensor = self.model.decoder(latent)

            # threadhold the output
            output = self.threashold(output)

            # Remove batch dimension for single sample
            output = output.squeeze(0)

            return output
