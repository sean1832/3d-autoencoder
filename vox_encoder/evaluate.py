from pathlib import Path
from typing import Tuple, Union

import torch
from torch import nn

from vox_encoder.ae_linear import (
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
    def load_conv(cls, ckpt_path: Union[Path, str], model: nn.Module) -> "Evaluate":
        checkpoint: dict = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model)

    def threashold(self, output: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (output > threshold).float()

    def inference_linear(
        self,
        input_data: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
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

            # calculate accuracy
            accuracy = self.calc_accuracy(input_data, thresholded_output)

            return output, thresholded_output, latent, accuracy

    def inference_cnn(
        self,
        input_data: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        with torch.no_grad():
            # Add batch dimension if it doesn't exist
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0)  # Add batch dimension

            # Forward pass
            output: torch.Tensor = self.model(input_data)

            # Apply threshold
            thresholded_output: torch.Tensor = self.threashold(output, threshold)

            # latent representation
            conv_latent = self.model.encoder_conv(input_data)
            latent: torch.Tensor = self.model.encoder_linear(conv_latent)

            # Remove batch dimension for single sample
            output = output.squeeze(0)
            thresholded_output = thresholded_output.squeeze(0)
            latent = latent.squeeze(0)

            # calculate accuracy
            accuracy = self.calc_accuracy(input_data, thresholded_output)
            return output, thresholded_output, latent, accuracy

    def inference_vae_cnn(
        self, input_data: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Perform inference for VAE CNN.

        Returns:
            - output: Reconstructed output
            - threashold_output: Thresholded reconstructed output
            - latent: latent representation
            - mu: Latent mean
            - logvar: Latent log-variance
            - accuracy: Accuracy
        """
        with torch.no_grad():
            # Add batch dimension if it doesn't exist
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0)  # Add batch dimension

            # Forward pass
            recon_x, mu, logvar = self.model(input_data)

            # Apply threshold to reconstruction
            thresholded_output: torch.Tensor = self.threashold(recon_x, threshold)

            # Sample latent vector (z) using reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mu + eps * std

            # Remove batch dimension for single sample
            recon_x = recon_x.squeeze(0)
            thresholded_output = thresholded_output.squeeze(0)
            mu = mu.squeeze(0)
            logvar = logvar.squeeze(0)

            # Calculate accuracy
            accuracy = self.calc_accuracy(input_data, thresholded_output)

            return recon_x, thresholded_output, latent, mu, logvar, accuracy

    def calc_accuracy(self, input: torch.Tensor, output: torch.Tensor) -> float:
        return (input == output).float().mean().item()
