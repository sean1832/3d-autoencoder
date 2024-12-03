import math

import torch
from torch import nn


class VAE_CNN(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int) -> None:  # input shape (1, 24, 24, 24)
        super(VAE_CNN, self).__init__()

        self.conv_dim: int = self._calc_conv_dim(  # input layer
            self._calc_conv_dim(  # layer 1
                self._calc_conv_dim(input_dim)  # layer 2
            )
        )

        # Encoder
        self.encoder_conv = nn.Sequential(
            # input layer
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # Shape: (16, 12, 12, 12)
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.1),
            # layer 1
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # Shape: (32, 6, 6, 6)
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),
            # layer 2
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # Shape: (64, 3, 3, 3)
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),
            # Flatten for latent
            nn.Flatten(),
        )

        # Separate linear layers for mean and log variance
        self.fc_mu = nn.Linear(64 * self.conv_dim**3, latent_dim)
        self.fc_logvar = nn.Linear(64 * self.conv_dim**3, latent_dim)

        # Decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64 * self.conv_dim**3),
            nn.LeakyReLU(0.1),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (32, 6, 6, 6)
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (16, 12, 12, 12)
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (1, 24, 24, 24)
            nn.Sigmoid(),
        )

    def _calc_conv_dim(self, input_dim: int) -> int:
        # output size = (n+2p-f)/s+1
        # (n = input size, p = padding, f = filter size, s = stride)
        return math.floor(
            (input_dim + 2 * 1 - 3) / 2 + 1  # kernel_size = 3, stride = 2, padding = 1
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0, 1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder_conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder_linear(z)
        x = x.view(-1, 64, self.conv_dim, self.conv_dim, self.conv_dim)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class EncoderWrapperCNN_VAE(nn.Module):
    def __init__(
        self,
        encoder_conv: nn.Module,
        fc_mu: nn.Module,
        fc_logvar: nn.Module,
    ):
        """
        Wrapper for the encoder to separate the convolutional layers and the two
        linear layers for mean and log variance.
        """
        super(EncoderWrapperCNN_VAE, self).__init__()
        self.encoder_conv = encoder_conv
        self.fc_mu = fc_mu
        self.fc_logvar = fc_logvar

    def forward(self, x):
        # First pass through convolutional layers
        x = self.encoder_conv(x)

        # Compute mean and log variance using separate linear layers
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


# Wrapper for the decoder since onnx export doesn't support `Unflatten` layer
class DecoderWrapperCNN_VAE(nn.Module):
    def __init__(
        self,
        decoder_linear: nn.Sequential,
        decoder_conv: nn.Sequential,
        conv_dim: int,
        channels: int,
    ):
        super(DecoderWrapperCNN_VAE, self).__init__()
        self.decoder_linear = decoder_linear
        self.decoder_conv = decoder_conv
        self.conv_dim = conv_dim
        self.channels = channels

    def forward(self, x):
        x = self.decoder_linear(x)  # Fully connected layers
        x = x.view(
            -1, self.channels, self.conv_dim, self.conv_dim, self.conv_dim
        )  # Reshape for ConvTranspose3d
        x = self.decoder_conv(x)  # Convolutional layers
        return x
