import torch
from torch import nn


class VoxelAutoencoder_linear2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VoxelAutoencoder_linear2, self).__init__()
        self.encoder = nn.Sequential(
            # input layer
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # layer 1
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # layer 2
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            # bottleneck
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        self.decoder = nn.Sequential(
            # bottleneck
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # layer 1
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # layer 2
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1),
            # output layer
            nn.Linear(4096, input_dim),
            nn.Sigmoid(),  # For binary output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)


class VoxelAutoencoder_linear1(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VoxelAutoencoder_linear1, self).__init__()
        self.encoder = nn.Sequential(
            # input layer
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # layer 1
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            # bottleneck
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        self.decoder = nn.Sequential(
            # bottleneck
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # layer 1
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # output layer
            nn.Linear(1024, input_dim),
            nn.Sigmoid(),  # For binary output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)


class VoxelAutoencoder_CNN(nn.Module):
    def __init__(self, latent) -> None:
        super(VoxelAutoencoder_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),  # Shape: (32, 11, 10, 10)
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # Shape: (64, 5, 5, 5)
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # Shape: (128, 2, 2, 2)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),
        )
        self.encoder_output = nn.Linear(128 * 3 * 3 * 3, latent)  # Flatten latent

        self.decoder_input = nn.Linear(latent, 128 * 3 * 3 * 3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (64, 6, 5, 5)
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (32, 12, 10, 10)
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(
                32, 1, kernel_size=3, stride=2, padding=(2, 3, 3), output_padding=(1, 1, 1)
            ),  # (1, 22, 20, 20)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # flatten for linear latent layer
        encoder_output = self.encoder_output(x)
        x = self.decoder_input(encoder_output)
        x = x.view(x.size(0), 128, 3, 3, 3)  # Reshape to match ConvTranspose3d input
        x = self.decoder(x)
        return x


class VoxelAutoencoder_CNN2(nn.Module):
    def __init__(self, latent_dim) -> None:  # input shape (N, 22, 20, 20)
        super(VoxelAutoencoder_CNN2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),  # Shape: (32, 11, 10, 10)
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # Shape: (64, 5, 5, 5)
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # Shape: (128, 3, 3, 3)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),
            # -----------------------------
            # Flatten for bottleneck
            nn.Flatten(),
            # Bottleneck (Fully Connected Layers)
            nn.Linear(128 * 3 * 3 * 3, 64),  # Intermediate layer (optional)
            nn.LeakyReLU(0.1),
            nn.Linear(64, latent_dim),  # Final latent space of size latent_dim
        )

        # Decoder: Fully Connected Layers
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128 * 3 * 3 * 3),
            nn.LeakyReLU(0.1),
        )

        # Decoder: Convolutional Layers
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Shape: (64, 6, 5, 5)
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Shape: (32, 12, 10, 10)
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(
                32, 1, kernel_size=3, stride=2, padding=(2, 3, 3), output_padding=(1, 1, 1)
            ),  # Shape: (N, 22, 20, 20)
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoding
        latent = self.encoder(x)
        # Decoding
        x = self.decoder_fc(latent)
        x = x.view(-1, 128, 3, 3, 3)  # Reshape for ConvTranspose3d
        x = self.decoder_conv(x)
        return x


# Wrapper for the decoder since onnx export doesn't support `Unflatten` layer
class DecoderWrapperCNN(nn.Module):
    def __init__(self, decoder_fc, decoder_conv):
        super(DecoderWrapperCNN, self).__init__()
        self.decoder_fc = decoder_fc
        self.decoder_conv = decoder_conv

    def forward(self, x):
        x = self.decoder_fc(x)  # Fully connected layers
        x = x.view(-1, 128, 3, 3, 3)  # Reshape for ConvTranspose3d
        x = self.decoder_conv(x)  # Convolutional layers
        return x
