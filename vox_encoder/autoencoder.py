from torch import nn


class VoxelAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VoxelAutoencoder, self).__init__()
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


# TODO: implement Convolutional Autoencoder
# Convolutional Autoencoder will likely perform better than the fully connected one
