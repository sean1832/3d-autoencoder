import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import dtype
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vox_encoder import DATA_RAW_PATH, OUTPUT_DIR
from vox_encoder.autoencoder import VoxelAutoencoder_linear1, VoxelAutoencoder_linear2

# from vox_encoder.data_utils import extract_2d
from vox_encoder.file_io import load_data
from vox_encoder.loss import weighted_binary_cross_entropy


def process_file(file_path: str, type: dtype) -> torch.Tensor:
    """Helper function to process a single file."""
    raw_data = load_data(file_path)
    # flat_data = extract_2d(raw_data, 3, float)
    flat_data = raw_data
    return torch.tensor(flat_data, dtype=type)


def construct_data(path: str, type: torch.dtype, count: int = -1) -> list[torch.Tensor]:
    """Optimized function to load a specified amount of data as torch.Tensors."""
    tensor_data = []
    print(f"Loading tensor data from {path}")

    # Get list of files
    file_paths = [os.path.join(path, file) for file in os.listdir(path)]

    # If count is not -1 and less than the total files, slice the list
    if count != -1:
        file_paths = file_paths[:count]

    # Using ThreadPoolExecutor for parallel file processing
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each file and collect results
        futures = [executor.submit(process_file, fp, type) for fp in file_paths]
        for future in tqdm(futures):
            tensor_data.append(future.result())

    return tensor_data


def train_model_1(model, train_loader, num_epochs=50, device: str | torch.device = "cpu"):
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    model.to(device)  # Move model to the specified device

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs in train_loader:
            optimizer.zero_grad()
            input = inputs[0].to(device)  # Move inputs to device
            outputs = model(input)
            loss = weighted_binary_cross_entropy(
                outputs, input, weights=[0.25, 0.75]
            )  # sparse voxel [0.28, 0.72]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Optionally add validation and early stopping

    print("Training complete.")
    return (model, optimizer, loss, epoch)


def main():
    input_dim = 8800  # voxel grid dimension
    latent_dim = 16
    num_epoch = 50
    load_dataset_num = -1  # number of datasets (-1 for all)

    # Determine if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VoxelAutoencoder_linear1(input_dim, latent_dim)

    tensor_datas = construct_data(DATA_RAW_PATH, torch.float32, load_dataset_num)
    print(f"Loaded {len(tensor_datas)} data files")
    tensor_dataset = TensorDataset(torch.stack(tensor_datas))
    train_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

    try:
        model, optimizer, loss, epoch = train_model_1(
            model, train_loader, num_epochs=num_epoch, device=device
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss.item(),
            },
            Path(OUTPUT_DIR, "AE_checkpoint.pth"),
        )
        print(f"Model saved to {Path(OUTPUT_DIR, 'AE_checkpoint.pth')}")


if __name__ == "__main__":
    main()
