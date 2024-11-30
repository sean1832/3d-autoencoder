import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import dtype, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import BATCH_SIZE, EPOCH_NUM, INPUT_DIM, LATENT_DIM, LOAD_DATASET_NUM
from vox_encoder import DATA_TRAIN, MODEL_TORCH_DIR
from vox_encoder.autoencoder import VoxelAutoencoder_linear1

# from vox_encoder.data_utils import extract_2d
from vox_encoder.file_io import load_data
from vox_encoder.loss import weighted_binary_cross_entropy


def process_file(file_path: str, type: dtype) -> torch.Tensor:
    """Helper function to process a single file."""
    raw_data = load_data(file_path)
    # flat_data = extract_2d(raw_data, 3, float)
    flat_data = raw_data
    return torch.tensor(flat_data, dtype=type)


def load_tensor(path: str | Path, type: torch.dtype, count: int = -1) -> list[torch.Tensor]:
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


def train_model(model: nn.Module, train_loader, num_epochs=50, device: str | torch.device = "cpu"):
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    model.to(device)  # Move model to the specified device

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        time_start = time.time()

        for inputs in train_loader:
            optimizer.zero_grad()
            input = inputs[0].to(device)  # Move inputs to device
            outputs = model(input)
            loss = weighted_binary_cross_entropy(
                outputs, input, weights=[0.2, 0.8]
            )  # sparse voxel [0.28, 0.72]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        time_end = time.time()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Avg.Loss: {avg_loss:.6f}, Loss: {loss:.6f}, Elapsed: {time_end - time_start:.2f}s"
        )

        # Optionally add validation and early stopping

    print("Training complete.")
    return (model, optimizer, loss, epoch)


def main():
    num_epoch = EPOCH_NUM
    load_dataset_num = LOAD_DATASET_NUM

    # Determine if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VoxelAutoencoder_linear1(INPUT_DIM, LATENT_DIM)

    tensor_datas = load_tensor(DATA_TRAIN, torch.float32, load_dataset_num)
    print(f"Loaded {len(tensor_datas)} data files")
    tensor_dataset = TensorDataset(torch.stack(tensor_datas))
    train_loader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)

    try:
        model, optimizer, loss, epoch = train_model(
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
            Path(MODEL_TORCH_DIR, "AE_checkpoint.pth"),
        )
        print(f"Model saved to {Path(MODEL_TORCH_DIR, 'AE_checkpoint.pth')}")


if __name__ == "__main__":
    main()
