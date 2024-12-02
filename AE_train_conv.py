import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
from torch import dtype, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import LATENT_DIM, ONNX_OPSET
from vox_encoder import (
    DATA_TRAIN_PROCESSED_20,
    DATA_TRAIN_PROCESSED_22,
    DATA_TRAIN_PROCESSED_24,
    DATA_TRAIN_PROCESSED_50,
    MODEL_DIR,
    MODEL_LATEST_DIR,
)
from vox_encoder.ae_cnn import DecoderWrapperCNN, EncoderWrapperCNN, VoxelAutoencoder_CNN

# from vox_encoder.data_utils import extract_2d
from vox_encoder.file_io import load_npy
from vox_encoder.loss import weighted_binary_cross_entropy


def process_file(file_path: str, type: dtype) -> torch.Tensor:
    """Helper function to process a single file."""
    raw_data = load_npy(file_path)
    return torch.tensor(raw_data, dtype=type).unsqueeze(0)


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


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs,
    device: str | torch.device = "cpu",
):
    weight = [0.2, 0.8]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
    print(f"Training on {device}...")

    model.to(device)  # Move model to the specified device

    # Variables for early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    n_epochs_stop = 20  # <- Adjust this value

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        time_start = time.time()

        for inputs in train_loader:
            optimizer.zero_grad()
            input = inputs[0].to(device)  # Move inputs to device
            outputs = model(input)
            loss = weighted_binary_cross_entropy(
                outputs, input, weights=weight
            )  # sparse voxel [0.28, 0.72]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()

        avg_loss = running_loss / len(train_loader)
        time_end = time.time()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                input = inputs[0].to(device)  # Move inputs to device
                outputs = model(input)
                loss = weighted_binary_cross_entropy(outputs, input, weights=weight)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Avg.Train Loss: {avg_loss:.6f}, Avg.Val Loss: {avg_val_loss:.6f}, Elapsed: {time_end - time_start:.2f}s"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Optionally save the best model here
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= n_epochs_stop:
            print("Early stopping!")
            break

    print("Training complete.")
    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return (model, optimizer, loss, epoch)


def save_model(model: nn.Module, optimizer, epoch, loss, latent_dim, input_size):
    model_dir = Path(MODEL_DIR, f"{datetime.now().strftime('%y%m%d%H%M%S')}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model.eval()  # Set model to evaluation mode
    # Save PyTorch model
    torch_path = Path(model_dir, "AE_checkpoint_conv.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss.item(),
        },
        torch_path,
    )
    print(f"pytorch model saved to {torch_path}")

    # move model to cpu for onnx export since it doesn't support GPU yet
    model.to("cpu")

    # Extract the encoder and decoder components
    encoder = EncoderWrapperCNN(model.encoder_linear, model.encoder_conv)
    decoder = DecoderWrapperCNN(model.decoder_linear, model.decoder_conv, model.conv_dim, 64)

    # Save ONNX model
    encoder_path = Path(model_dir, f"encoder_conv_opset{ONNX_OPSET}.onnx")
    decoder_path = Path(model_dir, f"decoder_conv_opset{ONNX_OPSET}.onnx")

    # Define dummy input tensors for ONNX export
    dummy_encoder_input = torch.randn(
        1, 1, input_size, input_size, input_size
    )  # Input for the encoder
    dummy_decoder_input = torch.randn(1, latent_dim)  # Reshaped latent vector

    # Export the encoder to ONNX
    torch.onnx.export(
        encoder,
        (dummy_encoder_input,),
        encoder_path,
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # Variable batch size
            "output": {0: "batch_size"},
        },
    )
    print(f"Encoder exported to {encoder_path}")

    # Export the decoder to ONNX
    torch.onnx.export(
        decoder,
        (dummy_decoder_input,),
        decoder_path,
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # Variable batch size
            "output": {0: "batch_size"},
        },
    )
    print(f"Decoder exported to {decoder_path}")

    # Copy the latest model to MODEL_LATEST_DIR
    latest_dir = Path(MODEL_LATEST_DIR)
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(torch_path, latest_dir)
    shutil.copy2(encoder_path, latest_dir)
    shutil.copy2(decoder_path, latest_dir)
    print(f"Copied model to {latest_dir}")


def main():
    load_dataset_num = -1
    latent_dim = LATENT_DIM
    input_dim = 24
    batch_size = 1024
    num_epoch = 800

    # Determine if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VoxelAutoencoder_CNN(latent_dim, input_dim)

    tensor_datas = load_tensor(DATA_TRAIN_PROCESSED_24, torch.float32, load_dataset_num)
    print(f"Loaded {len(tensor_datas)} data files")
    tensor_dataset = TensorDataset(torch.stack(tensor_datas))

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(tensor_dataset))
    val_size = len(tensor_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        tensor_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    try:
        model, optimizer, loss, epoch = train_model(
            model, train_loader, val_loader, num_epochs=num_epoch, device=device
        )
        save_model(model, optimizer, epoch, loss, latent_dim, input_dim)
    except KeyboardInterrupt:
        print("Training interrupted by user.")


if __name__ == "__main__":
    main()
