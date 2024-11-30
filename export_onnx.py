from datetime import datetime
from pathlib import Path

import torch

from config import INPUT_DIM, LATENT_DIM
from vox_encoder import MODEL_ONNX_DIR, MODEL_TORCH_DIR
from vox_encoder.autoencoder import VoxelAutoencoder_linear1

# Define latent and input dimensions
input_dim = INPUT_DIM
latent_dim = LATENT_DIM

torch_ckpt_path = Path(f"{MODEL_TORCH_DIR}/AE_checkpoint.pth")


# Instantiate the model
model = VoxelAutoencoder_linear1(input_dim, latent_dim)
model.eval()  # Set model to evaluation mode

# Load pre-trained weights if available
if Path(torch_ckpt_path).exists():
    checkpoint: dict = torch.load(torch_ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded weights from {torch_ckpt_path}")
else:
    print(f"No weights found at {torch_ckpt_path}. Proceeding with random initialization.")

# Extract the encoder and decoder components
encoder = model.encoder
decoder = model.decoder

# Define dummy input tensors for ONNX export
dummy_encoder_input = torch.randn(1, input_dim)  # Input for the encoder
dummy_decoder_input = torch.randn(1, latent_dim)  # Input for the decoder

# Define file paths for saving ONNX models
model_dir = Path(f"{MODEL_ONNX_DIR}/{datetime.now().strftime('%y%m%d%H%M%S')}")
encoder_path = Path(model_dir, "encoder.onnx")
decoder_path = Path(model_dir, "decoder.onnx")


# Create directories if they don't exist
encoder_path.parent.mkdir(parents=True, exist_ok=True)
decoder_path.parent.mkdir(parents=True, exist_ok=True)

# Export the encoder to ONNX
torch.onnx.export(
    encoder,
    (dummy_encoder_input,),
    encoder_path,
    export_params=True,
    opset_version=11,
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
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},  # Variable batch size
        "output": {0: "batch_size"},
    },
)
print(f"Decoder exported to {decoder_path}")
