from datetime import datetime
from pathlib import Path

import torch

from config import LATENT_DIM, ONNX_OPSET
from vox_encoder import MODEL_DIR, MODEL_TORCH_DIR
from vox_encoder.autoencoder import VoxelAutoencoder_CNN2

# Define latent and input dimensions
input_dim = (1, 22, 20, 20)  # 3D input dimensions
latent_dim = LATENT_DIM

torch_ckpt_path = Path(f"{MODEL_TORCH_DIR}/AE_checkpoint_conv.pth")


# Instantiate the model
model = VoxelAutoencoder_CNN2(latent_dim)
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
dummy_encoder_input = torch.randn(1, *input_dim)  # Input for the encoder
dummy_decoder_input = torch.randn(1, 128, 3, 3, 3)  # Reshaped latent vector

# Define file paths for saving ONNX models
model_dir = Path(f"{MODEL_DIR}/{datetime.now().strftime('%y%m%d%H%M%S')}")
encoder_path = Path(model_dir, f"encoder_conv_opset{ONNX_OPSET}.onnx")
decoder_path = Path(model_dir, f"decoder_conv_opset{ONNX_OPSET}.onnx")


# Create directories if they don't exist
encoder_path.parent.mkdir(parents=True, exist_ok=True)
decoder_path.parent.mkdir(parents=True, exist_ok=True)

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

# Export decoder
torch.onnx.export(
    decoder,
    (dummy_decoder_input,),
    str(decoder_path),
    export_params=True,
    opset_version=ONNX_OPSET,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},  # Variable batch size
        "output": {0: "batch_size"},
    },
)  # Export decoder
torch.onnx.export(
    decoder,
    (dummy_decoder_input,),
    str(decoder_path),
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
