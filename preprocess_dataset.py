import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vox_encoder import DATA_TRAIN_PROCESSED_22, DATA_TRAIN_PROCESSED_24, DATA_TRAIN_RAW_22, DATA_TRAIN_RAW_24
from vox_encoder.file_io import load_json


def process_file(file_path: str | Path, output_dir: str | Path) -> None:
    """
    Helper function to process a single file, convert its data to the desired dtype, and save it.
    """
    raw_data = load_json(file_path)
    np_data = np.array(raw_data["Data"])
    if np_data.ndim != 3:
        raise ValueError(f"Expected a 3D array but got {raw_data.ndim}D array from {file_path}")

    # Convert the data to the desired dtype
    converted_data = np_data.astype(np.float32)

    # Save the processed data to the output directory
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    np.save(output_path, converted_data)
    # print(f"Saved processed file to {output_path}")


def preprocess_data(input_dir: str | Path, output_dir: str | Path, count: int = -1) -> None:
    """
    Function to preprocess files by converting them to the specified dtype and saving them.
    """
    print(f"Preprocessing data from {input_dir} and saving to {output_dir}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of files in the input directory
    file_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    if len(file_paths) == 0:
        raise ValueError(f"No files found in {input_dir}")
    else:
        print(f"Processing {len(file_paths)} files")

    # If count is not -1 and less than the total files, slice the list
    if count != -1:
        file_paths = file_paths[:count]

    # # Process each file sequentially
    # for file_path in tqdm(file_paths):
    #     process_file(file_path, output_dir)

    # Using ThreadPoolExecutor for parallel file processing
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each file
        futures = [executor.submit(process_file, fp, output_dir) for fp in file_paths]
        for future in tqdm(futures):
            future.result()


try:
    preprocess_data(DATA_TRAIN_RAW_24, DATA_TRAIN_PROCESSED_24)
except KeyboardInterrupt:
    print("Preprocessing interrupted")
except Exception as e:
    print(f"An error occurred: {e}")
