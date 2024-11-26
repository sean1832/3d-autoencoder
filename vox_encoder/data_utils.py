from typing import Callable, Optional

import numpy as np


def flatten_data(data: list[list[float]]) -> np.ndarray:
    flat_data = np.array(data).flatten()
    # Check and report length
    print(f"Flattening data from length {len(data)} to {len(flat_data)}")
    return flat_data


def extract_2d(
    data: list[list[float]], index: int, cast_type: Optional[Callable] = None
) -> np.ndarray:
    """
    Extracts a column from a 2D list and optionally casts its type.

    Args:
        data (list[list[float]]): 2D list of floats.
        index (int): Index of the column to extract.
        cast_type (Optional[Callable]): A callable to cast the data type (e.g., bool, int, etc.). Defaults to None.

    Returns:
        np.ndarray: Extracted column as a NumPy array, optionally cast to the specified type.
    """
    # Convert data to NumPy array and extract the column
    extracted = np.array(data)[:, index]

    # If a cast_type is provided, apply it to the extracted data
    if cast_type:
        extracted = extracted.astype(cast_type)  # type: ignore #

    return extracted


def insert_and_replace_2d(
    data: list[list[float]], index: int, new_data: list[float]
) -> list[list[float]]:
    for i, row in enumerate(data):
        row[index] = new_data[i]
    return data
