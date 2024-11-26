import os

DATA_CLEAN_PATH = "data/clean"
DATA_RAW_PATH = "data/raw"
DATA_DIR = "data"

OUTPUT_DIR = "output"


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
