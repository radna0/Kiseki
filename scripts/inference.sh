#!/bin/bash

# Default folder path
DEFAULT_FOLDER="dataset/test/baka/"

# Use the first argument if provided, otherwise use the default
FOLDER_PATH="${1:-$DEFAULT_FOLDER}"

# Run the Python scripts with the resolved folder path
python3.10 clear.py --folder-path "$FOLDER_PATH"

TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=99999999999 \
python3.10 main.py --path "$FOLDER_PATH" --radius 4
