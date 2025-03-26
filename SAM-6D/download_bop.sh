#!/bin/bash
# This script installs the Hugging Face CLI and downloads a list of datasets

# Install the Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Set the base directory where datasets will be stored
BOP_DIR="./Data/BOP"
mkdir -p "$BOP_DIR"

# Predefined list of dataset repository names (the full repo IDs)
DATASETS=(
    # BOP Classic Core
    "bop-benchmark/lmo"
    "bop-benchmark/tless"
    "bop-benchmark/itodd"
    "bop-benchmark/hb"
    "bop-benchmark/ycbv"
    "bop-benchmark/icbin"
    "bop-benchmark/tudl"
    # BOP Classic Extra
    "bop-benchmark/lm"
    "bop-benchmark/ruapc"
    "bop-benchmark/icmi"
    "bop-benchmark/tyol"
    # BOP H3
    "bop-benchmark/hot3d"
    "bop-benchmark/handal"
    "bop-benchmark/hope"
    # BOP Industrail
    "bop-benchmark/itoddmv"
    "bop-benchmark/ipd"
    "bop-benchmark/xyzibd"
    # Megapose training data
    "bop-benchmark/megapose"
    # default det/seg/reconst
    "bop-benchmark/bop_extra"
)

# Loop over each dataset repo and download only the ZIP files
for DATASET in "${DATASETS[@]}"; do
    echo "Downloading dataset: $DATASET ..."
    # Extract the part after the slash to create a local directory name
    LOCAL_DIR="$BOP_DIR/${DATASET#*/}"
    mkdir -p "$LOCAL_DIR"
    # Download the dataset into the specific local directory
    huggingface-cli download "$DATASET" --local-dir "$LOCAL_DIR" --repo-type=dataset
done

echo "Download completed."