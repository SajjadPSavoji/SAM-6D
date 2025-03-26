#!/bin/bash
# Set the base directory where datasets will be stored
SNC_DIR="./Data/MegaPose-Training-Data/MegaPose-ShapeNetCore"
mkdir -p "$SNC_DIR"

# URL for the zip file
SNC_PATH="https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/shapenetcorev2.zip"

# Download the zip file to the GSO_DIR
wget -P "$SNC_DIR" "$SNC_PATH"

echo "Download completed. The file is stored in $SNC_DIR"