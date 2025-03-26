#!/bin/bash
# Set the base directory where datasets will be stored
GSO_DIR="./Data/MegaPose-Training-Data/MegaPose-GSO"
mkdir -p "$GSO_DIR"

# URL for the zip file
GSO_PATH="https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/google_scanned_objects.zip"

# Download the zip file to the GSO_DIR
wget -P "$GSO_DIR" "$GSO_PATH"

echo "Download completed. The file is stored in $GSO_DIR"