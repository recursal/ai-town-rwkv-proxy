#!/bin/bash

# Get the current script dir, normalize to it
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR"

# Perform the file download
MODEL_FILENAME="rwkv-1b5-ai-town-v1.3.pth"
MODEL_FILEURL="https://huggingface.co/recursal/rwkv-5-1b5-ai-town/resolve/main/rwkv-1b5-ai-town-v1.3.pth?download=true"

# Check if the model file exists
if [ ! -f "./assets/models/$MODEL_FILENAME" ]; then
    # If not, download it
    echo "## RWKV AI town file '$MODEL_FILENAME' is missing, downloading ..."
    curl -L "$MODEL_FILEURL" -o "$MODEL_FILENAME"
else
    echo "## RWKV AI town file `$MODEL_FILENAME` already exists, skipping download"
fi

# Install requirements.txt
echo "## Ensuring requirements.txt is fullfilled with 'pip3 install -r requirements.txt'"
pip3 install -r requirements.txt
echo "## requirements.txt installation should be completed"

# RUN IT
python3 ./src/api.py "$MODEL_FILENAME" "$@"