#!/bin/bash

# Get the current script dir


MODEL_FILENAME="rwkv-3b-ai-town-v1.pth"
MODEL_FILESIZE=""
MODEL_FILEURL="https://huggingface.co/recursal/rwkv-5-3b-ai-town/resolve/main/rwkv-3b-ai-town-v1.pth?download=true"

# Check if the model file exists
if [ ! -f "$MODEL_FILENAME" ]; then
    # If not, download it
    curl -L "$MODEL_FILEURL" -o "$MODEL_FILENAME"
fi