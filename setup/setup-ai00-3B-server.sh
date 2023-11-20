#!/bin/bash

#
# System dir, OS, and system env detection
#

# Get the current script dir, normalize to it
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

# AI00 directory to download the zip files to (if not already downloaded)
AI00_DIR="${PROJ_DIR}/assets/ai00"

# The executable file name for the server
AI00_SERVER_EXECUTABLE="ai00_server"

# OS settings
OS=$(uname)
OS_TYPE="unknown"
if [[ "$OS" == "Darwin" ]]; then
    OS_TYPE="macos"
elif [[ "$OS" == "Linux" ]]; then
    OS_TYPE="linux"
elif [[ "$OS" =~ "MINGW" || "$OS" =~ "MSYS" || "$OS" =~ "CYGWIN" ]]; then
    OS_TYPE="windows"
    AI00_SERVER_EXECUTABLE="ai00_server.exe"
else
    echo "Unknown operating system"
fi

#
# Check that the AI00 executable is downloaded
#

if [[ -f "$AI00_DIR/dist/$AI00_SERVER_EXECUTABLE" ]]; then
    echo "## AI00 RWKV server for embedding already downloaded"
else
    echo "## AI00 RWKV server for embedding not downloaded, please run setup/setup-ai00-files.sh"
    exit 1

fi

#
# RWKV v4 world model download (used for embedding)
#
AITOWN_MODEL_URL="https://huggingface.co/recursal/rwkv-5-3b-ai-town/resolve/main/rwkv-3b-ai-town-v1.st?download=true"
AITOWN_MODEL_FILENAME="rwkv-3b-ai-town-v1.st"

# Directory to download the model to (if not already downloaded)
AITOWN_MODEL_DIR="${PROJ_DIR}/assets/models"

# Download the file if it does not exist
if [[ ! -f "$AITOWN_MODEL_DIR/$AITOWN_MODEL_FILENAME" ]]; then
    echo "## Downloading RWKV v5 ai-town model ... "
    curl -L "$AITOWN_MODEL_URL" -o "$AITOWN_MODEL_DIR/$AITOWN_MODEL_FILENAME"
else 
    echo "## RWKV v5 world AI town downloaded : $AITOWN_MODEL_FILENAME"
fi

#
# Run the embedding server
#
cd "$AI00_DIR/dist"
chmod +x "$AI00_SERVER_EXECUTABLE"

# Run the server
echo "## Running AI00 RWKV server for embedding ... "
./$AI00_SERVER_EXECUTABLE --config "./../../../assets/config/ai00_3B_config.toml" --ip 127.0.0.1 --port 9995