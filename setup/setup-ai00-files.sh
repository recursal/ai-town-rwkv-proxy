#!/bin/bash

#
# System dir, OS, and system env detection
#

# Get the current script dir, normalize to it
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

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
# Download the ai00 project
#

# Distribution URL path
AI00_DIST_BASE_URL="https://github.com/cgisky1980/ai00_rwkv_server/releases/download/v0.2.6"

# Zip distribution file names
AI00_MACOS_ZIP_FILE="ai00_server-v0.2.6-x86_64-apple-darwin.zip"
AI00_WINDOWS_ZIP_FILE="ai00_server-v0.2.6-x86_64-pc-windows-msvc.zip"
AI00_LINUX_ZIP_FILE="ai00_server-v0.2.6-x86_64-unknown-linux-gnu.zip"

# Download the file if it does not exist
if [[ "$OS_TYPE" == "macos" && ! -f "$AI00_DIR/$AI00_MACOS_ZIP_FILE" ]]; then
    echo "## Downloading AI00 RWKV server for MacOS ... "
    curl -L "$AI00_DIST_BASE_URL/$AI00_MACOS_ZIP_FILE" -o "$AI00_DIR/$AI00_MACOS_ZIP_FILE"
elif [[ "$OS_TYPE" == "windows" && ! -f "$AI00_DIR/$AI00_WINDOWS_ZIP_FILE" ]]; then
    echo "## Downloading AI00 RWKV server for Windows ... "
    curl -L "$AI00_DIST_BASE_URL/$AI00_WINDOWS_ZIP_FILE" -o "$AI00_DIR/$AI00_WINDOWS_ZIP_FILE"
elif [[ "$OS_TYPE" == "linux" && ! -f "$AI00_DIR/$AI00_LINUX_ZIP_FILE" ]]; then
    echo "## Downloading AI00 RWKV server for Linux ... "
    curl -L "$AI00_DIST_BASE_URL/$AI00_LINUX_ZIP_FILE" -o "$AI00_DIR/$AI00_LINUX_ZIP_FILE"
else 
    echo "## AI00 RWKV server zip archive already downloaded"
fi

# Unzip the dist folder (inside the zip) into the assets/ai00/dist/ directory
if [[ "$OS_TYPE" == "macos" && ! -f "$AI00_DIR/dist/$AI00_SERVER_EXECUTABLE" ]]; then
    echo "## Unzipping AI00 RWKV server for MacOS (for embedding server) ... "
    unzip "$AI00_DIR/$AI00_MACOS_ZIP_FILE" -d "$AI00_DIR/"
elif [[ "$OS_TYPE" == "windows" && ! -f "$AI00_DIR/dist/$AI00_SERVER_EXECUTABLE" ]]; then
    echo "## Unzipping AI00 RWKV server for Windows (for embedding server) ... "
    unzip "$AI00_DIR/$AI00_WINDOWS_ZIP_FILE" -d "$AI00_DIR/"
elif [[ "$OS_TYPE" == "linux" && ! -f "$AI00_DIR/dist/$AI00_SERVER_EXECUTABLE" ]]; then
    echo "## Unzipping AI00 RWKV server for Linux (for embedding server) ... "
    unzip "$AI00_DIR/$AI00_LINUX_ZIP_FILE" -d "$AI00_DIR/"
else 
    echo "## AI00 RWKV server for embedding server setup already completed"
fi

# Unzip the dist folder (inside the zip) into the assets/ai00/dist/ directory
if [[ "$OS_TYPE" == "macos" && ! -f "$AI00_DIR/dist/$AI00_SERVER_EXECUTABLE" ]]; then
    echo "## Unzipping AI00 RWKV server for MacOS (for embedding server) ... "
    unzip "$AI00_DIR/$AI00_MACOS_ZIP_FILE" -d "$AI00_DIR/"
elif [[ "$OS_TYPE" == "windows" && ! -f "$AI00_DIR/dist/$AI00_SERVER_EXECUTABLE" ]]; then
    echo "## Unzipping AI00 RWKV server for Windows (for embedding server) ... "
    unzip "$AI00_DIR/$AI00_WINDOWS_ZIP_FILE" -d "$AI00_DIR/"
elif [[ "$OS_TYPE" == "linux" && ! -f "$AI00_DIR/dist/$AI00_SERVER_EXECUTABLE" ]]; then
    echo "## Unzipping AI00 RWKV server for Linux (for embedding server) ... "
    unzip "$AI00_DIR/$AI00_LINUX_ZIP_FILE" -d "$AI00_DIR/"
else 
    echo "## AI00 RWKV server setup already completed"
fi
