#!/bin/bash

# Get the current script dir, normalize to it
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR"

# Run it
python3 ./src/api.py