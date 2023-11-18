#!/bin/bash

#
# OS, and system env detection
#
OS=$(uname)
if [[ "$OS" == "Darwin" ]]; then
    echo "Running on macOS"
elif [[ "$OS" == "Linux" ]]; then
    echo "Running on Linux"
elif [[ "$OS" =~ "MINGW" || "$OS" =~ "MSYS" || "$OS" =~ "CYGWIN" ]]; then
    echo "Running on Windows"
else
    echo "Unknown operating system"
fi

#
# Setup all the required pre requsites
#

