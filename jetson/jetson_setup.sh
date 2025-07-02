#!/bin/bash

echo "=== SoftScreen Jetson Setup ==="

# Update & upgrade system
sudo apt-get update && sudo apt-get upgrade -y

# Install virtualenv
sudo apt-get install python3-venv -y

# Note for you: install Jetson-specific PyTorch manually!
echo "Reminder: Download and install the correct PyTorch wheel for Jetson manually."

# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r ../requirements.txt

echo "=== Jetson setup complete! ==="