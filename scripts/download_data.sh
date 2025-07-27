#!/bin/bash

# This script downloads the 'Chest X-Ray Images (Pneumonia)' dataset from Kaggle.
# You must have the Kaggle API installed and configured.

# 1. Install Kaggle API: pip install kaggle
# 2. Go to your Kaggle account, 'Settings' section.
# 3. Click 'Create New Token'. This will download a 'kaggle.json' file.
# 4. Place 'kaggle.json' in '~/.kaggle/' on Linux/Mac or 'C:\Users\<Your-Username>\.kaggle\' on Windows.
#    Or, you can place it in the root of this project folder.
# 5. Make sure the permissions are secure: chmod 600 ~/.kaggle/kaggle.json

echo "Downloading the dataset from Kaggle..."

# The dataset will be downloaded into a 'data/raw' directory.
mkdir -p data/raw
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip

echo "Dataset downloaded and unzipped successfully into data/raw/"