#!/bin/bash

# Set the environment name
ENV_NAME="my_local_env"

# Set the target path (relative to current directory)
ENV_PATH="./$ENV_NAME"

# Create the environment in the specified folder
echo " Creating local Conda environment at: $ENV_PATH"
conda create --prefix "$ENV_PATH" python=3.10 -y

# Activate the environment
echo " Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

# Install requirements (if available)
if [[ -f "Thesis_Project_Szabolcs_Pal/requirements.txt" ]]; then
    echo " Installing packages from requirements.txt"
    pip install -r requirements.txt
elif [[ -f "environment.yml" ]]; then
    echo " Installing packages from environment.yml"
    conda env update --prefix "$ENV_PATH" --file environment.yml --prune
else
    echo "⚠️ No requirements.txt or environment.yml found. Skipping package install."
fi

echo " Conda environment '$ENV_NAME' is ready at $ENV_PATH"
