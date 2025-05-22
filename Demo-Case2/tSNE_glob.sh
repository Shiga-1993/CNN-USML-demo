#!/bin/bash
# ------------------------------------------------------------------------------
# tSNE Processing Script
#
# Overview:
#   Automates t-SNE analysis on feature vectors from neural network models
#   (DenseNet/ResNet). It builds a comma-separated list of output filenames for
#   different datasets and network configurations, then calls the Python script
#   "tSNE_glob.py" with the generated parameters.
#
# Dependencies:
#   - PyTorch module (loaded via "module load pytorch")
#   - Python 3
#   - Input data should be in the directory containing the files from the Zenodo 
#     repository "Case3-DenseNet161-Data.zip". (For the demo, only 10 images per data 
#     type are included due to data size constraints.)
#   - Output will be stored in "DR_DATA/tSNE" (created if it doesn't exist)
#
# Usage:
#   1. Adjust the NN type by editing the NNtype variable (default: DenseNet).
#   2. Modify network numbers in the loop as needed (currently, only network 161 is active).
#   3. Change the parameter "pp" if necessary (default: 80).
#   4. Make the script executable:
#         chmod +x tSNE_glob.sh
#   5. Run the script:
#         ./tSNE_glob.sh
#
# How It Works:
#   - Iterates over selected network numbers and a parameter "pp".
#   - Constructs a comma-separated keyword string from the following datasets:
#         BB_2d25um_binary, BSG_2d25um_binary, BUG_2d25um_binary,
#         BanderaBrown_2d25um_binary, BanderaGray_2d25um_binary,
#         Bentheimer_2d25um_binary, Berea_2d25um_binary, CastleGate_2d25um_binary,
#         Kirby_2d25um_binary, Leopard_2d25um_binary, Parker_2d25um_binary.
#   - Executes:
#         python3 tSNE_glob.py yes 2 <pp> <input_dir> <output_dir> "<keyword>"
#   - Echoes the keyword string for verification.
#
# Customization:
#   Modify network numbers, datasets, or Python script parameters as needed.
# ------------------------------------------------------------------------------

# Load PyTorch module
#module load pytorch

# Set neural network type
NNtype="DenseNet"

# Define input and output directories
# The input directory should be the one containing the unzipped files from
# "Case3-DenseNet161-Data.zip" obtained from the Zenodo repository.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo $ROOT
input_dir="$ROOT/Demo-Case2/FeatureVector/DenseNetOut"   # 旧 ../Case2-DenseNet161-Data
output_dir="$ROOT/Demo-Case2/DR_DATA/tSNE"

mkdir -p "$output_dir"

# Loop over network numbers (currently only network 161 is active)
for net in 161 #121 169 201
do
    # Loop over parameter values (currently only 80)
    for pp in 5
    do
        # Initialize the keyword variable
        keyword=""

        # Loop over datasets to construct the keyword string
        for Dataset0 in \
            BB_2d25um_binary \
            BSG_2d25um_binary \
            BUG_2d25um_binary \
            BanderaBrown_2d25um_binary \
            BanderaGray_2d25um_binary \
            Bentheimer_2d25um_binary \
            Berea_2d25um_binary \
            CastleGate_2d25um_binary \
            Kirby_2d25um_binary \
            Leopard_2d25um_binary \
            Parker_2d25um_binary
        do
	    Dataset="$Dataset0"_png
            NetType="${NNtype}${net}"
            # Append to the keyword string, separated by commas
            if [ -z "$keyword" ]; then
                keyword="${NetType}_${Dataset}.out"
            else
                keyword+=",${NetType}_${Dataset}.out"
            fi
        done

        # Execute the Python script if a keyword was generated
        if [ -n "$keyword" ]; then
            python3 tSNE_glob.py yes 2 "$pp" "$input_dir" "$output_dir" "$keyword"
            echo "$keyword"
        else
            echo "No keyword matched for type"
        fi
    done
done

