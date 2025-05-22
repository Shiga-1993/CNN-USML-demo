#!/bin/bash
# ------------------------------------------------------------------------------
# Batch Image Transfer and Rename Script
#
# This script loops over all specified datasets (e.g., BB_2d25um_binary, Kirby_2d25um_binary, etc.),
# transfers files matching "slice_10*.png" from the source directory:
#   ../../ImageLibrary/<Dataset>_png/
# to the local destination directory:
#   ./ImageLibrary/<Dataset>_png/
#
# After copying, it renames files in the destination by replacing "slice" with "frame".
# If the destination directory does not exist, it is created.
#
# Usage:
#   chmod +x transfer_and_rename.sh
#   ./transfer_and_rename.sh
# ------------------------------------------------------------------------------
 
for Dataset in \
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
    # Construct source and destination directories (with _png suffix)
    src_dir="../../ImageLibrary/${Dataset}_png"
    dest_dir="./ImageLibrary/${Dataset}_png"
    
    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"
    
    echo "Copying files for $Dataset..."
    scp "$src_dir"/slice_10*.png "$dest_dir"/
    
    echo "Renaming files in $Dataset (replacing 'slice' with 'frame')..."
    # Loop over the copied files and rename them
    for file in "$dest_dir"/slice_10*.png; do
        if [ -f "$file" ]; then
            newfile="${file//slice/frame}"
            mv "$file" "$newfile"
            echo "Renamed: $file -> $newfile"
        fi
    done
done

