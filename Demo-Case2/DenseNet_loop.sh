#!/bin/bash

#module load pytorch
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/ImageLibrary"                 # 入力画像置き場
OUT="$ROOT/FeatureVector/DenseNetOut"    # 出力置き場
mkdir -p "$OUT"

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
    for n_Net in 161 # 201  # Adjust this loop as needed
    do
        mkdir -p FeatureVector/DenseNetOut
        results_file=FeatureVector/DenseNetOut/DenseNet"$n_Net"_"$Dataset".out
        
        for num0 in $(seq 100 1 109); do
            num=$(printf "%03d" $num0)
            DIR=ImageLibrary
            input_f="$DIR"/"$Dataset"/frame_"$num".png
            output_f="$OUT"/DenseNet"$n_Net"_"$Dataset"_"$num".out
            #if [ -f "$input_f" ] ; then
                python3 DenseNet_flex.py "$n_Net" "$input_f" "$output_f"
                wait
                if [ -f "$output_f" ]; then
                    cat "$output_f" >> "$results_file"
                    # Delete each slice output file if not needed
                    rm "$output_f"
                fi
            #fi
        done
    done
done

