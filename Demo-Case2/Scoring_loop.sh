#!/bin/bash

# Specify keywords
NNtype="DenseNet"
layer_type="161"
type_of_DR="tsne"
type_of_DR_for_dir="tSNE"
h_param=80

for NNtype in DenseNet #AlexNet GoogLeNet Inceptionv3Net  # DenseNet #ResNet
#for NNtype in ResNet #DenseNet
do
    # Set the list of layer types based on the NNtype
    if [ "$NNtype" = "DenseNet" ]; then
        #layer_types=(121 161 169 201)
        layer_types=(161)
    elif [ "$NNtype" = "ResNet" ]; then
        layer_types=(18 34 50 101 152)
    else
        echo "Unknown NNtype: $NNtype"
        layer_types=("")
        #exit 1
    fi

    for type_of_DR in tsne # isomap #mds #pca mds tsne umap # isomap # pca # isomap # mds # pca # umap # tsne
    do
        # Set type_of_DR_for_dir according to the value of type_of_DR
        if [ "$type_of_DR" = "tsne" ]; then
            type_of_DR_for_dir="tSNE"
            #h_params=(20 30 40 50 60 70 80 90 100)
            #h_params=(20 30 70 80 90 100)
            h_params=(80)
            hp_name="_pp"
        elif [ "$type_of_DR" = "umap" ]; then
            type_of_DR_for_dir="UMAP"
            h_params=(2 3 15 20 40 100 200)
            #h_params=(2 20 200)
            hp_name="_nn"
        elif [ "$type_of_DR" = "isomap" ]; then
            type_of_DR_for_dir="ISOMAP"
            h_params=(2 3 15 20 40 100 200)
            #h_params=(2 3 20)
            hp_name="_nn"
        elif [ "$type_of_DR" = "mds" ]; then
            type_of_DR_for_dir="MDS"
            hp_name=""
            h_params=""
        elif [ "$type_of_DR" = "pca" ]; then
            type_of_DR_for_dir="PCA"
            hp_name=""
            h_params=""
        else
            echo "Unknown type_of_DR: $type_of_DR"
            exit 1
        fi

        for h_param in "${h_params[@]}"; do
            for layer_type in "${layer_types[@]}"; do
                export layer_type  # Set layer_type as an environment variable

                mkdir -p ImageOut
                mkdir -p ImageOut/"$type_of_DR_for_dir"
                mkdir -p ImageOut/"$type_of_DR_for_dir"/"$NNtype"
                export output_dir=ImageOut/"$type_of_DR_for_dir"/"$NNtype"
                
                # Execute the Python script and pass the environment variables
                echo "$NNtype" "$type_of_DR"
                hp_name="$hp_name" type_of_DR_for_dir="$type_of_DR_for_dir" NNtype="$NNtype" layer_type="$layer_type" rock_type="$rock_type" type_of_DR="$type_of_DR" h_param="$h_param" python3 Scoring.py yes yes #_ver3.py #no yes
            done
        done
    done
done

