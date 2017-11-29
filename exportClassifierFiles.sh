#!/usr/bin/env bash

# Our default Image Caption Learn command
ImgCapLrnX='java -Xms8g -Xmx96g -jar target/ImageCaptionLearn-1.0.0-jar-with-dependencies.jar'

# Directory where we expect raw/ and feats/ directories
data_root="/home/ccervan2/data/tacl201801/"
datasets=("flickr30k" "mscoco")
splits=("train" "dev" "test")
tasks=("relation" "nonvis" "card")
types=("neural" "classifier")

for dataset in ${datasets[@]}; do
    for split in ${splits[@]}; do
        if [[ $dataset == "mscoco" && $split == "test" ]]; then
            continue
        fi

        # We only process reviewed coco documents
        dataset_arg="--data=$dataset"
        if [[ $dataset == "mscoco" ]]; then
            dataset_arg="--data=${dataset} --reviewed_only"
        fi
        preproc_root="${data_root}raw/${dataset}_${split}"

        # Export the caption file for this split
        eval "${ImgCapLrnX} ${dataset_arg} --split=${split} --out=${preproc_root} Data --neuralPreproc=caption"

        # Export the affinity files for this split
        eval "${ImgCapLrnX} ${dataset_arg} --split=${split} --out=${preproc_root} Data --neuralPreproc=affinity"

        # Each task must export three sets of files; one for
        # neural preprocessing, one for features used in baseline
        # classifiers, and one for features used in neural classifiers
        for task in ${tasks[@]}; do
            eval "${ImgCapLrnX} ${dataset_arg} --split=${split} --out=${preproc_root} Data --neuralPreproc=${task}"

            for type in ${types[@]}; do
                feature_file="${data_root}feats/${dataset}_${split}_${task}_${type}"
                cmd_str="${ImgCapLrnX} --threads=24 ${dataset_arg} --split=${split} --out=${feature_file} Data --extractFeats=${task} --exclude_partof"
                if [[ $type == "neural" ]]; then
                    cmd_str="${cmd_str} --for_neural"
                fi
                success=eval "${cmd_str}"
                if ! [[ success ]]; then
                    exit
                fi
            done
        done
    done
done
