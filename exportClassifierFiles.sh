#!/usr/bin/env bash

# Our default Image Caption Learn command
ImgCapLrnX='java -Xms8g -Xmx96g -jar target/ImageCaptionLearn-1.0.0-jar-with-dependencies.jar'

# Feature and raw root directories
raw_root="/home/ccervan2/data/tacl201711/raw/"
feats_root="/home/ccervan2/data/tacl201711/feats/"
datasets=("flickr30k" "mscoco")
splits=("train" "dev" "test")
tasks=("relation" "nonvis" "card")

for dataset in ${datasets[@]}; do
    for split in ${splits[@]}; do
        if [[ $dataset == "mscoco" && $split == "test" ]]; then
            continue
        fi

        # We only process reviewed coco document
        dataset_arg="--data=$dataset"
        if [[ $dataset == "mscoco" ]]; then
            dataset_arg="--data=$dataset --reviewed_only"
        fi

        # Export the caption file for this split
        preproc_file="$raw_root$dataset"_"$split"
        eval "$ImgCapLrnX $dataset_arg --split=$split --out=$preproc_file Data --neuralPreproc=caption"

        # Export the feature and mention idx files for each task
        for task in ${tasks[@]}; do
            feature_file="$feats_root$dataset"_"$split"_"$task"
            eval "$ImgCapLrnX --threads=24 $dataset_arg --split=$split --out=$feature_file Data --extractFeats=$task --for_neural --exclude_partof"
            eval "$ImgCapLrnX $dataset_arg --split=$split --out=$preproc_file Data --neuralPreproc=$task"
        done
    done
done

