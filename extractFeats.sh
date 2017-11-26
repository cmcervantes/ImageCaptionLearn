#!/usr/bin/env bash

# Our default Image Caption Learn command
ImgCapLrnX='java -Xms8g -Xmx96g -jar target/ImageCaptionLearn-1.0.0-jar-with-dependencies.jar'

# Feature directory and variations
feats_root="/home/ccervan2/data/tacl201712/feats/"
datasets=("flickr30k" "mscoco")
splits=("train" "dev" "test")
#tasks=("relation" "nonvis" "card")
tasks=("nonvis" "card")
types=("neural" "classifier")

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

        # Export the feature and mention idx files for each task
        for task in ${tasks[@]}; do
            for type in ${types[@]}; do
                feature_file="$feats_root$dataset"_"$split"_"$task"_"$type"
                cmd_str="$ImgCapLrnX --threads=24 $dataset_arg --split=$split --out=$feature_file Data --extractFeats=$task --exclude_partof"
                if [[ $type == "neural" ]]; then
                    cmd_str="$cmd_str --for_neural"
                fi
                success= eval "$cmd_str"
                if ! [[ success ]]; then
                    exit
                fi
            done
        done
    done
done

