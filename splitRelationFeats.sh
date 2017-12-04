#!/usr/bin/env bash

py="python /home/ccervan2/source/ImageCaptionLearn_py/icl_feat_splitter.py"
data_root="/home/ccervan2/data/tacl201801/"
datasets=("flickr30k" "mscoco")
splits=("train" "dev" "test")
types=("neural" "classifier")

for dataset in ${datasets[@]}; do
    for split in ${splits[@]}; do
        if [[ $dataset == "mscoco" && $split == "test" ]]; then
            continue
        fi

        for type in ${types[@]}; do
            feature_file="${data_root}feats/${dataset}_${split}_relation_${type}.feats"
            eval "${py} --feats_file=${feature_file} --ordered_intra"
        done
    done
done

echo "Done!"