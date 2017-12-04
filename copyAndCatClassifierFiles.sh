#!/usr/bin/env bash

# Directory where we expect raw/ and feats/ directories
data_root="/home/ccervan2/data/tacl201801/"
datasets=("flickr30k" "mscoco")
splits=("train" "dev" "test")
types=("neural" "classifier")

# Copy relevant cardinality files to the same directory with the 'affinity' name,
for dataset in ${datasets[@]}; do
    for split in ${splits[@]}; do
        if [[ $dataset == "mscoco" && $split == "test" ]]; then
            continue
        fi

        eval "cp ${data_root}raw/${dataset}_${split}_mentions_card.txt ${data_root}raw/${dataset}_${split}_mentions_affinity.txt"
        for type in ${types[@]}; do
            eval "cp ${data_root}feats/${dataset}_${split}_card_${type}.feats ${data_root}feats/${dataset}_${split}_affinity_${type}.feats"
            eval "cp ${data_root}feats/${dataset}_${split}_card_${type}_meta.json ${data_root}feats/${dataset}_${split}_affinity_${type}_meta.json"
        done
    done
done

# Concatenate appropriate files for the trainDev and coco30k tasks
raw_file_suffixes=("affinity_labels.txt" "box_cats.txt" "captions.txt" "mentionPair_labels.txt"
                   "mentionPairs_cross.txt" "mentionPairs_intra_ij.txt" "mentionPairs_intra.txt"
                   "mentions_card.txt" "mentions_nonvis.txt" "mentions_affinity.txt")
for raw_suffix in ${raw_file_suffixes[@]}; do
    eval "cat ${data_root}raw/flickr30k_train_${raw_suffix} ${data_root}raw/flickr30k_dev_${raw_suffix} > ${data_root}raw/flickr30k_trainDev_${raw_suffix}"
    eval "cat ${data_root}raw/flickr30k_trainDev_${raw_suffix} ${data_root}raw/mscoco_train_${raw_suffix} > ${data_root}raw/coco30k_trainDev_${raw_suffix}"
done

# All the non-relation files require the same concatenations
tasks=("nonvis" "card" "affinity")
for task in ${tasks[@]}; do
    for type in ${types[@]}; do
        eval "cat ${data_root}feats/flickr30k_train_${task}_${type}.feats ${data_root}feats/flickr30k_dev_${task}_${type}.feats > ${data_root}feats/flickr30k_trainDev_${task}_${type}.feats"
        eval "cat ${data_root}feats/flickr30k_trainDev_${task}_${type}.feats ${data_root}feats/mscoco_train_${task}_${type}.feats > ${data_root}feats/coco30k_trainDev_${task}_${type}.feats"
        eval "cp ${data_root}feats/flickr30k_train_${task}_${type}_meta.json ${data_root}feats/flickr30k_trainDev_${task}_${type}_meta.json"
        eval "cp ${data_root}feats/flickr30k_trainDev_${task}_${type}_meta.json ${data_root}feats/coco30k_trainDev_${task}_${type}_meta.json"
    done
done

# Relation files require we concatenate intra/cross files, as well as copying some meta files
relation_suffixes=("cross" "intra" "intra_ij")
for type in ${types[@]}; do
    for rel_suffix in ${relation_suffixes[@]}; do
        for dataset in ${datasets[@]}; do
            for split in ${splits[@]}; do
                if [[ $dataset == "mscoco" && $split == "test" ]]; then
                    continue
                fi
                eval "cp ${data_root}feats/${dataset}_${split}_relation_${type}_meta.json ${data_root}feats/${dataset}_${split}_relation_${type}_${rel_suffix}_meta.json"
            done
        done
    done
    for rel_suffix in ${relation_suffixes[@]}; do
        eval "cat ${data_root}feats/flickr30k_train_relation_${type}_${rel_suffix}.feats ${data_root}feats/flickr30k_dev_relation_${type}_${rel_suffix}.feats > ${data_root}feats/flickr30k_trainDev_relation_${type}_${rel_suffix}.feats"
        eval "cat ${data_root}feats/flickr30k_trainDev_relation_${type}_${rel_suffix}.feats ${data_root}feats/mscoco_train_relation_${type}_${rel_suffix}.feats > ${data_root}feats/coco30k_trainDev_relation_${type}_${rel_suffix}.feats"
        eval "cp ${data_root}feats/flickr30k_train_relation_${type}_${rel_suffix}_meta.json ${data_root}feats/flickr30k_trainDev_relation_${type}_${rel_suffix}_meta.json"
        eval "cp ${data_root}feats/flickr30k_trainDev_relation_${type}_${rel_suffix}_meta.json ${data_root}feats/coco30k_trainDev_relation_${type}_${rel_suffix}_meta.json"
    done
done


echo "Done!"