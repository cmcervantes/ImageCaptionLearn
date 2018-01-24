#!/usr/bin/env bash

source_root="~/source/ImageCaptionLearn/"
dest_root="clgrad5:~/source/ImageCaptionLearn/"
staging_dirs=("staging_0/" "staging_1/" "staging_2/")

# Copy our source dir, all bash scripts, the pom and readme to each
# of our staging areas (which we use to conduct multiple experiments
# with different builds
for staging_dir in ${staging_dirs[@]}; do
    eval "scp -r ${source_root}src/ ${dest_root}${staging_dir} >/dev/null"
    eval "scp ${source_root}*.sh ${dest_root}${staging_dir} >/dev/null"
    eval "scp ${source_root}pom.xml ${dest_root}${staging_dir} >/dev/null"
    eval "scp ${source_root}README ${dest_root}${staging_dir} >/dev/null"
    eval "scp ${source_root}paths.config ${dest_root}${staging_dir} >/dev/null"
done
