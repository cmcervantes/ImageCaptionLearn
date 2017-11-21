#!/usr/bin/env bash

data="mscoco"
split="dev"
input="$data""_""$split""_imgIDs.txt"
while IFS= read -r var
do
  eval "mv /home/ccervan2/data/tacl201712/feats/$data""_boxes/${var/.jpg/.feats} /home/ccervan2/data/tacl201712/feats/$data""_boxes/$split/"
done < "$input"