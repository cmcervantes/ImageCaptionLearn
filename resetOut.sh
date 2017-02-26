#!/usr/bin/env bash

# If there's no out/ dir, make one; do not
# delete it if it already exists
if [ ! -d "out" ]; then
    mkdir "out/"
fi

# create a separate directory for each setting
for dir in "relation" "grounding" "joint"
do
    # If these directories exist, remove them
    dirname="out/$dir"
    if [ -d "$dirname" ]; then
        rm -r -f "$dirname"
    fi
    mkdir "$dirname"
done

# In the relation folder, add a conll directory for
# each of the relation settings; also add an htm folder
for dir in "pairwise" "plus_pronom" "inf"
do
    dirname="out/relation/$dir""_conll/"
    if [ -d "$dirname" ]; then
        rm -r -f "$dirname"
    fi
    mkdir "$dirname"
done
if [ -d "out/relation/htm/" ]; then
    rm -r -f "out/relation/htm/"
fi
mkdir "out/relation/htm/"
if [ -d "out/relation/relation/" ]; then
    rm -r -f "out/relation/relation/"
fi
mkdir "out/relation/relation/"

# In the joint folder, add a conll, htm, and grounding
# directory
for dir in "conll" "htm" "grounding" "relation"
do
    dirname="out/joint/$dir/"
    if [ -d "$dirname" ]; then
        rm -r -f "$dirname"
    fi
    mkdir "$dirname"
done