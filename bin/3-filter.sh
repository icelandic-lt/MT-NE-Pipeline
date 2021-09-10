#!/bin/bash
for f in out/*embedded.*.fixed; do
    echo "Extracting $f"
    DATASET_NAME="$(basename $f)"
    mt extract-embeds $f out/${DATASET_NAME}_entities out/${DATASET_NAME}_idxs out/${DATASET_NAME}_clean   
done
