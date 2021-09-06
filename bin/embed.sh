#!/bin/bash
EN_TEXTS="/data/datasets/text/en/reykjavik_grapevine/reykjavik_grapevine.txt /data/datasets/text/en/iceland_review/iceland_review.txt"
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/out"
for EN_TEXT in $EN_TEXTS; do
    echo "Processing $EN_TEXT"
    ORIGINAL=$EN_TEXT
    NER_ENTITIES="$OUT_DIR/$(basename $EN_TEXT)"
    NER_IDXS="$OUT_DIR/$(basename $EN_TEXT).idxs"
    EMBEDDED="$OUT_DIR/$(basename $EN_TEXT).embedded"
    mt embed $ORIGINAL $NER_ENTITIES $NER_IDXS $EMBEDDED
done
