#!/bin/bash
EN_TEXTS="/data/datasets/text/en/reykjavik_grapevine/reykjavik_grapevine.txt /data/datasets/text/en/iceland_review/iceland_review.txt"
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/out"
export CUDA_VISIBLE_DEVICES="0"
mkdir -p $OUT_DIR
for EN_TEXT in $EN_TEXTS; do
    mt ner - - < $EN_TEXT  > "$OUT_DIR/$(basename $EN_TEXT)" --device cuda:0 --idx_file "$OUT_DIR/$(basename $EN_TEXT).idxs" --batch_size 256
done
