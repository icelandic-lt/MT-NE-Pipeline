#!/bin/bash
#SBATCH --job-name=ner
#SBATCH --output=ner.out
#SBATCH --gres=gpu:1
EN_TEXTS="/data/datasets/text/en/reykjavik_grapevine/reykjavik_grapevine.txt /data/datasets/text/en/iceland_review/iceland_review.txt"
IS_TEXTS="/data/datasets/text/is/greynir_articles/articles-dump-01-11-2020:01-06-2021/articles.txt"
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/out2"
echo $CUDA_VISIBLE_DEVICES
mkdir -p $OUT_DIR
for TEXT in $EN_TEXTS; do
    mt ner $TEXT "$OUT_DIR/$(basename $TEXT)".ner --device cuda --batch_size 256 --lang en
done

for TEXT in $IS_TEXTS; do
    mt ner $TEXT "$OUT_DIR/$(basename $TEXT)".ner --device cuda --batch_size 256 --lang is
done

