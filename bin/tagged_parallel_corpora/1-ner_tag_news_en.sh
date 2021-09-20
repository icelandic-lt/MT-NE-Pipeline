#!/bin/bash
#SBATCH --job-name=ner_tag_news_en
#SBATCH --gres=gpu:1
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/parallel_corpora_out"
echo $CUDA_VISIBLE_DEVICES
mkdir -p $OUT_DIR

DATASETS="greynir_articles_01-11-2020:01-06-2021 newscrawl_2007-2019"
DATASETS="newscrawl_2007-2019"
LANGS="en is"
LANGS="en"
for lang in $LANGS; do
    for dataset in $DATASETS; do
        echo "Processing $dataset $lang"
        to_tag="$OUT_DIR/$dataset.$lang.sampled.no_id"
        unnorm_tags="$OUT_DIR/$dataset.$lang".unnorm
        mt ner $to_tag $unnorm_tags --device cuda --batch_size 64 --lang $lang
    done
done
