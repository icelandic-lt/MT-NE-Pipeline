#!/bin/bash
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/out/parallel_corpora"
DATASETS="greynir_articles_01-11-2020:01-06-2021 newscrawl_2007-2019"
DATASETS="greynir_articles_01-11-2020:01-06-2021"
TGT_LANG="en"
SRC_LANG="is"
for dataset in $DATASETS; do
    tgt_text="$OUT_DIR/$dataset.filtered.corrected.$TGT_LANG"
    tgt_entities="$OUT_DIR/$dataset.filtered.corrected.$TGT_LANG.ner"
    tgt_embedded="$OUT_DIR/$dataset.embedded.$TGT_LANG"
    mt embed $tgt_text $tgt_entities $tgt_embedded
done
for dataset in $DATASETS; do
    src_text="$OUT_DIR/$dataset.filtered.$SRC_LANG"
    src_entities="$OUT_DIR/$dataset.filtered.$SRC_LANG.ner"
    src_embedded="$OUT_DIR/$dataset.embedded.$SRC_LANG"
    mt embed $src_text $src_entities $src_embedded
done
