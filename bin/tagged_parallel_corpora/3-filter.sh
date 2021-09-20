#!/bin/bash
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/parallel_corpora_out"
DATASETS="greynir_articles_01-11-2020:01-06-2021 newscrawl_2007-2019"
TGT_LANG="is"
SRC_LANG="en"
for dataset in $DATASETS; do
    src_text="$OUT_DIR/$dataset.$SRC_LANG"
    tgt_text="$OUT_DIR/$dataset.$TGT_LANG"
    src_entities="$OUT_DIR/$dataset.$SRC_LANG.unnorm"
    tgt_entities="$OUT_DIR/$dataset.$TGT_LANG.unnorm"
    src_text_out="$OUT_DIR/$dataset.filtered.$SRC_LANG"
    tgt_text_out="$OUT_DIR/$dataset.filtered.$TGT_LANG"
    src_entities_out="$OUT_DIR/$dataset.filtered.$SRC_LANG.ner"
    tgt_entities_out="$OUT_DIR/$dataset.filtered.$TGT_LANG.ner"
    mt filter-text-by-ner $src_text $tgt_text $src_entities $tgt_entities $src_text_out $tgt_text_out $src_entities_out $tgt_entities_out 
done
