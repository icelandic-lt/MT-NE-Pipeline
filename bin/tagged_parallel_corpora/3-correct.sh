#!/bin/bash
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/out/parallel_corpora"
DATASETS="greynir_articles_01-11-2020:01-06-2021 newscrawl_2007-2019"
DATASETS="greynir_articles_01-11-2020:01-06-2021"
TGT_LANG="en"
SRC_LANG="is"
for dataset in $DATASETS; do
    src_text="$OUT_DIR/$dataset.filtered.$SRC_LANG"
    tgt_text="$OUT_DIR/$dataset.filtered.$TGT_LANG"
    src_entities="$OUT_DIR/$dataset.filtered.$SRC_LANG.ner"
    tgt_entities="$OUT_DIR/$dataset.filtered.$TGT_LANG.ner"
    tgt_text_out="$OUT_DIR/$dataset.filtered.corrected.$TGT_LANG"
    mt --debug --log_file $OUT_DIR/$dataset.corrections.log correct $src_text $tgt_text $src_entities $tgt_entities $tgt_text_out --to_nominative_case
done
