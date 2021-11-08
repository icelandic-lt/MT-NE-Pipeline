#!/bin/bash
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/out/parallel_corpora"
DATASETS="greynir_articles_01-11-2020:01-06-2021 newscrawl_2007-2019"
DATASETS="greynir_articles_01-11-2020:01-06-2021"
TGT_LANG="en"
SRC_LANG="is"
for dataset in $DATASETS; do
    echo "Filtering $dataset"
    idx_file="$OUT_DIR/$dataset.filtered.correction_idxs.$TGT_LANG"
    tgt_text="$OUT_DIR/$dataset.filtered.corrected.$TGT_LANG"
    tgt_text_only_correct="$OUT_DIR/$dataset.only-correct-names.$TGT_LANG"
    src_text="$OUT_DIR/$dataset.filtered.$SRC_LANG"
    src_text_only_correct="$OUT_DIR/$dataset.only-correct-names.$SRC_LANG"
    
    mt filter-by-idxs $tgt_text $idx_file $tgt_text_only_correct
    mt filter-by-idxs $src_text $idx_file $src_text_only_correct
done
