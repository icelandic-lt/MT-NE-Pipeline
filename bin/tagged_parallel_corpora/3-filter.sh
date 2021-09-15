#!/bin/bash
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/parallel_corpora_out"
DATASETS="greynir_articles_01-11-2020:01-06-2021 newscrawl_2007-2019"
LANGS="en is"
    DATASET_NAME="$(basename $f)"
    src_text=""
    tgt_text=""
    src_entities=""
    tgt_entities=""
    src_text_out=""
    tgt_text_out=""
    src_entities_out=""
    tgt_entities_out=""
    src_
        tags="$OUT_DIR/$dataset.$lang".ner
    mt filter-text-by-ner $f out/${DATASET_NAME}_entities out/${DATASET_NAME}_idxs out/${DATASET_NAME}_clean   
done
