#!/bin/bash
source /data/models/datasets.sh
OUT_DIR="out/evaluation_with_corrections"
mkdir -p $OUT_DIR
DATASETS="abstracts wmt-2021-dev flores-dev"
TGT_LANG="en"
SRC_LANG="is"
for dataset in $DATASETS; do
    src_text=$(get_dataset $dataset "test" $SRC_LANG)
    tgt_text="$OUT_DIR/$dataset.$TGT_LANG"
    src_entities="$OUT_DIR/$dataset.$SRC_LANG.ner"
    tgt_entities="$OUT_DIR/$dataset.$TGT_LANG.ner"
    tgt_text_out="$OUT_DIR/$dataset.corrected.$TGT_LANG"
    mt --debug correct $src_text $tgt_text $src_entities $tgt_entities $tgt_text_out --to_nominative_case 
done
