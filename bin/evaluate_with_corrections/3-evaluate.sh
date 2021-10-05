#!/bin/bash
OUT_DIR="out/evaluation_with_corrections"
mkdir -p $OUT_DIR
DATASETS="abstracts wmt-2021-dev flores-dev"
TGT_LANG="en"
SRC_LANG="is"

source /data/models/datasets.sh

for dataset in $DATASETS; do
    REF=$(get_dataset $d "test" $TGT_LANG)
    tgt_text_corrected="$OUT_DIR/$dataset.corrected.$TGT_LANG"
    tgt_text_original="$OUT_DIR/$dataset.corrected.$TGT_LANG"
done
# DIRECTION="is-en"
# LANG="en"
# for MODEL in $IS_EN_MODELS; do
#     MODEL_DIR="$OUT_DIR/$MODEL"
#     MODEL_OUT_DIR="$OUT_DIR/$MODEL"
#     mkdir -p $MODEL_OUT_DIR
#     for dataset in $DATASETS; do
#         cp "$MODEL_DIR/$dataset".translation.$DIRECTION "$MODEL_OUT_DIR/$dataset".$LANG
#         mt ner "$MODEL_DIR/$dataset".translation.$DIRECTION "$MODEL_OUT_DIR/$dataset".$LANG.ner.unnorm --device cuda --batch_size 32 --lang $LANG
#         mt normalize "$MODEL_OUT_DIR/$dataset".$LANG.ner.unnorm "$MODEL_OUT_DIR/$dataset".$LANG.ner
#         rm "$MODEL_OUT_DIR/$dataset".$LANG.ner.unnorm
#     done
# done
