#!/bin/bash
#SBATCH --job-name=ner
#SBATCH --output=ner.out
#SBATCH --gres=gpu:1

source /data/models/datasets.sh

echo $CUDA_VISIBLE_DEVICES
OUT_DIR="out/evaluation_with_corrections"
mkdir -p $OUT_DIR
DATASETS="abstracts wmt-2021-dev flores-dev"
DATASETS="wmt-2021-dev"
DIRECTION="is-en"
LANG="en"
SOURCE_LANG="is"
for dataset in $DATASETS; do
    cp "$OUT_DIR/$dataset".translation.$DIRECTION "$OUT_DIR/$dataset".$LANG
    mt ner "$OUT_DIR/$dataset".translation.$DIRECTION "$OUT_DIR/$dataset".$LANG.ner.unnorm --device cuda --batch_size 32 --lang $LANG
    mt normalize "$OUT_DIR/$dataset".$LANG.ner.unnorm "$OUT_DIR/$dataset".$LANG.ner
    rm "$OUT_DIR/$dataset".$LANG.ner.unnorm

    # Also NER tag the test set source
    mt ner $(get_dataset $dataset "test" $SOURCE_LANG) "$OUT_DIR/$dataset".$SOURCE_LANG.ner.unnorm --device cuda --batch_size 32 --lang $SOURCE_LANG
    mt normalize "$OUT_DIR/$dataset".$SOURCE_LANG.ner.unnorm "$OUT_DIR/$dataset".$SOURCE_LANG.ner
    rm "$OUT_DIR/$dataset".$SOURCE_LANG.ner.unnorm
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
