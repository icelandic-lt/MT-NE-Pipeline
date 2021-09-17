#!/bin/bash
#SBATCH --job-name=ner
#SBATCH --output=ner.out
#SBATCH --gres=gpu:1
echo $CUDA_VISIBLE_DEVICES
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/evaluation_out"
mkdir -p $OUT_DIR
EN_IS_MODELS="mb25-enis mb25c-enis"
IS_EN_MODELS="mb25c-isen mb25-isen"
DATASETS="eso bible ees emea2016 os2018 tatoeba wmt-2021-dev flores-dev"
DIRECTION="en-is"
LANG="is"
for MODEL in $EN_IS_MODELS; do
    MODEL_DIR="$OUT_DIR/$MODEL"
    MODEL_OUT_DIR="$OUT_DIR/$MODEL"
    mkdir -p $MODEL_OUT_DIR
    for dataset in $DATASETS; do
        cp "$MODEL_DIR/$dataset".translation.$DIRECTION "$MODEL_OUT_DIR/$dataset".$LANG
        mt ner "$MODEL_DIR/$dataset".translation.$DIRECTION "$MODEL_OUT_DIR/$dataset".$LANG.ner.unnorm --device cuda --batch_size 32 --lang $LANG
        mt normalize "$MODEL_OUT_DIR/$dataset".$LANG.ner.unnorm "$MODEL_OUT_DIR/$dataset".$LANG.ner
        rm "$MODEL_OUT_DIR/$dataset".$LANG.ner.unnorm
    done
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
