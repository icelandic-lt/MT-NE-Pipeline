#!/bin/bash
TEST_SETS="./test_sets"
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/evaluation_out"
mkdir -p $OUT_DIR

EN_IS_MODELS="mb25-enis mb25c-enis"
IS_EN_MODELS="mb25c-isen mb25-isen"

DATASETS="eso bible ees emea2016 os2018 tatoeba wmt-2021-dev flores-dev"
LANG="is"
for MODEL in $EN_IS_MODELS; do
    MODEL_OUT_DIR="$OUT_DIR/$MODEL"
    mkdir -p $MODEL_OUT_DIR
    joined_results="$MODEL_OUT_DIR.$LANG".results
    if [[ -f $joined_results ]]; then
        rm $joined_results
    fi
    for dataset in $DATASETS; do
        results="$MODEL_OUT_DIR.$dataset.$LANG".results
        cat $results >> $joined_results
    done
    echo "******************************************"
    echo "Results for $MODEL"
    mt combine-results $joined_results
done
LANG="en"
for MODEL in $IS_EN_MODELS; do
    MODEL_OUT_DIR="$OUT_DIR/$MODEL"
    mkdir -p $MODEL_OUT_DIR
    joined_results="$MODEL_OUT_DIR.$LANG".results
    if [[ -f $joined_results ]]; then
        rm $joined_results
    fi
    for dataset in $DATASETS; do
        results="$MODEL_OUT_DIR.$dataset.$LANG".results
        cat $results >> $joined_results
    done
    echo "******************************************"
    echo "Results for $MODEL"
    mt combine-results $joined_results
done
