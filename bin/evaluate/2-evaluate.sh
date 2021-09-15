#!/bin/bash
TEST_SETS="./test_sets"
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/evaluation_out"
mkdir -p $OUT_DIR

EN_IS_MODELS="out_mb25_enis out_mb25c_enis"
IS_EN_MODELS="out_mb25c_default_isen out_mb25_isen"

DATASETS="eso bible ees emea2016 os2018 tatoeba wmt-2021-dev flores-dev"
LANG="is"
for MODEL in $EN_IS_MODELS; do
    MODEL_OUT_DIR="$OUT_DIR/$MODEL"
    mkdir -p $MODEL_OUT_DIR
    for dataset in $DATASETS; do
        ref_text="$TEST_SETS/$dataset".$LANG
        sys_text="$MODEL_OUT_DIR/$dataset".$LANG
        ref_entities="$TEST_SETS/$dataset".$LANG.ner
        sys_entities="$MODEL_OUT_DIR/$dataset".$LANG.ner
        wc -l $ref_text
        wc -l $sys_text
        wc -l $ref_entities
        wc -l $sys_entities
        results="$MODEL_OUT_DIR.$dataset.$LANG".results
        mt eval $ref_text $sys_text $ref_entities $sys_entities > $results
    done
done
LANG="en"
for MODEL in $IS_EN_MODELS; do
    MODEL_OUT_DIR="$OUT_DIR/$MODEL"
    mkdir -p $MODEL_OUT_DIR
    for dataset in $DATASETS; do
        ref_text="$TEST_SETS/$dataset".$LANG
        sys_text="$MODEL_OUT_DIR/$dataset".$LANG
        ref_entities="$TEST_SETS/$dataset".$LANG.ner
        sys_entities="$MODEL_OUT_DIR/$dataset".$LANG.ner
        wc -l $ref_text
        wc -l $sys_text
        wc -l $ref_entities
        wc -l $sys_entities
        results="$MODEL_OUT_DIR.$dataset.$LANG".results
        mt eval $ref_text $sys_text $ref_entities $sys_entities > $results
    done
done
