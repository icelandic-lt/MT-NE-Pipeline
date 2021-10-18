#!/bin/bash
TEST_SETS="./out/test_sets"
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/evaluation_out"
mkdir -p $OUT_DIR

EN_IS_MODELS="mb25-enis mb25c-enis"
IS_EN_MODELS="mb25c-isen mb25-isen"
EN_IS_MODELS="tf-enis"
IS_EN_MODELS="tf-isen"

DATASETS="eso bible ees emea2016 os2018 tatoeba wmt-2021-dev flores-dev"
DATASETS="eso bible ees os2018 tatoeba wmt-2021-dev flores-dev"
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
        mt eval $ref_text $sys_text $ref_entities $sys_entities --tsv > $results
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
        mt eval $ref_text $sys_text $ref_entities $sys_entities --tsv > $results
    done
done
