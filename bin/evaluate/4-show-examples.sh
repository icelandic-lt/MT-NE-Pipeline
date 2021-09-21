#!/bin/bash
TEST_SETS="./test_sets"
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/evaluation_out"
mkdir -p $OUT_DIR

MODEL="mb25c-enis"

DATASETS="eso bible ees emea2016 os2018 tatoeba wmt-2021-dev flores-dev"
dataset="os2018"
LANG="is"
MODEL_OUT_DIR="$OUT_DIR/$MODEL"
ref_text="$TEST_SETS/$dataset".$LANG
sys_text="$MODEL_OUT_DIR/$dataset".$LANG
ref_entities="$TEST_SETS/$dataset".$LANG.ner
sys_entities="$MODEL_OUT_DIR/$dataset".$LANG.ner
mt show-examples $ref_text $sys_text $ref_entities $sys_entities --tag "T"
