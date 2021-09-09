#!/bin/bash
#SBATCH --job-name=ner_translate
#SBATCH --output=translate.out
#SBATCH --gres=gpu:1
EN_TEXTS="/data/datasets/text/en/reykjavik_grapevine/reykjavik_grapevine.txt /data/datasets/text/en/iceland_review/iceland_review.txt"
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/out"
echo $CUDA_VISIBLE_DEVICES

source /data/models/mbart25-cont-enis/scripts/generate.sh

for EN_TEXT in $EN_TEXTS; do
    INPUT=$EN_TEXT
    DATASET_NAME="$(basename $EN_TEXT)"
    export MB25C_NBEAMS=1
    export MB25C_LENPEN=1.0
    translate $INPUT $OUT_DIR $DATASET_NAME
    TRANSLATED_LOG=$OUT_DIR/$DATASET_NAME.log
    grep -P "^(D)" < $TRANSLATED_LOG | \
        sed 's/^..//' | \
        sort -n --stable | \
        cut -f3 > $OUT_DIR/$DATASET_NAME.translated.en-is
done

for EN_TEXT in $EN_TEXTS; do
    DATASET_NAME="$(basename $EN_TEXT)".embedded
    INPUT=$OUT_DIR/$DATASET_NAME
    export MB25C_NBEAMS=1
    export MB25C_LENPEN=1.0
    translate $INPUT $OUT_DIR $DATASET_NAME
    TRANSLATED_LOG=$OUT_DIR/$DATASET_NAME.log
    # accumulate translations
    grep -P "^(D)" < $TRANSLATED_LOG | \
        sed 's/^..//' | \
        sort -n --stable | \
        cut -f3 > $OUT_DIR/$DATASET_NAME.translated.en-is
done
