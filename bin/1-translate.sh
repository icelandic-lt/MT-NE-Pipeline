#!/bin/bash
#SBATCH --job-name=ner_translate
#SBATCH --output=translate.out
#SBATCH --gres=gpu:1
EN_TEXTS="/data/datasets/text/en/reykjavik_grapevine/reykjavik_grapevine.txt /data/datasets/text/en/iceland_review/iceland_review.txt"
IS_TEXTS="/data/datasets/text/is/greynir_articles/articles-dump-01-11-2020:01-06-2021/articles.txt"
OUT_DIR="/data/scratch/haukurpj/Projects/MT_NER_EVAL/out"
echo $CUDA_VISIBLE_DEVICES

source /data/models/mbart25-cont-enis/scripts/generate.sh

for TEXT in $EN_TEXTS; do
    INPUT=$TEXT
    DATASET_NAME="$(basename $TEXT)"
    export MB25C_NBEAMS=1
    export MB25C_LENPEN=1.0
    #translate $INPUT $OUT_DIR $DATASET_NAME
    TRANSLATED_LOG=$OUT_DIR/$DATASET_NAME.log
    grep -P "^(D)" < $TRANSLATED_LOG | \
        sed 's/^..//' | \
        sort -n --stable | \
        cut -f3 > $OUT_DIR/$DATASET_NAME.translated.en-is
done

source /data/models/mbart25-cont-isen/scripts/generate.sh

for TEXT in $IS_TEXTS; do
    INPUT=$TEXT
    DATASET_NAME="$(basename $TEXT)"
    export MB25C_NBEAMS=1
    export MB25C_LENPEN=1.0
    translate $INPUT $OUT_DIR $DATASET_NAME
    TRANSLATED_LOG=$OUT_DIR/$DATASET_NAME.log
    grep -P "^(D)" < $TRANSLATED_LOG | \
        sed 's/^..//' | \
        sort -n --stable | \
        cut -f3 > $OUT_DIR/$DATASET_NAME.translated.is-en
done
