#!/bin/bash
#SBATCH --job-name=tf-base-translate
#SBATCH --gres=gpu:1
set -e
source /data/models/datasets.sh

DATASETS="eso bible ees emea2016 os2018 tatoeba wmt-2021-dev flores-dev"
OUT_DIR="evaluation_out/tf-isen"
SOURCE_LANG="is"
TARGET_LANG="en"
mkdir -p $OUT_DIR

translate () {
    TO_TRANSLATE=$1
    OUT_DIR=$2
    NAME=$3
    L1=$SOURCE_LANG
    L2=$TARGET_LANG
    USER_DIR=/data/scratch/haukurpj/miniconda3/envs/mt_ner_eval/src/greynirseq/src/greynirseq/

    DATA_DIR=/data/models/isl-eng-base-v6
    VOCAB_JSON=/data/models/eng-isl-vocab/eng-isl-vocab/eng-isl-bbpe-32k/eng-isl-bbpe-32k-vocab.json
    VOCAB_MERGES=/data/models/eng-isl-vocab/eng-isl-vocab/eng-isl-bbpe-32k/eng-isl-bbpe-32k-merges.txt

    cat $TO_TRANSLATE | fairseq-interactive $DATA_DIR \
        --task translation_with_backtranslation \
        --source-lang $L1 --target-lang $L2 \
        --user-dir $USER_DIR \
        --max-tokens 16000 --max-source-positions 400 --max-target-positions 400 \
        --path $DATA_DIR/checkpoint_5_60000.is-en.base-v6.pt \
        --skip-invalid-size-inputs-valid-test \
        --bpe gpt2 \
        --gpt2-encoder-json $VOCAB_JSON \
        --gpt2-vocab-bpe $VOCAB_MERGES \
        --num-workers 10 \
        --beam 4 --nbest 4 \
        --lenpen 0.6 \
        --input - > $OUT_DIR/$NAME.translation.log.$SOURCE_LANG-$TARGET_LANG
}

translate_all_dev () {
    for d in $DATASETS; do
        FULL_PATH=$(get_dataset $d dev $SOURCE_LANG)
        echo "Translating $d $SOURCE_LANG -> $TARGET_LANG"
        translate $FULL_PATH $OUT_DIR $d
        # This creates
        # $OUT_DIR/$d.translation.${REF_LANG}-${SYS_LANG} 
    done
}

extract_from_log () {
    for d in $DATASETS; do
        cat $OUT_DIR/$d.translation.log.$SOURCE_LANG-$TARGET_LANG | get_beam_no_score_no_index > $OUT_DIR/$d.translation.${SOURCE_LANG}-$TARGET_LANG
    done
}

get_beam_no_score_no_index () {
    NUM_BEAMS=4
    grep -P "^(D)" | awk "NR % $NUM_BEAMS == 1" | sed 's/^..//' | sort -n --stable | cut -f3
}

#translate_all_dev
extract_from_log
