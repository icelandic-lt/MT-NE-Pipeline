#!/bin/bash
#SBATCH --job-name=mb25-enis
#SBATCH --output=mb25-enis.out
#SBATCH --gres=gpu:1
set -e
source /data/models/datasets.sh

DATASETS="eso bible ees emea2016 os2018 tatoeba wmt-2021-dev flores-dev"
OUT_DIR="evaluation_out/mb25-enis"
SOURCE_LANG="en"
mkdir -p $OUT_DIR

detokenize_is_mb25enis () {
    for f in $OUT_DIR/*.translation.en-is; do
        mv $f $f.tok
        python ./bin/evaluate/detokenize.py < $f.tok > $f
    done
}

translate_all_dev () {
    for d in $DATASETS; do
        FULL_PATH=$(get_dataset $d dev $SOURCE_LANG)
        translate $FULL_PATH $OUT_DIR $d
        # This creates
        # $OUT_DIR/$d.translation.${REF_LANG}-${SYS_LANG} 
    done
}
source /data/models/mbart25-enis/generate.sh
translate_all_dev
detokenize_is_mb25enis
