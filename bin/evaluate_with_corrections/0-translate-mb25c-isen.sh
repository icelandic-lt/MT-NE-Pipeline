#!/bin/bash
#SBATCH --job-name=mb25c-isen
#SBATCH --output=mb25c-isen.out
#SBATCH --gres=gpu:1
set -e
source /data/models/datasets.sh

DATASETS="abstracts wmt-2021-dev flores-dev"
OUT_DIR="out/evaluation_with_corrections"
SOURCE_LANG="is"
mkdir -p $OUT_DIR

translate_all () {
    for d in $DATASETS; do
        FULL_PATH=$(get_dataset $d "test" $SOURCE_LANG)
        translate $FULL_PATH $OUT_DIR $d
        # This creates
        # $OUT_DIR/$d.translation.${REF_LANG}-${SYS_LANG} 
    done
}
source /data/models/mbart25-cont-isen/scripts/generate.sh
translate_all
