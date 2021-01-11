#!/bin/bash

export PYTHONPATH="../libs:$PATH"
echo $PYTHONPATH

export KERAS_BACKEND=tensorflow
export CUDA_VISIBLE_DEVICES=""

TOTAL_STEPS=20000000
UPDATE_INTERVAL=20

if [[ $# -le 3 ]] ; then
    echo './train.sh save_model n_workers [load_model]'
    exit 1
fi
if [ $# -eq 4 ]
then
    python runner.py --train --model $1 --n-workers $2 --sampler-update-interval $UPDATE_INTERVAL --model-load $3 --memory-load $4 --steps $TOTAL_STEPS
fi
