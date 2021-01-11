#!/bin/bash
export PYTHONPATH="../libs:$PATH"
export KERAS_BACKEND=tensorflow
export CUDA_VISIBLE_DEVICES=""

if [[ $# -le 2 ]] ; then
    echo './test_parallel.sh model pool_size num_test_runs'
    exit 1
fi
if [ $# -eq 3 ]
then
    python test_parallel.py $1 --pool-size $2 --run-count $3
fi

