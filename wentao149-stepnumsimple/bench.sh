#!/bin/bash

export PYTHONPATH="../libs"
export KERAS_BACKEND=tensorflow
export CUDA_VISIBLE_DEVICES=""

mkdir -p models/

python runner.py --train --model models/bench --n-workers 1 
