#!/bin/bash
export PYTHONPATH="../libs:$PATH"
export CUDA_VISIBLE_DEVICES=""
export KERAS_BACKEND=tensorflow
python runner.py --test --model $1
