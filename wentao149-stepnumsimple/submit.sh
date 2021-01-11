export PYTHONPATH="../libs:$PATH"
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=""
export KERAS_BACKEND=tensorflow
python -i runner.py --submit --model $1 --token `cat ../wentao.token`
