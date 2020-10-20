export TRAIN_FILE=/home/shizai/home/songxu/dataset/train.txt
export TEST_FILE=/home/shizai/home/songxu/dataset/dev.txt
#export MODEL_PATH=/home/songshu/pretrain_model
CUDA=0
CUDA_VISIBLE_DEVICES=$CUDA
python3 run_language_modeling.py \
    --output_dir=/datadisk2/songxu/output \
    --model_type=bert \
    --model_name_or_path=bert-base-chinese \
    --do_train \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 64 \
	  --per_gpu_eval_batch_size 64 \
	  --num_train_epochs 30.0 \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --block_size 20 \
    --save_steps 1000 \
    --logging_steps 50 \
    --logging_dir=/datadisk2/songxu/log \
    --mlm