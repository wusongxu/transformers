DATA_PATH=./mrpc
MODEL_PATH=/home/songshu/pretrain_model/bert-base-chinese
EXPR=0
CUDA=1
CUDA_VISIBLE_DEVICES=$CUDA
python3 run_glue.py \
        --data_dir $DATA_PATH \
        --model_type bert \
        --task_name mrpc \
        --model_name_or_path $MODEL_PATH \
        --output_dir ./train/$EXPR/ \
        --do_train \
        --evaluate_during_training \
        --overwrite_output_dir \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 32 \
        --num_train_epochs 50.0 \
        --learning_rate 3e-5 \
        --save_steps 50 \
        --logging_steps 100