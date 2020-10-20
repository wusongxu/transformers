DATA_PATH=./dianshang_dataset
MODEL_PATH=/datadisk2/songxu/pretrain_model/bert-base-chinese
EXPR=0
CUDA=1
CUDA_VISIBLE_DEVICES=$CUDA
python3 run_glue.py \
        --data_dir $DATA_PATH \
        --task_name qianniu \
        --model_name_or_path $MODEL_PATH \
        --output_dir  /datadisk2/songxu/text-classification \
        --do_train \
        --do_eval \
        --evaluate_during_training \
        --eval_steps 50 \
        --overwrite_output_dir \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 32 \
        --num_train_epochs 50.0 \
        --learning_rate 3e-5 \
        --save_steps 50 \
        --logging_steps 50 \
        --logging_dir /datadisk2/songxu/text-classification/log