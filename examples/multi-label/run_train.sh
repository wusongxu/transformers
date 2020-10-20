data_dir=./dataset
model_path=/datadisk2/songxu/pretrain_model/bert-base-chinese
python main.py \
--data_dir $data_dir \
--model_type bert \
--model_name_or_path bert-base-chinese \
--task_name MyProcessor \
--output_dir /datadisk2/songxu/multi-chice \
--do_train \
--do_eval \
--evaluate_during_training \
--eval_step 50 \
--overwrite_output_dir \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-5 \
--save_steps 50 \
--logging_steps 50 \
--logging_dir /datadisk2/songxu/multi-chice/log \
--num_train_epochs 3 \
--max_seq_length 80 \



