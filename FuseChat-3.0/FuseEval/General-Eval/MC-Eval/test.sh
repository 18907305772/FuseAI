#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
save_dir=$1
global_record_file="eval_record_collection.csv"
selected_subjects="all"
gpu_util=0.95
MODEL_NAME_OR_PATH=$2
selected_dataset=$3

 python evaluate_from_local.py \
--selected_subjects $selected_subjects \
--chat_template  \
--zero_shot  \
--selected_dataset $selected_dataset \
--save_dir $save_dir \
--model $MODEL_NAME_OR_PATH \
--global_record_file $global_record_file \
--gpu_util $gpu_util



