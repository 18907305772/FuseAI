# FuseChat-Llama-3.1-8B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="FuseAI/FuseChat-Llama-3.1-8B-Instruct"

python get_answers.py \
    --model $MODEL_NAME_OR_PATH \
    --question-file data/data_v1.1_release.jsonl \
    --save-dir data/model_answer