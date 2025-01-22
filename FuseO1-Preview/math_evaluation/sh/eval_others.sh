set -ex
proj_dir="<PROJECT_DIR>"
PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=$3
SEED=$4

SPLIT="test"
NUM_TEST_SAMPLE=-1


# English open datasets
DATA_NAME="amc23,math,gsm8k,olympiadbench,college_math"
TOKENIZERS_PARALLELISM=false \
python3 -u ${proj_dir}/math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_model_len 32768 \
    --max_tokens_per_call 32768 \
