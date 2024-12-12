export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL_DIR=$1
MODEL_NAME=$2
NUM_GPUS=$3
NUM_GPUS_PER=$4

python gen_model_answer_vllm.py \
--model-path ${MODEL_DIR}/${MODEL_NAME} \
--num-gpus-per-model ${NUM_GPUS_PER} \
--num-gpus-total ${NUM_GPUS} \
--answer-file "data/model_answer/${MODEL_NAME}.jsonl"

