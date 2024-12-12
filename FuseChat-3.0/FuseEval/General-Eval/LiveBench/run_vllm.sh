export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL_DIR_OR_PREFIX=$1
MODEL_NAME=$2

python ${pack_path}/gen_model_answer_vllm.py  \
--bench-name live_bench \
--num-gpus-per-model 1 \
--num-gpus-total 1 \
--model-path ${MODEL_DIR_OR_PREFIX}/${MODEL_NAME} \
--model-id ${MODEL_NAME}_vllm \
--dtype bfloat16

python ${pack_path}/gen_ground_truth_judgment.py \
--bench-name live_bench \
--model-list ${MODEL_NAME}_vllm

python ${pack_path}/show_livebench_result.py \
--bench-name live_bench \
--model-list ${MODEL_NAME}_vllm
