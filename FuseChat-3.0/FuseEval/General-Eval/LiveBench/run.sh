MODEL_DIR_OR_PREFIX=$1
MODEL_NAME=$2

python livebench/gen_model_answer.py  \
--bench-name live_bench \
--num-gpus-per-model 1 \
--num-gpus-total 1 \
--model-path ${MODEL_DIR_OR_PREFIX}/${MODEL_NAME} \
--model-id ${MODEL_NAME} \
--dtype bfloat16

python livebench/gen_ground_truth_judgment.py \
--bench-name live_bench \
--model-list ${MODEL_NAME}

python livebench/show_livebench_result.py \
--bench-name live_bench \
--model-list ${MODEL_NAME}

