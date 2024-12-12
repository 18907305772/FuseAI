export HF_ENDPOINT=https://hf-mirror.com
export PATH=./envs/vllm_python310/bin:$PATH

MODEL_DIR=${1}
MODEL_DIR=${MODEL_DIR:-"/path/to/model"}
TP=${2}
TP=${TP:-TP}
OUTPUT_DIR=${3}
MODEL_TYPE=${4} # meta-llama/Meta-Llama-3.1-8B-Instruct  Qwen/Qwen2.5-7B-Instruct  google/gemma-2-9b-it
OUTPUT_DIR=${OUTPUT_DIR:-"./evaluation/livecode_bench"}
echo "LiveCodeBench: ${MODEL_DIR}, OUPTUT_DIR: ${OUTPUT_DIR}"

python -m lcb_runner.runner.main  --release_version release_latest --model ${MODEL_TYPE} --local_model_path ${MODEL_DIR} --scenario codegeneration --evaluate --tensor_parallel_size ${TP} --output_dir ${OUTPUT_DIR}

saved_eval_all_file="${OUTPUT_DIR}/log.json"
# LiveCodeBench(2408-2411)
python -m lcb_runner.evaluation.compute_scores --start_date 2024-06-30 --eval_all_file ${OUTPUT_DIR}/log.json | tee ${OUTPUT_DIR}/livecodebench-20240701-20241101.txt



