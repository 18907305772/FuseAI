#! /bin/bash
set -e
export LC_ALL="POSIX"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Config
# ==============================
PYTHON_BIN=python
MODEL_DIR=$1
OUTPUT_DIR=$2
# ==============================


TP=1
TAG=${1:-"run"}

${PYTHON_BIN} -m pip install pebble pyext anthropic

OUTPUT_NAME="n1_32k_official_${TAG}"

START_DATE=2024-08-01
END_DATE=2025-02-01

echo "[TP=${TP}] ${MODEL_DIR}"
echo "Evaluating LiveCodeBench [${START_DATE} - ${END_DATE}]"
echo "Output Root: ${OUTPUT_DIR}"
echo "Tag: ${TAG}"
echo "Subfolder: ${OUTPUT_NAME}"


mkdir -p ${OUTPUT_DIR}

${PYTHON_BIN} -u -m lcb_runner_cq.runner.main \
  --model ${MODEL_DIR} \
  --n 16 \
  --model_path ${MODEL_DIR} \
  --output_name ${OUTPUT_NAME} \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_tokens 32768 \
  --stop '<｜end▁of▁sentence｜>' \
  --scenario codegeneration \
  --evaluate \
  --start_date ${START_DATE} --end_date ${END_DATE} \
  --tensor_parallel_size ${TP} \
  --output_dir ${OUTPUT_DIR} \
  --seed 10008
  
saved_eval_all_file="${OUTPUT_DIR}/${OUTPUT_NAME}/log.json"

echo "================================="
echo "[2024-08-01] --- [2025-02-01]"
${PYTHON_BIN} -u -m lcb_runner_cq.evaluation.compute_scores \
  --eval_all_file ${saved_eval_all_file} \
  --start_date 2024-08-01 --end_date 2025-02-01 | tee ${OUTPUT_DIR}/${OUTPUT_NAME}/results_2408-2502.txt

