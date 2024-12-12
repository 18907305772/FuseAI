MODEL_NAME=$1
SAVE_DIR=outputs
ANSWER_DIR=${SAVE_DIR}/${MODEL_NAME}/alpacaeval2
gpt4_baseline="ref_data/alpaca_eval_gpt4_baseline.json"

alpaca_eval evaluate \
--model_outputs $ANSWER_DIR/model_outputs.json \
--reference_outputs $gpt4_baseline \
--annotators_config 'weighted_alpaca_eval_gpt4_turbo' \
--output_path "${ANSWER_DIR}" \
--name $model_name \
--caching_path "${ANSWER_DIR}/weighted_alpaca_eval_gpt4_turbo_annotations.json"  \



