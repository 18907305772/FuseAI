MODEL_DIR=$1
MODEL_NAME=$2
SAVE_DIR=outputs
ALPACA_REF=ref_data/alpaca_eval_gpt4_baseline.json
RESULT_DIR=${SAVE_DIR}/${MODEL_NAME}

mkdir -p ${RESULT_DIR}
mkdir -p ${RESULT_DIR}/alpacaeval2

python cp_yaml.py \
--ref-yaml configs/llama-3-alpacaeval.yaml \
--model-name ${MODEL_NAME} \
--model-root-path ${MODEL_DIR}

alpaca_eval evaluate_from_model \
--model_configs "configs/${MODEL_NAME}.yaml" \
--evaluation_dataset ${ALPACA_REF} \
--output_path "${RESULT_DIR}/alpacaeval2" \
--chunksize None \
