MODEL_DIR=$1
MODEL_NAME=$2
SAVE_DIR=outputs
ARENA_REF=ref_data/arena-hard-gpt-4-0314.json
RESULT_DIR=${SAVE_DIR}/${MODEL_NAME}

mkdir -p ${RESULT_DIR}
mkdir -p ${RESULT_DIR}/model_answer

python cp_yaml.py \
--ref-yaml configs/llama-3-arenahard.yaml \
--model-name ${MODEL_NAME} \
--model-root-path ${MODEL_DIR}

alpaca_eval evaluate_from_model \
--model_configs "configs/${MODEL_NAME}.yaml" \
--evaluation_dataset ${ARENA_REF} \
--output_path "${RESULT_DIR}/model_answer" \
--chunksize None \

python trans_alpaca_to_arena.py \
--input-dir "${RESULT_DIR}/model_answer" \
--model-name ${MODEL_NAME}
