MODEL_NAME=$1
SOURCE_ANS_PATH=outputs/${MODEL_NAME}/model_answer
TARGET_ANS_PATH=data/model_answer
SOURCE_JUDGE_PATH=outputs/${MODEL_NAME}/model_judgment
TARGET_JUDGE_PATH=data/model_judgment


mv "${SOURCE_ANS_PATH}/${model_name}.jsonl" "$TARGET_ANS_PATH"
mv "${SOURCE_JUDGE_PATH}/${model_name}.jsonl" "$TARGET_JUDGE_PATH"

python show_result.py \
--answer-dir ${TARGET_ANS_PATH} \
--judgement-dir ${TARGET_JUDGE_PATH} \
--eval-model ${model_name}

mv "${TARGET_ANS_PATH}/${model_name}.jsonl" "$SOURCE_ANS_PATH"
mv "${TARGET_JUDGE_PATH}/${model_name}.jsonl" "$SOURCE_JUDGE_PATH"


