MODEL_NAME=$1

python gen_judgment.py \
--model-list ${MODEL_NAME} \
--answer-dir "data/model_answer" \
--output-dir "data/model_judge" \
--mode single \
--output_file_name ${MODEL_NAME} \
--parallel 10 \
--judge-model "gpt-4-0125-preview" \
--use-reference "math" "reasoning" "coding" \

python show_result.py --input-file "data/model_judge/${judge_model}_judge_${output_name}_single.jsonl"

python plot_radar.py --input-file "data/model_judge/${judge_model}_judge_${output_name}_single.jsonl"



