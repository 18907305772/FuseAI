## Evaluation for AlignBench v1.1

The whole evaluation process contains three steps: inference, LLM judgments and results display. 

1. **Step I** inference on target LLM and get the results

    ```bash
    export CUDA_VISIBLE_DEVICES="0"
    MODEL_NAME_OR_PATH="FuseAI/FuseChat-Llama-3.1-8B-Instruct"
    
    python get_answers.py \
        --model $MODEL_NAME_OR_PATH \
        --question-file data/data_v1.1_release.jsonl \
        --save-dir data/model_answer
    ```
    
    The answers will be saved in `data/model_answer` and ready for the LLM Judge process.

2. **Step II** get the GPT-4 judgments

   First, fill in your GPT-4 API key in `config/multi-dimension.json`.

   Then, modify and run the following script to get the judgments of the target LLM.

   ```bash
   MODEL=do_nothing # TODO modify the model name(the same as your API calling class)
   
   python judge.py \
       --config-path config/multi-dimension.json \
       --model-name $MODEL \
       --parallel 2 \
   ```

   The answers will be stored in `data/jugdment`

3. **Step III** results display

   Run the following script to get the results of all the LLM judgments saved in `data/judgment`.

   ```bash
   python show_result.py \
       --input-dir data/judgment \
       --ques-file data/data_release.jsonl \
       --save-file data/results/results.xlsx
   ```

   The calulated resultss will be stored in `data/results` in `xlsx` format.


## Acknowledgement
The codebase is adapted from [THUDM/AlignBench](https://github.com/THUDM/AlignBench).
