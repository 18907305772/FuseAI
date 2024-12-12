## Evaluation for ArenaHard

### Requirements
You can install the required packages with the following command:
```bash
pip install -r requirements.txt
```
Please refer to [lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto) for detailed setup instructions.
### Evaluation
The whole evaluation process contains three steps: inference, LLM judgments, and results display. 

1. **Step I** inference on target LLM and get the results

    This step is similar to AlpacaEval-2, make sure you have installed requirements for AlpacaEval-2.

    ```bash
    export CUDA_VISIBLE_DEVICES="0"
    MODEL_DIR=path/to/model
    MODEL_NAME="FuseChat-Llama-3.1-8B-Instruct"
  
    bash generate.sh ${MODEL_DIR} ${MODEL_NAME}
    ```
    
    The answers will be saved in `outputs/${MODEL_NAME}` and ready for the LLM Judge process.

2. **Step II** get the GPT-4 judgments

   **(1).** Fill in your GPT-4 API endpoint in `judge_configs/api_config.yaml`.
   ```yaml
   gpt-4-1106-preview:
    model_name: gpt-4-1106-preview
    endpoints:
     - api_base: [YOUR-ENDPOINT-URL]
       api_key: [YOUR-API-KEY]
    api_type: openai
    parallel: 8
   ```

   **(2).** In `judge_configs/judge_config.yaml`, add your model name in `model_list`.
   ```yaml
   model_list:
     - [YOUR-MODEL-NAME]
   ```

   **(3).** Run the command to generate judgments:
   ```console
   python gen_judgment.py --setting-file judge_configs/judge_config.yaml --endpoint-file judge_configs/api_config.yaml
   ```
   Judgment caching is also implemented. It will skip generating judgments that has already been generated or lacks one of the model answers.  

3. **Step III** results display

   Display Win Rate:
   ```bash
   bash eval.sh 
   ```
   
## Acknowledgement
The codebase is adapted from [lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto).
