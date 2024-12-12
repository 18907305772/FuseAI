## Evaluation for AlpacaEval-2

### Requirements
You can install the required packages with the following command:
```bash
pip install -e .
```
Please refer to [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) for detailed setup instructions.
### Evaluation
The whole evaluation process contains two steps: inference and LLM judgments.

1. **Step I** inference on target LLM and get the results

    ```bash
    export CUDA_VISIBLE_DEVICES="0"
    MODEL_DIR=path/to/model
    MODEL_NAME="FuseChat-Llama-3.1-8B-Instruct"
  
    bash generate.sh ${MODEL_DIR} ${MODEL_NAME}
    ```
    
    The answers will be saved in `outputs/${MODEL_NAME}` and ready for the LLM Judge process.

2. **Step II** get the GPT-4 judgments

   First, fill in your GPT-4 API key in `config/multi-dimension.json`.

   Then, modify and run the following script to get the judgments of the target LLM.

   ```bash
   export OPENAI_API_KEY="xx" # YOUR_API_KEY
   export OPENAI_MAX_CONCURRENCY=8
   export OPENAI_API_BASE="xx" # YOUR_API_BASE
   MODEL_NAME="FuseChat-Llama-3.1-8B-Instruct"
   
   bash eval.sh ${MODEL_NAME}
   ```

   The judgment file will be stored in `outputs/${MODEL_NAME}`.

## Acknowledgement
The codebase is adapted from [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval).
