## Evaluation for MT-Bench

### Requirements
Please refer to [lm-sys/FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) for detailed setup instructions.

### Evaluation
The whole evaluation process contains two steps: inference and LLM judgments.

1. **Step I** inference on target LLM and get the results

    ```bash
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    MODEL_DIR=path/to/model
    MODEL_NAME="FuseChat-Llama-3.1-8B-Instruct"
  
    bash generate_hf.sh ${MODEL_DIR} ${MODEL_NAME} 4 1 # use hf backend (official)
    # bash generate_vllm.sh ${MODEL_DIR} ${MODEL_NAME} 4 1 # use vllm backend
    ```
    
    The answers will be saved in `data/model_answer` and ready for the LLM Judge process.

2. **Step II** get the GPT-4 judgments

   First, fill in your GPT-4 API key and base in `common.py`.

   Then, modify and run the following script to get the judgments of the target LLM.

   ```bash
   MODEL_NAME="FuseChat-Llama-3.1-8B-Instruct"
   
   bash eval.sh ${MODEL_NAME}
   ```

   The judgment file will be stored in `data/model_judge`.

## Acknowledgement
The codebase is adapted from [lm-sys/FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).
