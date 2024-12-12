## Evaluation for LiveBench

### Evaluation
You can evaluate FuseChat series model with the following command:
```bash
# FuseChat-Llama-3.1-8B-Instruct
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
MODEL_DIR_OR_PREFIX="FuseAI"
MODEL_NAME="FuseChat-Llama-3.1-8B-Instruct"

# prepare evaluation dataset
python livebench/download_questions.py
# Run evaluation
bash run.sh  ${MODEL_DIR_OR_PREFIX} ${MODEL_NAME}
# Run evaluation with vllm
#bash run_vllm.sh  ${MODEL_DIR_OR_PREFIX} ${MODEL_NAME}
```

## Acknowledgement
The codebase is adapted from [LiveBench/LiveBench](https://github.com/LiveBench/LiveBench).
