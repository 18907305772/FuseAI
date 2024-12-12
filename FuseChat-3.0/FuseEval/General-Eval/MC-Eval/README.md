## Evaluation for Multiple-Choice Benchmarks
Benchmarks support: MMLU-PRO, MMLU, GPQA. We evalute all our models under zero-shot COT setting.
### Evaluation
You can evaluate FuseChat series model with the following command:
```bash
# FuseChat-Llama-3.1-8B-Instruct
export CUDA_VISIBLE_DEVICES="0"
OUTPUT_DIR=results # path/to/save_results
MODEL_NAME_OR_PATH="FuseAI/FuseChat-Llama-3.1-8B-Instruct"
mkdir -p ${OUTPUT_DIR}

bash test.sh ${OUTPUT_DIR}/MMLU-Pro ${MODEL_NAME_OR_PATH} "mmlu_pro"

bash test.sh ${OUTPUT_DIR}/MMLU ${MODEL_NAME_OR_PATH} "mmlu"

bash test.sh ${OUTPUT_DIR}/gpqa_diamond ${MODEL_NAME_OR_PATH} "gpqa_diamond"

```

## Acknowledgement
The codebase is adapted from [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro).
