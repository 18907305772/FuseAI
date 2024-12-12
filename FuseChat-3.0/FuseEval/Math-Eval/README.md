### Requirements
You can install the required packages with the following command:
```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
```

### Evaluation
You can evaluate FuseChat series model with the following command:
```bash
# FuseChat-Llama-3.1 Series
PROMPT_TYPE="llama3-math-cot"
# FuseChat-Llama-3.1-8B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="FuseAI/FuseChat-Llama-3.1-8B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

```

## Acknowledgement
The codebase is adapted from [QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).
