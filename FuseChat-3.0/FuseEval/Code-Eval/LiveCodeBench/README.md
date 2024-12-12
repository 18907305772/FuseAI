## Evaluation for LiveCodeBench

This folder contains the code and scripts to evaluate the performance of the **FuseChat** series models on [**LiveCodeBench**](https://github.com/LiveCodeBench/LiveCodeBench) benchmark, which provides holistic and contamination-free evaluation of coding capabilities of LLMs. Particularly, LiveCodeBench continuously collects new problems over time from contests across three competition platforms -- LeetCode, AtCoder, and CodeForces.

### Setup

```bash
# Make sure the CUDA version > 12.0.
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Please refer to [**LiveCodeBench**](https://github.com/LiveCodeBench/LiveCodeBench) for details.

### Evaluation

```bash
OUTPUT_DIR=results # path/to/save_results
MODEL_DIR="FuseAI/FuseChat-Llama-3.1-8B-Instruct" # path/to/model
MODEL_TYPE="meta-llama/Meta-Llama-3.1-8B-Instruct" 
TP=1 # number of GPUs for vllm inference
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/livecodebench
bash test.sh ${MODEL_DIR} ${TP} ${OUTPUT_DIR}/livecodebench ${MODEL_TYPE}
```

### Troubleshooting

If you experience poor network conditions that prevent downloading models or datasets directly from the repository during runtime, you can switch to using locally downloaded models. 

- **Download the Model and Dataset**: Ensure you have the model and dataset files downloaded to your local machine.
- **Update the Model Path**: Modify the "format_prompt_generation" function in `lcb_runner/prompts/code_generation.py`
    ```python
    tokenizer = AutoTokenizer.from_pretrained(
            "path/to/meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
    ```
- **Update the Dataset Path**: Modify the "load_code_generation_dataset" function in `lcb_runner/benchmarks/code_generation.py`
    ```python
    dataset = load_dataset("path/to/livecodebench/code_generation", split="test")
    ```
