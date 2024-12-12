## Evaluation for HumanEval(+) and MBPP(+)

This folder contains the code and scripts to evaluate the performance of the **FuseChat** series models on [**EvalPlus**](https://github.com/evalplus/evalplus) benchmark, which includes HumanEval(+) and MBPP(+) datasets. These datasets are designed to test code generation capabilities under varied conditions.

### Setup

Please refer to [**EvalPlus**](https://github.com/evalplus/evalplus) for detailed setup instructions. Install the required packages using:

```bash
pip install evalplus --upgrade
pip install -r requirements.txt
```

### Evaluation

```bash
OUTPUT_DIR=results # path/to/save_results
MODEL_DIR="FuseAI/FuseChat-Llama-3.1-8B-Instruct" # path/to/model
TP=1 # number of GPUs for vllm inference
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/evalplus
bash test.sh ${MODEL_DIR} ${TP} ${OUTPUT_DIR}/evalplus ${GENERATE}
```

### Troubleshooting

If you experience poor network conditions that prevent downloading models directly from the repository during runtime, you can switch to using locally downloaded models. 

- **Download the Model**: Ensure you have the model files downloaded to your local machine.
- **Update the `MODEL_MAPPING`**: Modify the MODEL_MAPPING in `generate.py`
    ```python
    MODEL_MAPPING = {
        "llama3": {
            "chat": "/path/to/your/local/FuseChat-Llama-3.1-8B-Instruct",
            # ...
        },
    }
    ```