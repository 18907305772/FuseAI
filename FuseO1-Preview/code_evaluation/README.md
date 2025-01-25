### Requirements
You can install the required packages with the following command:

```bash
pip install -r requirements.txt 
pip install vllm==v0.6.3.post1
pip install transformers==4.43.1
```

### Dataset

```bash
huggingface-cli download --repo-type dataset livecodebench/code_generation_lite --local-dir code_generation_lite
```

### Evaluation

You can evaluate with the following command:

```bash
bash ./sh/evaluate_ds.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B ./lcb_results/DeepSeek-R1-Distill-Qwen-32B

bash ./sh/evaluate_qwq.sh Qwen/QwQ-32B-Preview ./lcb_results/QwQ-32B-Preview

bash ./sh/evaluate_ds.sh FuseAI/FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview ./lcb_results/FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview

bash ./sh/evaluate_ds.sh FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-Flash-32B-Preview ./lcb_results/FuseO1-DeepSeekR1-QwQ-SkyT1-Flash-32B-Preview

bash ./sh/evaluate_ds.sh FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview ./lcb_results/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview
```