### Requirements
You can install the required packages with the following command:

```bash
pip install -r requirements.txt 
pip install vllm==v0.6.3.post1
pip install transformers==4.43.1
```

### Evaluation

We use the evaluation code from [SkyThought](https://github.com/NovaSky-AI/SkyThought/tree/main/skythought/tools) and make a slight modification to support evaluation for FuseO1-Preview.

You can evaluate with the following command:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python3 eval.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=GPQA,MMLUPro,MMLU --tp=8 --output_file=results.txt --temperatures 0.7

python3 eval.py --model Qwen/QwQ-32B-Preview --evals=GPQA,MMLUPro,MMLU --tp=8 --output_file=results.txt --temperatures 0.7

python3 eval.py --model NovaSky-AI/Sky-T1-32B-Preview --evals=GPQA,MMLUPro,MMLU --tp=8 --output_file=results.txt --temperatures 0.7

python3 eval.py --model Qwen/Qwen2.5-32B-Instruct --evals=GPQA,MMLUPro,MMLU --tp=8 --output_file=results.txt --temperatures 0.7

python3 eval.py --model FuseAI/FuseO1-DeepSeekR1-Qwen2.5-Instruct-32B-Preview --evals=GPQA,MMLUPro,MMLU --tp=8 --output_file=results.txt --temperatures 0.7

python3 eval.py --model FuseAI/FuseO1-DeepSeekR1-QwQ-32B-Preview --evals=GPQA,MMLUPro,MMLU --tp=8 --output_file=results.txt --temperatures 0.7

python3 eval.py --model FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-Flash-32B-Preview --evals=GPQA,MMLUPro,MMLU --tp=8 --output_file=results.txt --temperatures 0.7

python3 eval.py --model FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview --evals=GPQA,MMLUPro,MMLU --tp=8 --output_file=results.txt --temperatures 0.7
```