<p align="center" width="100%">
</p>

<div id="top" align="center">

FuseO1-Preview: System-II Reasoning Fusion of LLMs
-----------------------------

<h4> |<a href="https://arxiv.org/abs/2408.07990"> 📑 Paper </a> |
<a href="https://github.com/fanqiwan/FuseAI"> 🐱 GitHub Repo </a> |
<a href="https://huggingface.co/FuseAI"> 🤗 Hugging Face </a> |
</h4>

<!-- **Authors:** -->

_Fanqi Wan, Longguang Zhong, Ziyi Yang_


<!-- **Affiliations:** -->

_Sun Yat-sen University_

</div>

## Overview

FuseO1-Preview is our initial endeavor to enhance the System-II reasoning capabilities of large language models (LLMs) through innovative model fusion techniques. By employing advanced [SCE](https://arxiv.org/abs/2408.07990) merging methodologies, we integrate multiple open-source o1-like LLMs into a unified model. Our goal is to incorporate the distinct knowledge and strengths from different reasoning LLMs into a single, unified model with strong System-II reasoning abilities, particularly in mathematics, coding, and science domains.

To achieve this, we conduct two types of model merging:

- **Long-Long Reasoning Merging**: This approach involves model fusion across LLMs that utilize long-CoT reasoning, with the goal of enhancing long-CoT reasoning capabilities. The resulted [FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview) achieves an accuracy of **60.00 on AIME24**,  demonstrating significant performance improvements compared to the o1-preview model (44.60) and approaching the performance of the o1-mini model (63.60).
- **Long-Short Reasoning Merging**: This approach involves model fusion between long-CoT and short-CoT LLMs, aiming to improve reasoning capabilities in both long and short reasoning processes. The resulted [FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview) is capable of utilizing both long and short reasoning processes and demonstrates relatively strong performance in long reasoning tasks.

| Model Name | Model Type | Source LLMs | Model Link |
| ----- | ---- |---- |
| FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview | Long-Long Reasoning Merge | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B), [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview)  | [🤗 Hugging Face](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview), [NovaSky-AI/Sky-T1-32B-Preview](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview) |
| FuseAI/FuseO1-DeekSeekR1-QwQ-32B-Preview | Long-Long Reasoning Merge | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B), [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview)  | [🤗 Hugging Face](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview) | [🤗 Hugging Face](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-QwQ-32B-Preview) |
| FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview | Long-Short Reasoning Merge | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B),[Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | [🤗 Hugging Face](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview) |


## Long-Long Reasoning Merging

We conduct experiments on these folloing long-cot LLMs.

- [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview)
- [NovaSky-AI/Sky-T1-32B-Preview](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview)

To reproduce the merged [FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview) model, using the script below.

```sh
cd FuseAI/FuseO1-Preview/mergekit
pip3 install -e .
model_save_dir=xx # your path to save the merged models
mergekit-yaml fuseo1_configs/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview.yaml ${model_save_dir}/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview
```

To reproduce the merged [FuseAI/FuseO1-DeekSeekR1-QwQ-32B-Preview](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-QwQ-32B-Preview) model, using the script below.

```sh
cd FuseAI/FuseO1-Preview/mergekit
pip3 install -e .
model_save_dir=xxx # your path to save the merged models
mergekit-yaml fuseo1_configs/FuseO1-DeekSeekR1-QwQ-32B-Preview.yaml ${model_save_dir}/FuseO1-DeekSeekR1-QwQ-32B-Preview
```

We provide the example code to use FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview.

```python3
from vllm import LLM, SamplingParams

llm = LLM(model="FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview", tensor_parallel_size=8)
sampling_params = SamplingParams(max_tokens=32768, temperature=0.7, stop=["<｜end▁of▁sentence｜>", "<｜User｜>"], stop_token_ids=[151643, 151644])

conversations = [
    [
        {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
        {"role": "user", "content": "Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$."},
    ],
]

responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)

for response in responses:
    print(response.outputs[0].text.strip())
```

## Long-Short Reasoning Merging

We conduct experiments on these folloing long-cot and short-cot LLMs.

- [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)

To reproduce the merged [FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview) model, using the script below.

```sh
cd FuseAI/FuseO1-Preview/mergekit
pip3 install -e .
model_save_dir=xxx # your path to save the merged models
mergekit-yaml fuseo1_configs/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview.yaml ${model_save_dir}/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview
```

We provide the code to use FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview.

```python3
from vllm import LLM, SamplingParams

llm = LLM(model="FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview", tensor_parallel_size=8)
sampling_params = SamplingParams(max_tokens=32768, temperature=0.7, stop=["<｜end▁of▁sentence｜>", "<｜User｜>"], stop_token_ids=[151643, 151644])

conversations = [
    [
        {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
        {"role": "user", "content": "Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$."},
    ],
]

responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)

for response in responses:
    print(response.outputs[0].text.strip())
```

## Evaluation Results

We test the resulted models on three kinds of benchmarks, including **Math Reasoning**, **Code Reasoning** , and **Scientific Reasoning**.

Math Reasoning
  - AIME24
  - MATH500
  - GSM8K

Scientific Reasoning
  - GPQA-Diamond
  - ARC-Challenge
  - MMLU-Pro
  - MMLU
  

Code Reasoning
  - LiveCodeBench

The [evaluation code](https://github.com/fanqiwan/FuseAI/tree/main/FuseO1-Preview/evaluation) is modified from [SkyThought](https://github.com/NovaSky-AI/SkyThought). In our evaluation, we set the temperature to 0.7 (sampling) and the max_tokens to 32768.

The evaluation results are shown in the table below:

|  | AIME24 | MATH500 | GSM8K | GPQA-Diamond | ARC-Challenge | MMLU-Pro | MMLU | LiveCodeBench |
|:-| ------ | ------- | ----- | ------------ | ------------- | -------- | ---- | ------------- |
| o1-preview | 44.60 | 85.50 | - | 73.30 | - | - | 90.80 | - |
| o1-mini | 63.60 | 90.00 | - | 60.00 | - | 80.30 | 85.20| 53.80 |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | 46.67 | 88.20 | - | 57.58 | - | - | - | - |
| [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) |43.33 | 87.80 | 95.45 | 49.49 | 95.73 | 63.49 | 85.19 | 51.86 |
| [NovaSky-AI/Sky-T1-32B-Preview](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview) | 43.33 | 86.80 | 95.15 | 50.51 | 95.56 | 65.80 | 82.71 | 51.66 |
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | 20.00 | 81.60 | 93.63 | 46.46 | 95.22 | 56.27 | 79.63 | 48.53 |
| [FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-Qwen2.5-Instruct-32B-Preview) | 46.67 | 87.20 | - | - | - | - | - | - |
| [FuseAI/FuseO1-DeekSeekR1-QwQ-32B-Preview](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-QwQ-32B-Preview) | 56.67 | - | - | - | - | - | - | - |
| [FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview](https://huggingface.co/FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview) | 60.00 | - | - | - | - | - | - | - |

## Future Works

This work is our first attempt effort to achieve knowledge fusion of System-II reasoning LLMs through a model merging approach, which is limited to LLMs with identical scale and architecture. In future work, we plan to employ our [explicit model fusion](https://arxiv.org/abs/2401.10491) method, based on multi-teacher knowledge distillation, and our [implici model fusion](https://arxiv.org/abs/2412.03187) method, which utilizes weighted-reward preference optimization for LLMs with different scales and architectures.
Furthermore, we intend to explore the combination of knowledge fusion with reinforcement learning (RL) methods, which have been demonstrated as the most effective approach for enhancing reasoning abilities. Stay tuned for the next version of FuseO1!

## Citations

```
@article{wan2024fusechat,
  title={Fusechat: Knowledge fusion of chat models},
  author={Wan, Fanqi and Zhong, Longguang and Yang, Ziyi and Chen, Ruijun and Quan, Xiaojun},
  journal={arXiv preprint arXiv:2408.07990},
  year={2024}
}
```