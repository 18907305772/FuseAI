### Requirements
You can install the required packages with the following command:

```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==v0.6.3.post1
pip install transformers==4.43.1
```

### Evaluation

You can evaluate with the following command:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

model_path="<YOUR_MODEL_PATH>"
result_dir="<YOUR_RESULT_DIR>"

# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

for model_name in "DeepSeek-R1-Distill-Qwen-32B"
do
for ((seed=0; seed<=31; seed++)); do
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="deepseek-math-cot"
  bash ./sh/eval_aime.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done
done

for model_name in "DeepSeek-R1-Distill-Qwen-32B"
do
  seed=2
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="deepseek-math-cot"
  bash ./sh/eval_others.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done

# Qwen/QwQ-32B-Preview

for model_name in "QwQ-32B-Preview"
do
for ((seed=0; seed<=31; seed++)); do
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="qwen25-math-cot"
  bash ./sh/eval_aime.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done
done

for model_name in "QwQ-32B-Preview"
do
  seed=2
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="qwen25-math-cot"
  bash ./sh/eval_others.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done

# NovaSky-AI/Sky-T1-32B-Preview

for model_name in "Sky-T1-32B-Preview"
do
for ((seed=0; seed<=31; seed++)); do
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="sky-t1-math-cot"
  bash ./sh/eval_aime.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done
done

for model_name in "Sky-T1-32B-Preview"
do
  seed=2
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="sky-t1-math-cot"
  bash ./sh/eval_others.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done

# Qwen/Qwen2.5-32B-Instruct

for model_name in "Qwen2.5-32B-Instruct"
do
for ((seed=0; seed<=31; seed++)); do
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="qwen25-math-cot"
  bash ./sh/eval_aime.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done
done

for model_name in "Qwen2.5-32B-Instruct"
do
  seed=2
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="qwen25-math-cot"
  bash ./sh/eval_others.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done

# FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview

for model_name in "FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview"
do
for ((seed=0; seed<=31; seed++)); do
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="deepseek-math-cot"
  bash ./sh/eval_aime.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done
done

for model_name in "FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview"
do
  seed=2
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="deepseek-math-cot"
  bash ./sh/eval_others.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done

# FuseAI/FuseO1-DeepSeekR1-QwQ-32B-Preview
or model_name in "FuseO1-DeepSeekR1-QwQ-32B-Preview"
do
for ((seed=0; seed<=31; seed++)); do
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="deepseek-math-cot"
  bash ./sh/eval_aime.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done
done

for model_name in "FuseO1-DeepSeekR1-QwQ-32B-Preview"
do
  seed=2
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="deepseek-math-cot"
  bash ./sh/eval_others.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done

# FuseAI/FuseO1-DeepSeekR1-Qwen2.5-Instruct-32B-Preview

for model_name in "FuseO1-DeepSeekR1-Qwen2.5-Instruct-32B-Preview"
do
for ((seed=0; seed<=31; seed++)); do
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="deepseek-math-cot"
  bash ./sh/eval_aime.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done
done

for model_name in "FuseO1-DeepSeekR1-Qwen2.5-Instruct-32B-Preview"
do
  seed=2
  mkdir -p "${result_dir}/${model_name}-seed${seed}"
  prompt_type="deepseek-math-cot"
  bash ./sh/eval_others.sh $prompt_type ${model_path}/${model_name} "${result_dir}/${model_name}-seed${seed}" $seed
done
```