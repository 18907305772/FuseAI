# Evaluation

We use the evaluation code from [SkyThought](https://github.com/NovaSky-AI/SkyThought/tree/main/skythought/tools) and make a slight modification to support evaluation for FuseO1-Preview.

Example Usage

```sh
python3 eval.py --model FuseAI/FuseO1-DeekSeekR1-QwQ-SkyT1-32B-Preview --evals=AIME,MATH500,GSM8K --tp=8 --output_file=results.txt --temperatures 0.7
```