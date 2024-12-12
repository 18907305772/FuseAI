import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ref-yaml", type=str, default="configs/llama-3-alpacaeval.yaml")
parser.add_argument("--model-name", type=str, default="FuseChat-Llama-3.1-8B-Instruct")
parser.add_argument("--model-root-path", type=str, default="ckpt")
args = parser.parse_args()


with open(args.ref_yaml, 'r') as file:
    data = yaml.safe_load(file)

if "llama-3" in args.model_name.lower():
    data[args.model_name] = data.pop("Meta-Llama-3.1-8B-Instruct")
elif "gemma" in args.model_name.lower():
    data[args.model_name] = data.pop("gemma-2-9b-it")
elif "qwen2" in args.model_name.lower():
    data[args.model_name] = data.pop("Qwen2.5-7B-Instruct")
data[args.model_name]["completions_kwargs"]["model_kwargs"]["model_root_path"] = args.model_root_path + "/"
data[args.model_name]["completions_kwargs"]["model_name"] = args.model_name

with open(f"configs/{args.model_name}.yaml", 'w') as file:
    data = yaml.safe_dump(data, file)