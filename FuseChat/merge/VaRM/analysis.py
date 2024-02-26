import json
import torch
from fire import Fire
from transformers import AutoModelForCausalLM
from collections import defaultdict


def main(model1_path, model2_path, save_path, merge_type):
    # Load models
    model1 = AutoModelForCausalLM.from_pretrained(model1_path)
    model2 = AutoModelForCausalLM.from_pretrained(model2_path)

    # Get state_dicts
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Initialize variables
    total_param_num = 0
    total_diff_num = 0
    aveage_var_degree = 0
    key_to_params = defaultdict(float)
    module_to_params = defaultdict(float)
    layer_to_params = defaultdict(float)

    # Analyze parameters
    for key in state_dict1.keys():
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        total_param_num += param1.numel()
        if merge_type == "default":
            diff = (param2 - param1).sum().item()
        elif merge_type == "square":
            diff = torch.sum((param2 - param1) ** 2).item()
        elif merge_type == "abs":
            diff = torch.abs(param2 - param1).sum().item()
        else:
            raise NotImplementedError
        key_to_params[key] = diff / param1.numel()

        if "layer" in key:
            layer_num = key.split(".layers.")[1].split(".")[0]
            layer_to_params[layer_num] += key_to_params[key]
            if "self_attn" in key:
                module_to_params[f"model.layers.{layer_num}.self_attn"] += key_to_params[key]
            if "mlp" in key:
                module_to_params[f"model.layers.{layer_num}.mlp"] += key_to_params[key]
            if "input_layernorm" in key:
                module_to_params[f"model.layers.{layer_num}.input_layernorm"] += key_to_params[key]
            if "post_attention_layernorm" in key:
                module_to_params[f"model.layers.{layer_num}.post_attention_layernorm"] += key_to_params[key]
        
        if any([name in key for name in ["model.embed_tokens","model.norm", "lm_head"]]):
            module_to_params[key] = key_to_params[key]

        total_diff_num += torch.sum((param1 != param2)).item()
        aveage_var_degree += diff

    # Calculate average variation degree
    aveage_var_degree /= total_param_num

    # Save results
    with open(save_path, "w") as f:
        json.dump({
            "aveage_var_degree": aveage_var_degree,
            "total_diff_num": total_diff_num,
            "total_param_num": total_param_num,
            "diff_rate": total_diff_num / total_param_num * 100,
            "layer_to_params": dict(layer_to_params),
            "key_to_params": dict(key_to_params),
            "module_to_params": dict(module_to_params)
        }, f, indent=4)


    print({
        "aveage_var_degree": aveage_var_degree,
        "total_diff_num": total_diff_num,
        "total_param_num": total_param_num,
        "diff_rate": total_diff_num / total_param_num * 100
    })


if __name__ == "__main__":
    Fire(main)
