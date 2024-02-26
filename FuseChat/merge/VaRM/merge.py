import json
from fire import Fire
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from copy import deepcopy

BASE_MODEL_NAME="openchat/openchat_3.5"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)


def get_merge_weights(weights, temp=1., use_sfmax=False):
    if use_sfmax:
        weights = torch.nn.functional.softmax(torch.tensor(weights/temp), dim=-1).numpy()
    else:
        weights = weights / sum(weights)
    return weights


def merge_with_total_wegihts(model_name_list, analysis_result, excluded_pattern=None, temp=1., use_sfmax=False, merge_type=None):
    models=[]
    diff_rate_list=[]

    for model_name in model_name_list:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(dtype=torch.bfloat16)
        models.append(model)
        with open(analysis_result, "r") as f:
            dict = json.load(f)
            diff_rate = dict["diff_rate"]
            avd = abs(dict["aveage_var_degree"])
            difff = diff_rate * avd
            diff_rate_list.append(difff)

    weights = np.array(diff_rate_list)
    weights = get_merge_weights(weights, temp=temp, use_sfmax=use_sfmax)
    print(weights)

    merged_model = deepcopy(models[0])

    for param_name, param in merged_model.named_parameters():
        if excluded_pattern != None and len(excluded_pattern) != 0:
            pattern = "|".join(map(re.escape, excluded_pattern))
            if re.search(pattern, param_name):
                continue
        weighted_param = sum(weight * model.state_dict()[param_name] for model, weight in zip(models, weights))

        # Convert the weighted parameters to bfloat16
        weighted_param = weighted_param.to(torch.bfloat16)

        # Update the parameter in the merged model
        param.data.copy_(weighted_param)

    return merged_model


def merge_with_model_wegihts(model_name_list, analysis_result, excluded_pattern=None, temp=1., use_sfmax=False, merge_type=None):
    models=[]
    aveage_var_degree_list=[]

    for model_name in model_name_list:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(dtype=torch.bfloat16)
        models.append(model)
        with open(analysis_result, "r") as f:
            dict = json.load(f)
            aveage_var_degree = dict["aveage_var_degree"]
            aveage_var_degree_list.append(aveage_var_degree)

    weights = np.array(aveage_var_degree_list)
    weights = get_merge_weights(weights, temp=temp, use_sfmax=use_sfmax)
    print(weights)

    merged_model = deepcopy(models[0])

    for param_name, param in merged_model.named_parameters():
        if excluded_pattern != None and len(excluded_pattern) != 0:
            pattern = "|".join(map(re.escape, excluded_pattern))
            if re.search(pattern, param_name):
                continue
        weighted_param = sum(weight * model.state_dict()[param_name] for model, weight in zip(models, weights))

        # Convert the weighted parameters to bfloat16
        weighted_param = weighted_param.to(torch.bfloat16)

        # Update the parameter in the merged model
        param.data.copy_(weighted_param)

    return merged_model


def merge_with_layer_wegihts(model_name_list, analysis_result, excluded_pattern=None, temp=1., use_sfmax=False, merge_type=None):
    models=[]
    layer_to_params_list=[]

    for model_name in model_name_list:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(dtype=torch.bfloat16)
        models.append(model)
        with open(analysis_result, "r") as f:
            dict = json.load(f)
            layer_to_params = dict["layer_to_params"]
            layer_to_params_list.append(layer_to_params)

    merged_model = deepcopy(models[0])

    for param_name, param in merged_model.named_parameters():
        if excluded_pattern != None and len(excluded_pattern) != 0:
            pattern = "|".join(map(re.escape, excluded_pattern))
            if re.search(pattern, param_name):
                continue 

        weights=np.array([1] * len(layer_to_params_list))

        if "layers" in param_name:
            layer_num = param_name.split(".layers.")[1].split(".")[0]
            weights = np.array([abs(ltp[layer_num]) for ltp in layer_to_params_list])

        weights = get_merge_weights(weights, temp=temp, use_sfmax=use_sfmax)
        print(f"param_name:{param_name} merge_weights:{weights}\n")
        weighted_param = sum(weight * model.state_dict()[param_name] for model, weight in zip(models, weights))

        # Convert the weighted parameters to bfloat16
        weighted_param = weighted_param.to(torch.bfloat16)

        # Update the parameter in the merged model
        param.data.copy_(weighted_param)

    return merged_model


def merge_with_module_wegihts(model_name_list, analysis_result, excluded_pattern=None, temp=1., use_sfmax=False, merge_type=None):
    models=[]
    module_to_params_list=[]

    for model_name in model_name_list:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(dtype=torch.bfloat16)
        models.append(model)
        with open(analysis_result, "r") as f:
            dict = json.load(f)
            module_to_params = dict["module_to_params"]
            module_to_params_list.append(module_to_params)

    merged_model = deepcopy(models[0])
    merged_weights=[]

    for param_name, param in merged_model.named_parameters():
        if excluded_pattern != None and len(excluded_pattern) != 0:
            pattern = "|".join(map(re.escape, excluded_pattern))
            if re.search(pattern, param_name):
                continue

        trimed_name = deepcopy(param_name)
        for module_name in ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]:
            if module_name in trimed_name:
                trimed_name = trimed_name[:trimed_name.find(module_name) + len(module_name)]
                break

        weights = np.array([abs(ktp[trimed_name]) for ktp in module_to_params_list])
        for ktp in module_to_params_list:
            if ktp[trimed_name] == 0:
                weights=np.array([1] * len(module_to_params_list))

        weights = get_merge_weights(weights, temp=temp, use_sfmax=use_sfmax)
        print(f"param_name:{param_name} merge_weights:{weights}\n")
        merged_weights.append(f"param_name:{param_name} merge_weights:{weights}\n")
        weighted_param = sum(weight * model.state_dict()[param_name] for model, weight in zip(models, weights))

        # Convert the weighted parameters to bfloat16
        weighted_param = weighted_param.to(torch.bfloat16)

        # Update the parameter in the merged model
        param.data.copy_(weighted_param)

    return merged_model


def merge_with_param_wegihts(model_name_list, analysis_result, excluded_pattern=None, temp=1., use_sfmax=False, merge_type=None):
    models=[]
    key_to_params_list=[]
    for model_name in model_name_list:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(dtype=torch.bfloat16)
        models.append(model)
        with open(analysis_result, "r") as f:
            dict = json.load(f)
            key_to_params = dict["key_to_params"]
            key_to_params_list.append(key_to_params)

    merged_model = deepcopy(models[0])

    for param_name, param in merged_model.named_parameters():
        if excluded_pattern != None and len(excluded_pattern) != 0:
            pattern = "|".join(map(re.escape, excluded_pattern))
            if re.search(pattern, param_name):
                continue

        weights = np.array([abs(ktp[param_name]) for ktp in key_to_params_list])

        for ktp in key_to_params_list:
            if ktp[param_name] == 0:
                weights=np.array([1] * len(key_to_params_list))
        weights = get_merge_weights(weights, temp=temp, use_sfmax=use_sfmax)
        print(f"layer:{param_name} merge_weights:{weights}\n")
        weighted_param = sum(weight * model.state_dict()[param_name] for model, weight in zip(models, weights))

        # Convert the weighted parameters to bfloat16
        weighted_param = weighted_param.to(torch.bfloat16)

        # Update the parameter in the merged model
        param.data.copy_(weighted_param)

    return merged_model


def merge_with_item_wegihts(model_name_list, analysis_result, excluded_pattern=None, temp=1., use_sfmax=False, merge_type=None):
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    models=[]
    for model_name in model_name_list:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(dtype=torch.bfloat16)
        models.append(model)

    merged_model = deepcopy(models[0]) 

    for param_name, param in merged_model.named_parameters():
        if excluded_pattern != None and len(excluded_pattern) != 0:
            pattern = "|".join(map(re.escape, excluded_pattern))
            if re.search(pattern, param_name):
                continue

        weights = []
        for model in models:
            if merge_type == "default":
                weight = model.state_dict()[param_name] - base_model.state_dict()[param_name]
            elif merge_type == "abs":
                weight = torch.abs(model.state_dict()[param_name] - base_model.state_dict()[param_name])
            elif merge_type == "squre":
                weight = (model.state_dict()[param_name] - base_model.state_dict()[param_name]) ** 2
            else:
                raise NotImplementedError
            weights.append(weight)
        sum_weights = torch.zeros_like(weights[0])
        for weight in weights:
            sum_weights += weight
        sum_weights[sum_weights < 1e-13] = 1e-13
        weights = [weight / sum_weights for weight in weights]
        weighted_param = sum(weight * model.state_dict()[param_name] for model, weight in zip(models, weights))

        # Convert the weighted parameters to bfloat16
        weighted_param = weighted_param.to(torch.bfloat16)

        # Update the parameter in the merged model
        param.data.copy_(weighted_param)

    return merged_model


def weighted_merge_models(model_name_list, weights, excluded_pattern=None):
    """
    Merge multiple Transformers models with weighted parameters.
    :param models: list of Transformers models
    :param weights: list of floats, weights for each model
    :return: Transformers model, merged model
    """
    models=[]
    for model_name in model_name_list:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(dtype=torch.bfloat16)
        models.append(model)

    # Clone the first model
    merged_model = deepcopy(models[0])

    # Iterate through the parameters of the merged model
    for param_name, param in merged_model.named_parameters():
        if excluded_pattern != None and len(excluded_pattern) != 0:
            pattern = "|".join(map(re.escape, excluded_pattern))
            if re.search(pattern, param_name):
                continue
        # Calculate the weighted average of the parameters across all models
        weighted_param = sum(weight * model.state_dict()[param_name] for model, weight in zip(models, weights))

        # Convert the weighted parameters to bfloat16
        weighted_param = weighted_param.to(torch.bfloat16)

        # Update the parameter in the merged model
        param.data.copy_(weighted_param)

    return merged_model

def main(merged_model_names, analysis_result=None, merged_model_save_dir=None, merge_method="linear", temp=1., use_sfmax=False, merge_type=None, linear_weights="1,1", excluded_pattern=""):
    merged_method_maps = {
        "avg_model": merge_with_model_wegihts,
        "avg_layer": merge_with_layer_wegihts,
        "avg_module": merge_with_module_wegihts,
        "avg_param": merge_with_param_wegihts,
        "avg_item": merge_with_item_wegihts,
        "avg_weight": merge_with_total_wegihts,
    }
    merged_model_names = merged_model_names.split(",")
    excluded_pattern = excluded_pattern.split(",")

    if merge_method=="linear":
        per = np.array([int(weight) for weight in linear_weights.split(",")])
        per = per / sum(per)
        print(f"merge weights: {per}")
        merged_model = weighted_merge_models(merged_model_names, per, excluded_pattern)
        per = [str(round(p, 2)) for p in per]
        merged_model.save_pretrained(merged_model_save_dir, use_bfloat16=True, max_shard_size="20GB", safe_serialization=False)
        tokenizer.save_pretrained(merged_model_save_dir)
    else:
        merged_model = merged_method_maps[merge_method](merged_model_names, analysis_result, excluded_pattern, temp=temp, use_sfmax=use_sfmax, merge_type=merge_type)
        merged_model.save_pretrained(merged_model_save_dir, use_bfloat16=True, max_shard_size="20GB", safe_serialization=False)
        tokenizer.save_pretrained(merged_model_save_dir)


if __name__ == "__main__":
    Fire(main)