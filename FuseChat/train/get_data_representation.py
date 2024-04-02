"""Get representation for sft training data."""

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import load_from_disk
from train import LazySupervisedDataset, SupervisedDataset
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forward for each teacher model to get logits of each token."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models. It is the base model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The input data dir. Should contain the training files.",
    )
    parser.add_argument(
        "--tknz_dataset_path",
        type=str,
        required=True,
        help="The local dir to save tknzed data.",
    )
    parser.add_argument(
        "--dataset_save_dir",
        type=str,
        required=True,
        help="The local dir to save processed data.",
    )
    parser.add_argument(
        "--dataset_sample_prop",
        type=float,
        default=None,
        help="The prop to sample dataset. Debugging only.",
    )
    parser.add_argument(
        "--dataset_split_num",
        type=int,
        default=None,
        help="The number to split dataset.",
    )
    parser.add_argument(
        "--dataset_index", type=int, default=None, help="The index of current dataset."
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache dir.")
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="model max length.",
    )
    parser.add_argument(
        "--load_in_half",
        type=str,
        default="none",
        help="none/fp16/bf16",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to do data loading.",
    )
    parser.add_argument(
        "--top_k_logits", type=int, default=10, help="The number of logit for saving."
    )
    parser.add_argument(
        "--save_per_token_metric",
        action="store_true",
        help="Save per token metric.",
    )
    parser.add_argument(
        "--no_assert",
        action="store_true",
        help="Delete the assert.",
    )
    parser.add_argument(
        "--conv_temp", type=str, default="vicuna", help="The conversation template."
    )
    parser.add_argument(
        "--flash_attn_transformers", action="store_true", help="Use flash attention 2."
    )
    parser.add_argument(
        "--mask_instruction", action="store_true", help="Mask instruction in the data."
    )
    parser.add_argument("--device_map", type=str, default=None, help="The device map.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(42)
    print("Get data representation.")
    print(f"Data processing args: {args}")
    data_json = json.load(open(args.data_path, "r"))

    if args.mask_instruction is False:
        print(
            "WARNING: We will not mask instruction in the dialog since we set 'mask_instruction=False'."
        )

    if args.dataset_sample_prop is not None:
        print(f"Sample prop: {args.dataset_sample_prop}.")
        select_size = int(len(data_json) * args.dataset_sample_prop)
        data_json = random.sample(data_json, select_size)
        print(f"Select size: {len(data_json)}")

    if args.dataset_split_num is not None:
        args.dataset_split_num = int(args.dataset_split_num)
        args.dataset_index = int(args.dataset_index)
        print(
            f"Split num: {args.dataset_split_num}; Split index: {args.dataset_index}."
        )
        select_size = int(len(data_json) / args.dataset_split_num)
        start_index = args.dataset_index * select_size
        end_index = (
            (args.dataset_index + 1) * select_size
            if args.dataset_index + 1 < args.dataset_split_num
            else len(data_json)
        )
        select_dataset = data_json[start_index:end_index]
        print(
            f"start_index: {start_index}, end_index: {end_index}, select_size: {len(select_dataset)}"
        )
        data_json = select_dataset

    # 1. tokenize the dataset for the model.
    trust_remote_code = False
    tknz_trust_remote_code = False
    use_fast = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        trust_remote_code=tknz_trust_remote_code,
        use_fast=use_fast,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if os.path.exists(args.tknz_dataset_path):
        tknz_hfdataset = load_from_disk(args.tknz_dataset_path)
    else:
        tknz_dataset = SupervisedDataset(
            data_json, tokenizer, args.conv_temp, args.mask_instruction
        )
        tknz_data_dict = {"input_ids": [], "labels": [], "attention_mask": []}
        for item in tknz_dataset:
            tknz_data_dict["input_ids"].append(item["input_ids"])
            tknz_data_dict["labels"].append(item["labels"])
            tknz_data_dict["attention_mask"].append(item["attention_mask"])
        tknz_hfdataset = HFDatasetDict({"train": HFDataset.from_dict(tknz_data_dict)})
        tknz_hfdataset.save_to_disk(args.tknz_dataset_path)

    # 3. get logits of the dataset for the model.
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    compute_dtype = (
        torch.bfloat16
        if args.load_in_half == "bf16"
        else (torch.float16 if args.load_in_half == "fp16" else torch.float32)
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        use_flash_attention_2=True if args.flash_attn_transformers else False,
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
        device_map=args.device_map,
    )

    if args.load_in_half == "fp16":
        tensor_type = torch.float16
    elif args.load_in_half == "bf16":
        tensor_type = torch.bfloat16
    else:
        tensor_type = torch.float32
    if torch.cuda.is_available() and args.device_map is None:
        model = model.cuda()
    model.eval()
    collate_function = DataCollatorWithPadding(tokenizer)

    def dict_to_list(examples):
        return [
            {key: examples[key][i] for key in examples}
            for i in range(len(examples[next(iter(examples))]))
        ]

    def forward_for_logits(examples):
        features = dict_to_list(examples)
        features = collate_function(features)
        if model.device.type == "cuda":
            input_ids = features["input_ids"].cuda()
            attention_mask = features["attention_mask"].cuda()
            labels = features["labels"].cuda()
        else:
            input_ids = features["input_ids"]
            attention_mask = features["attention_mask"]
            labels = features["labels"]
        with torch.no_grad():  # logits[-1] is not used; labels[0] is not used;
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.to(torch.float16)
            metric_ce = (
                F.cross_entropy(
                    logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                    labels[..., 1:].contiguous().view(-1),
                    reduction="none",
                )
                .view(logits.size(0), -1)
                .to(torch.float16)
            )
            if args.save_per_token_metric:
                examples["per_step_metric_ce"] = metric_ce.cpu()  # [bs, 2047]
            if args.mask_instruction is True:
                metric_ce = (metric_ce * labels.ne(IGNORE_TOKEN_ID)[..., 1:]).sum(
                    dim=-1
                ) / labels.ne(IGNORE_TOKEN_ID)[..., 1:].sum(dim=-1)
            else:
                metric_ce = (metric_ce * attention_mask[..., 1:]).sum(
                    dim=-1
                ) / attention_mask[..., 1:].sum(dim=-1)
            logits = logits.cpu()
            metric_ce = metric_ce.cpu()
            if not args.no_assert:
                assert not bool(torch.isnan(logits).any().item())
                assert not bool(torch.isnan(metric_ce).any().item())
            input_ids.cpu()
            del input_ids
            attention_mask.cpu()
            del attention_mask
            labels.cpu()
            del labels
        if args.top_k_logits:
            top_k_logits, top_k_indices = torch.topk(logits.cuda(), k=args.top_k_logits)
            top_k_logits = top_k_logits.cpu()
            top_k_indices = top_k_indices.cpu()
            examples["per_step_logits"] = top_k_logits
            examples["per_step_indices"] = top_k_indices
        else:
            print("ERROR: Saving all logits is too large!")
            raise ValueError
        examples["metric_ce"] = metric_ce
        return examples

    tknz_with_logits_hfdataset = tknz_hfdataset.map(
        forward_for_logits,
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=1000,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc="Get data representation.",
    )

    tknz_with_logits_hfdataset.save_to_disk(args.dataset_save_dir)
