"""2. Tokenize and then forward with all models for their logits."""

import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import DatasetDict, Features, load_dataset, load_from_disk
from src.utils.common import load_tokenizer_and_model
from src.utils.data_collator import DataCollatorForSeq2Seq
from src.utils.others import (
    IGNORE_TOKEN_ID,
    AttrDict,
    dict_to_list,
    get_logger,
    get_tokenizer,
    release_model_and_tensor,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forward for each teacher model to get logits of each token."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The input data dir. Should contain the training files.",
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
        "--model_max_length", type=int, default=2048, help="The model max length."
    )
    parser.add_argument(
        "--training_mode", type=str, default="full", help="full or qlora."
    )
    parser.add_argument(
        "--load_in_half", type=str, default="none", help="none or fp16 or bf16."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=80,
        help="The number of processes to do data loading.",
    )
    parser.add_argument(
        "--top_k_logits", type=int, default=10, help="The number of logit for saving."
    )
    parser.add_argument(
        "--save_per_token_metric", action="store_true", help="Save per token metric."
    )
    parser.add_argument("--no_assert", action="store_true", help="Delete the assert.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Data processing args: {args}")
    dataset_mapping = load_from_disk(args.dataset)
    if args.dataset_sample_prop is not None:
        logger.info(f"Sample prop: {args.dataset_sample_prop}.")
        for k, v in dataset_mapping.items():
            select_size = int(len(v) * args.dataset_sample_prop)
            select_dataset = v.select(range(select_size))
            dataset_mapping[k] = select_dataset
            logger.info(f"{k}: select_size: {len(select_dataset)}")
    if args.dataset_split_num is not None:
        args.dataset_split_num = int(args.dataset_split_num)
        args.dataset_index = int(args.dataset_index)
        logger.info(
            f"Split num: {args.dataset_split_num}; Split index: {args.dataset_index}."
        )
        for k, v in dataset_mapping.items():
            select_size = int(len(v) / args.dataset_split_num)
            start_index = args.dataset_index * select_size
            end_index = (
                (args.dataset_index + 1) * select_size
                if args.dataset_index + 1 < args.dataset_split_num
                else len(v)
            )
            select_dataset = v.select(range(start_index, end_index))
            dataset_mapping[k] = select_dataset
            logger.info(
                f"{k}: start_index: {start_index}, end_index: {end_index}, select_size: {len(select_dataset)}"
            )

    tokenizer, _ = get_tokenizer(
        args.model_name_or_path, args.cache_dir, args.model_max_length
    )

    def tokenize_dataset(examples):
        text: List[str] = examples["text"]
        text = [x + tokenizer.eos_token for x in text]  # add eos in the end
        tknz_text = tokenizer(
            text,
            add_special_tokens=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        tknz_text["labels"] = tknz_text["input_ids"].copy()
        return tknz_text

    tokenized_dataset = dataset_mapping.map(
        tokenize_dataset,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc="Tokenize the dataset.",
    )

    model_args = {
        "model_name_or_path": args.model_name_or_path,
        "cache_dir": args.cache_dir,
        "model_max_length": args.model_max_length,
        "training_mode": args.training_mode,
        "use_flash_attn": False,
    }

    _, model = load_tokenizer_and_model(AttrDict(model_args))
    if args.load_in_half == "fp16":
        model = model.half()
    elif args.load_in_half == "bf16":
        model = model.to(dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    collate_function = DataCollatorForSeq2Seq(
        tokenizer,
        padding="max_length",
        max_length=args.model_max_length,
        label_pad_token_id=IGNORE_TOKEN_ID,
    )

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
        with torch.no_grad():
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
                examples["per_step_metric_ce"] = metric_ce.cpu()
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
            logger.warning("Saving all logits is too large!")
            raise ValueError
        examples["metric_ce"] = metric_ce
        return examples

    logits_datasets = tokenized_dataset.map(
        forward_for_logits,
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=1000,
        num_proc=None,
        load_from_cache_file=True,
        desc="Forward and get logits of the dataset.",
    )
    release_model_and_tensor(model)
    logits_datasets.save_to_disk(args.dataset_save_dir)
