"""Tokenize and patch dataset before training (Pure pretraining recipe, not for distilling)."""

from itertools import chain
from typing import List, Dict
from datasets import Features, load_dataset, load_from_disk
import argparse

from src.utils.others import (
    get_logger,
    get_tokenizer,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize and patch dataset before training.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The input data dir. Should contain the training files."
    )
    parser.add_argument(
        "--dataset_save_dir",
        type=str,
        default=None,
        help="The local dir to save processed data."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The cache dir."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=80,
        help="The number of processes to do data loading"
    )
    parser.add_argument(
        "--content_key",
        type=str,
        default="text",
        help="The key to fetch text"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tokenizer, kwargs = get_tokenizer(args.model_name_or_path, args.cache_dir, args.block_size)
    logger.info(f"Data processing args: {args}")
    dataset_kwargs = dict(
        data_dir=args.dataset,
        content_key=args.content_key,
        block_size=args.block_size,
    )
    dataset_mapping = load_from_disk(args.dataset)

    def preprocess_pretrain_dataset(examples):
        # build grouped texts with format `X1 <eos> X2 <eos> X3 <eos>...`
        text: List[str] = examples[dataset_kwargs["content_key"]]
        text = [x + tokenizer.eos_token for x in text]  # add eos in the end
        tknz_text = tokenizer(text, add_special_tokens=False)

        concatenated_examples = {k: list(chain(*tknz_text[k])) for k in tknz_text.keys()}
        total_length = len(concatenated_examples[list(tknz_text.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        block_size = dataset_kwargs["block_size"]
        # total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    column_names = dataset_mapping["train"].column_names
    logger.info(f"Remove {column_names} in the final file.")
    tokenized_datasets = dataset_mapping.map(
        preprocess_pretrain_dataset,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc="Running tokenizer and grouping in chunks on dataset",
    )
    tokenized_datasets.save_to_disk(args.dataset_save_dir)
