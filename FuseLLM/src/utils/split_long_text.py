"""1. Split long text in the dataset."""

import argparse

from datasets import Dataset, DatasetDict, Features, load_dataset, load_from_disk
from src.utils.others import (
    get_logger,
    get_tokenizer,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Split long text into shorter one.")
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models. It is the base model.",
    )
    parser.add_argument(
        "--blending_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models. It is the blending model.",
    )
    parser.add_argument(
        "--another_blending_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models. It is the blending model.",
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
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache dir.")
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
        help="The number of processes to do data loading",
    )
    parser.add_argument(
        "--dataset_sample_prop",
        type=float,
        default=None,
        help="The prop to sample dataset. Debugging only.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Data processing args: {args}")
    base_tokenizer, _ = get_tokenizer(
        args.base_model_name_or_path, args.cache_dir, args.block_size
    )
    blending_tokenizer, _ = get_tokenizer(
        args.blending_model_name_or_path, args.cache_dir, args.block_size
    )
    another_blending_tokenizer, _ = get_tokenizer(
        args.another_blending_model_name_or_path, args.cache_dir, args.block_size
    )
    dataset_mapping = load_from_disk(args.dataset)
    if args.dataset_sample_prop is not None:
        logger.info(f"Sample prop: {args.dataset_sample_prop}.")
        for k, v in dataset_mapping.items():
            select_size = int(len(v) * args.dataset_sample_prop)
            select_dataset = v.select(range(select_size))
            dataset_mapping[k] = select_dataset
            logger.info(f"{k}: select_size: {len(select_dataset)}")

    threshold = args.block_size - 48
    logger.info(f"Maximum length: {threshold}")

    # 0. split long text in the dataset based on all models
    def truncate_sequences(input_ids, max_length):
        truncated_sequences = []
        for i in range(0, len(input_ids), max_length):
            truncated_sequences.append(input_ids[i : i + max_length])
        return truncated_sequences

    def split_text(examples):
        split_texts_batch = []
        for text in examples["text"]:
            base_tokenized_text = base_tokenizer(text, add_special_tokens=False)[
                "input_ids"
            ]
            blending_tokenized_text = blending_tokenizer(
                text, add_special_tokens=False
            )["input_ids"]
            another_blending_tokenized_text = another_blending_tokenizer(
                text, add_special_tokens=False
            )["input_ids"]
            if (
                len(base_tokenized_text) > threshold
                or len(blending_tokenized_text) > threshold
                or len(another_blending_tokenized_text) > threshold
            ):
                max_length = max(
                    len(base_tokenized_text),
                    len(blending_tokenized_text),
                    len(another_blending_tokenized_text),
                )
                if len(base_tokenized_text) == max_length:
                    max_tokenized_text = base_tokenized_text
                    max_tokenizer = base_tokenizer
                elif len(blending_tokenized_text) == max_length:
                    max_tokenized_text = blending_tokenized_text
                    max_tokenizer = blending_tokenizer
                else:
                    max_tokenized_text = another_blending_tokenized_text
                    max_tokenizer = another_blending_tokenizer
                truncated_sequences = truncate_sequences(
                    max_tokenized_text, max_length=threshold
                )
                split_texts = [
                    max_tokenizer.decode(seq, skip_special_tokens=True)
                    for seq in truncated_sequences
                ]
                split_texts_batch.append(split_texts)
            else:
                split_texts_batch.append([text])
        return {"text": split_texts_batch}

    split_dataset = dataset_mapping.map(
        split_text,
        batched=True,
        remove_columns=["text"],
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc="Split long text",
    )
    flat_data = dict()
    for k, v in split_dataset.items():
        flat_data[k] = Dataset.from_dict(
            {"text": [text for sublist in v["text"] for text in sublist]}
        )
    flat_dataset = DatasetDict(flat_data)
    flat_dataset.save_to_disk(args.dataset_save_dir)
