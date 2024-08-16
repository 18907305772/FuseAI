"""3. Align tokens and their logits from different models."""

import argparse
import json

from datasets import DatasetDict, Features, load_dataset, load_from_disk
from src.utils.others import (
    dict_to_list,
    get_logger,
    get_tokenizer,
    transform_step_logits,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align tokens and their logits from different models."
    )
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
        "--base_dataset_dir",
        type=str,
        required=True,
        help="The local dir to load data.",
    )
    parser.add_argument(
        "--blending_dataset_dir",
        type=str,
        required=True,
        help="The local dir to load data.",
    )
    parser.add_argument(
        "--dataset_save_dir",
        type=str,
        required=True,
        help="The local dir to save processed data.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache dir.")
    parser.add_argument(
        "--model_max_length", type=int, default=2048, help="The model max length."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=80,
        help="The number of processes to do data loading.",
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="The batch size.")
    parser.add_argument(
        "--blending_model_index",
        type=int,
        default=0,
        help="The index of blending model.",
    )
    parser.add_argument(
        "--vocab_align_type", type=str, default="hard", help="hard or soft."
    )
    parser.add_argument(
        "--vocab_mapping_save_dir", type=str, default=None, help="The vocab mapping."
    )
    parser.add_argument(
        "--metric_level", type=str, default="sequence", help="sequence or token."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Data processing args: {args}")

    base_model_logits_datasets = load_from_disk(args.base_dataset_dir)
    blending_model_logits_datasets = load_from_disk(args.blending_dataset_dir)

    base_tokenizer, _ = get_tokenizer(
        args.base_model_name_or_path, args.cache_dir, args.model_max_length
    )
    blending_tokenizer, _ = get_tokenizer(
        args.blending_model_name_or_path, args.cache_dir, args.model_max_length
    )

    blending_to_base_mapping = (
        json.loads(open(args.vocab_mapping_save_dir, "r").read())
        if (args.vocab_mapping_save_dir is not None and args.vocab_align_type != "hard")
        else None
    )

    def align_blending_model_logits_with_base_model_logits(
        examples_1, indices, dataset_2
    ):
        features_1 = dict_to_list(examples_1)
        features_2 = [dataset_2[idx] for idx in indices]
        aligned_per_step_logits_list, aligned_per_step_indices_list = [], []
        per_step_logits_list, per_step_indices_list = [], []
        metric_ce_aligned = []
        for feature_1, feature_2 in zip(features_1, features_2):
            feature_1["per_step_logits"] = feature_1["per_step_logits"][
                : len(feature_1["input_ids"])
            ]
            feature_1["per_step_indices"] = feature_1["per_step_indices"][
                : len(feature_1["input_ids"])
            ]
            feature_2["per_step_logits"] = feature_2["per_step_logits"][
                : len(feature_2["input_ids"])
            ]
            feature_2["per_step_indices"] = feature_2["per_step_indices"][
                : len(feature_2["input_ids"])
            ]
            if args.metric_level == "token":
                feature_1["per_step_metric_ce"] = feature_1["per_step_metric_ce"][
                    : len(feature_1["input_ids"])
                ]  # The last one is zero
                feature_2["per_step_metric_ce"] = feature_2["per_step_metric_ce"][
                    : len(feature_2["input_ids"])
                ]  # The last one is zero
            (
                aligned_blending_model_per_step_logits,
                aligned_blending_model_per_step_indices,
            ) = transform_step_logits(
                base_model_tokenizer=base_tokenizer,
                blending_model_tokenizer=blending_tokenizer,
                base_model_vocab=base_tokenizer.get_vocab(),
                base_model_input_ids=feature_1["input_ids"],
                blending_model_input_ids=feature_2["input_ids"],
                blending_model_per_step_logits=feature_2["per_step_logits"],
                blending_model_per_step_indices=feature_2["per_step_indices"],
                vocab_align_type=args.vocab_align_type,
                blending_to_base_mapping=blending_to_base_mapping,
            )
            aligned_per_step_logits_list.append(aligned_blending_model_per_step_logits)
            aligned_per_step_indices_list.append(
                aligned_blending_model_per_step_indices
            )
            per_step_logits_list.append(feature_1["per_step_logits"])
            per_step_indices_list.append(feature_1["per_step_indices"])
            if args.metric_level == "sequence":
                metric_ce_aligned.append(feature_2["metric_ce"])
            else:
                metric_ce_aligned.append(feature_2["per_step_metric_ce"])
        examples_1["per_step_logits"] = per_step_logits_list
        examples_1["per_step_indices"] = per_step_indices_list
        examples_1[f"per_step_aligned_logits_{args.blending_model_index}"] = (
            aligned_per_step_logits_list
        )
        examples_1[f"per_step_aligned_indices_{args.blending_model_index}"] = (
            aligned_per_step_indices_list
        )
        if args.metric_level == "sequence":
            examples_1[f"metric_ce_aligned_{args.blending_model_index}"] = (
                metric_ce_aligned
            )
            if "per_step_metric_ce" in examples_1:
                del examples_1["per_step_metric_ce"]
        else:
            examples_1[f"per_step_metric_ce_aligned_{args.blending_model_index}"] = (
                metric_ce_aligned
            )
            if "metric_ce" in examples_1:
                del examples_1["metric_ce"]
        return examples_1

    base_model_blending_model_logits_datasets = DatasetDict({})
    for k in base_model_logits_datasets.keys():
        base_model_blending_model_logits_datasets[k] = base_model_logits_datasets[
            k
        ].map(
            align_blending_model_logits_with_base_model_logits,
            batched=True,
            batch_size=args.batch_size,
            with_indices=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=True,
            fn_kwargs={"dataset_2": blending_model_logits_datasets[k]},
            remove_columns=["text"]
            if "text" in base_model_logits_datasets[k].column_names
            else None,
            keep_in_memory=True,
            desc="Align blending model's logits with base model's logits.",
        )
    base_model_blending_model_logits_datasets.save_to_disk(args.dataset_save_dir)
