"""Replace the model only and do not align vocab."""

import argparse

import numpy as np
from datasets import DatasetDict, Features, load_dataset, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Replace model.")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="The local dir to load data."
    )
    parser.add_argument(
        "--replace_dataset_dir",
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
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to do data loading.",
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="The batch size.")
    parser.add_argument(
        "--replace_model",
        type=str,
        default="base",
        help="The model to be replaced. Could be base or model_0 or model_1.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("Replace model.")
    print(f"Data processing args: {args}")

    raw_dataset = load_from_disk(args.dataset_dir)
    replace_dataset = load_from_disk(args.replace_dataset_dir)

    def replace_dataset_map(examples_1, indices, dataset_2):
        features_2 = [dataset_2[idx] for idx in indices]
        if args.replace_model == "base":
            input_ids, attention_mask, labels = [], [], []
        per_step_logits, per_step_indices, metric_ce = [], [], []
        for feature_2 in features_2:
            if args.replace_model == "base":
                input_ids.append(feature_2["input_ids"])
                attention_mask.append(feature_2["attention_mask"])
                labels.append(feature_2["labels"])
            feature_2["per_step_logits"] = feature_2["per_step_logits"][
                : len(feature_2["input_ids"])
            ]
            feature_2["per_step_indices"] = feature_2["per_step_indices"][
                : len(feature_2["input_ids"])
            ]
            per_step_logits.append(feature_2["per_step_logits"])
            per_step_indices.append(feature_2["per_step_indices"])
            metric_ce.append(feature_2["metric_ce"])
        if args.replace_model == "base":
            examples_1["input_ids"] = input_ids
            examples_1["attention_mask"] = attention_mask
            examples_1["labels"] = labels
            examples_1["per_step_logits"] = per_step_logits
            examples_1["per_step_indices"] = per_step_indices
            examples_1["metric_ce"] = metric_ce
        elif args.replace_model == "model_0":
            examples_1["per_step_aligned_logits_0"] = per_step_logits
            examples_1["per_step_aligned_indices_0"] = per_step_indices
            examples_1["metric_ce_aligned_0"] = metric_ce
        elif args.replace_model == "model_1":
            examples_1["per_step_aligned_logits_1"] = per_step_logits
            examples_1["per_step_aligned_indices_1"] = per_step_indices
            examples_1["metric_ce_aligned_1"] = metric_ce
        else:
            raise NotImplementedError
        return examples_1

    dataset = DatasetDict({})
    for k in raw_dataset.keys():
        dataset[k] = raw_dataset[k].map(
            replace_dataset_map,
            batched=True,
            batch_size=args.batch_size,
            with_indices=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=True,
            fn_kwargs={"dataset_2": replace_dataset[k]},
            keep_in_memory=True,
            remove_columns=["per_step_metric_ce"]
            if "per_step_metric_ce" in raw_dataset[k][0].keys()
            else None,
            desc="Replace model.",
        )

    dataset.save_to_disk(args.dataset_save_dir)
