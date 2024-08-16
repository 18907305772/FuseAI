"""Replace the model only, do not align vocab."""

from datasets import Features, load_dataset, load_from_disk, DatasetDict
import argparse
import numpy as np
import datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Replace model.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The local dir to load data."
    )
    parser.add_argument(
        "--replace_dataset_dir",
        type=str,
        required=True,
        help="The local dir to load data."
    )
    parser.add_argument(
        "--dataset_save_dir",
        type=str,
        required=True,
        help="The local dir to save processed data."
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=None, help="The number of processes to do data loading."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="The batch size."
    )
    parser.add_argument(
        "--replace_model", type=str, default="source", help="The model to be replaced. Could be pivot or source."
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("Replace model.")
    print(f"Data processing args: {args}")

    dataset_list = args.dataset_dir.split(",")
    replace_dataset_list = args.replace_dataset_dir.split(",")

    if len(dataset_list) == 1:
        raw_dataset = load_from_disk(dataset_list[0])
    else :
        raw_dataset = datasets.DatasetDict()
        raw_dataset["train"] = datasets.concatenate_datasets([datasets.load_from_disk(_)['train'] for _ in dataset_list])

    if len(replace_dataset_list) == 1 :
        replace_dataset = load_from_disk(replace_dataset_list[0])
    else :
        replace_dataset = datasets.DatasetDict()
        replace_dataset["train"] = datasets.concatenate_datasets([datasets.load_from_disk(_)['train'] for _ in replace_dataset_list])

    def replace_dataset_map(examples_1, indices, dataset_2):
        features_2 = [dataset_2[idx] for idx in indices]
        if args.replace_model == "pivot":
            input_ids, attention_mask, labels = [], [], []
        per_step_logits, per_step_indices, metric_ce = [], [], []
        for feature_2 in features_2:
            if args.replace_model == "pivot":
                input_ids.append(feature_2["input_ids"])
                attention_mask.append(feature_2["attention_mask"])
                labels.append(feature_2["labels"])
            feature_2["per_step_logits"] = feature_2["per_step_logits"][:len(feature_2['input_ids'])]
            feature_2["per_step_indices"] = feature_2["per_step_indices"][:len(feature_2['input_ids'])]
            per_step_logits.append(feature_2["per_step_logits"])
            per_step_indices.append(feature_2["per_step_indices"])
            metric_ce.append(feature_2["metric_ce"])
        if args.replace_model == "pivot":
            examples_1["input_ids"] = input_ids
            examples_1["attention_mask"] = attention_mask
            examples_1["labels"] = labels
            examples_1["per_step_logits"] = per_step_logits
            examples_1["per_step_indices"] = per_step_indices
            examples_1["metric_ce"] = metric_ce
        elif args.replace_model == "source":
            examples_1["fused_per_step_logits"] = per_step_logits
            examples_1["fused_per_step_indices"] = per_step_indices
            examples_1["fused_metric_ce"] = metric_ce
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
            remove_columns=["per_step_metric_ce"] if "per_step_metric_ce" in raw_dataset[k][0].keys() else None,
            desc="Replace model.",
        )

    dataset.save_to_disk(args.dataset_save_dir)