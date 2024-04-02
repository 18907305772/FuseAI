"""4. Packing all features."""

import argparse
import math
from itertools import chain

from datasets import DatasetDict, Features, load_dataset, load_from_disk
from src.utils.others import (
    get_logger,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Packing all features in dataset.")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="The local dir to load data."
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
        default=None,
        help="The number of processes to do data loading.",
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="The batch size.")
    parser.add_argument(
        "--metric_level", type=str, default="sequence", help="sequence or token."
    )
    parser.add_argument(
        "--fiter_nan", action="store_true", help="Filter nan instances."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Data processing args: {args}")
    raw_dataset = load_from_disk(args.dataset_dir)

    if args.fiter_nan:

        def not_nan(example):
            return (
                not math.isnan(example["metric_ce"])
                and not math.isnan(example["metric_ce_aligned_0"])
                and not math.isnan(example["metric_ce_aligned_1"])
            )

        for s, d in raw_dataset.items():
            original_size = len(raw_dataset[s])
            filtered_d = d.filter(not_nan)
            raw_dataset[s] = filtered_d
            filtered_size = len(raw_dataset[s])
            deleted_examples = original_size - filtered_size
            logger.info(f"Delete {deleted_examples} instances.")

    def packing_dataset(examples):
        if args.metric_level == "sequence":
            for k, v in examples.items():
                if "metric_ce" not in k:
                    continue
                examples[k] = [
                    [x] * len(examples["input_ids"][i])
                    for i, x in enumerate(examples[k])
                ]
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        block_size = args.model_max_length
        # total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if args.metric_level == "sequence":
            for k, v in result.items():
                if "metric_ce" not in k:
                    continue
                result[k] = [sum(x) / len(x) for i, x in enumerate(result[k])]
        else:
            remove_keys = []
            for k, v in result.items():
                if "per_step_metric_ce" not in k:
                    continue
                result[k.replace("per_step_", "")] = [
                    sum(x) / len(x) for i, x in enumerate(result[k])
                ]
                remove_keys.append(k)
            for key in remove_keys:
                del result[key]
        return result

    dataset = raw_dataset.map(
        packing_dataset,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        keep_in_memory=True,
        desc="Packing all features in dataset.",
    )

    dataset.save_to_disk(args.dataset_save_dir)
