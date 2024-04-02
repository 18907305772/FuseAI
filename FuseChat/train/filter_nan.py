"""Filter instances with NaN loss."""

import argparse

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Filter NaN.")
    parser.add_argument(
        "--input_data_dir",
        type=str,
        required=True,
        help="Path to input dataset.",
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        required=True,
        help="Path to output dataset.",
    )
    args = parser.parse_args()
    return args


def check_nan(example):
    return not (
        np.isnan(example["metric_ce"])
        or np.isnan(example["metric_ce_aligned_0"])
        or np.isnan(example["metric_ce_aligned_1"])
    )


if __name__ == "__main__":
    args = parse_args()
    print("Filter NaN.")
    print(f"Data processing args: {args}")

    data = load_from_disk(args.input_data_dir)
    new_data = DatasetDict({})
    for k, v in data.items():
        new_data[k] = data[k].filter(check_nan, num_proc=64)
        print(f"filtered_num: {len(data[k]) - len(new_data[k])}")
    new_data.save_to_disk(args.output_data_dir)
