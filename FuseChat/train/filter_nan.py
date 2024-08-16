"""Filter instances with NaN loss."""
import argparse
from datasets import load_dataset, load_from_disk, DatasetDict
import numpy as np


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
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        required=True,
        help="preprocessing_num_workers.",
    )

    args = parser.parse_args()
    return args



def check_nan(example):
    return not (np.isnan(example["metric_ce"]) or np.isnan(example["fused_metric_ce"]))

def check_nan_base(example):
    return not(np.isnan(example["metric_ce"]))

if __name__ == "__main__":
    args = parse_args()
    print("Filter NaN.")
    print(f"Data processing args: {args}")

    input_data = load_from_disk(args.input_data_dir)
    new_data = DatasetDict({})

    for k, v in input_data.items():
        new_data[k] = input_data[k].filter(check_nan, num_proc=args.preprocessing_num_workers)
        print(f"filtered_num: {len(input_data[k]) - len(new_data[k])}")
    new_data.save_to_disk(args.output_data_dir)
