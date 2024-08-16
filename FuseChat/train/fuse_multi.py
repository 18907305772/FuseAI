'''Fuse distributions from multiple LLMs.'''
from datasets import Features, load_dataset, load_from_disk, DatasetDict
import argparse
import numpy as np
import datasets


def dict_to_list(examples):
    return [{key: examples[key][i] for key in examples} for i in range(len(examples[next(iter(examples))]))]

def parse_args():
    parser = argparse.ArgumentParser(description="Replace model.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The local dir to load data."
    )
    parser.add_argument(
        "--fused_dataset_dir",
        type=str,
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
        "--mode",
        type=str,
        default="fuse-all-mince",
        choices=["fuse-all-mince", "add-new"],
        help=(
            "Fuse mode. "
            "`fuse-all : fuse all selected models in fused_dataset_dir "
            "`add-new` add new model,replace metric_ce < fused_metric_ce "
        ),
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("Fuse model.")
    print(f"Data processing args: {args}")

    if args.mode=="fuse-all-mince":
        dataset_list = args.dataset_dir.split(",")
        fused_dataset_list = args.fused_dataset_dir.split(",")

        if len(dataset_list) == 1:
            raw_dataset = load_from_disk(dataset_list[0])
        else:
            raw_dataset = datasets.DatasetDict()
            raw_dataset["train"] = datasets.concatenate_datasets(
                [datasets.load_from_disk(_)['train'] for _ in dataset_list])
        fused_dataset_all = []
        for data_name in fused_dataset_list:
            fused_dataset = load_from_disk(data_name)
            fused_dataset_all.append(fused_dataset["train"])

    # fused_per_step_indices fused_per_step_logits fused_metric_ce
        def fused_dataset_map(examples_1, indices, fuse_dataset):
            # fuse_dataset: all dataset to fuse,choose rule:mince
            features_2 = []
            for idx in indices:
                metric_ce_list = []
                data_list=[]
                for dataset in fuse_dataset:
                    cur_data=dataset.select([idx])
                    data_list.append(cur_data[0])
                    cur_ce = cur_data[0]["fused_metric_ce"] if not np.isnan(cur_data[0]["fused_metric_ce"]) else 1e5
                    metric_ce_list.append(cur_ce)
                min_idx = metric_ce_list.index(min(metric_ce_list)) # min index in metric list
                features_2.append(data_list[min_idx])

                del metric_ce_list,data_list

            per_step_logits, per_step_indices, metric_ce = [], [], []
            for feature_2 in features_2:
                per_step_logits.append(feature_2["fused_per_step_logits"])
                # per_step_logits.append(list(map(np.float64, feature_2["fused_per_step_logits"])))
                per_step_indices.append(feature_2["fused_per_step_indices"])
                metric_ce.append(feature_2["fused_metric_ce"])

            examples_1["fused_per_step_indices"] = per_step_indices
            examples_1["fused_per_step_logits"] = per_step_logits
            examples_1["fused_metric_ce"] = metric_ce

            del per_step_logits, per_step_indices, metric_ce,features_2

            return examples_1


        dataset = DatasetDict({})
        for k in raw_dataset.keys():
            dataset[k] = raw_dataset[k].map(
                fused_dataset_map,
                batched=True,
                batch_size=args.batch_size,
                with_indices=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=True,
                fn_kwargs={"fuse_dataset": fused_dataset_all},
                keep_in_memory=True,
                remove_columns=["per_step_metric_ce"] if "per_step_metric_ce" in raw_dataset[k][0].keys() else None,
                desc="Fuse all source llms.",
            )

        dataset.save_to_disk(args.dataset_save_dir)

    elif args.mode=="add-new":
        raw_dataset = load_from_disk(args.dataset_dir)
        fused_dataset = load_from_disk(args.fused_dataset_dir)



        def fused_dataset_map_add_new(examples_1, indices, fuse_dataset):
            # fuse_dataset: all dataset to fuse,choose rule:mince
            features_1 = dict_to_list(examples_1)
            features_2 = fuse_dataset.select(indices)
            per_step_logits, per_step_indices, metric_ce = [], [], []
            for example_1, feature_2 in zip(features_1, features_2):
                cur_ce = example_1["fused_metric_ce"] if not np.isnan(example_1["fused_metric_ce"]) else 1e5
                new_ce = feature_2["fused_metric_ce"] if not np.isnan(feature_2["fused_metric_ce"]) else 1e5
                if new_ce < cur_ce:
                    per_step_logits.append(feature_2["fused_per_step_logits"])
                    per_step_indices.append(feature_2["fused_per_step_indices"])
                    metric_ce.append(feature_2["fused_metric_ce"])
                else:
                    per_step_logits.append(example_1["fused_per_step_logits"])
                    per_step_indices.append(example_1["fused_per_step_indices"])
                    metric_ce.append(example_1["fused_metric_ce"])

            examples_1["fused_per_step_indices"] = per_step_indices
            examples_1["fused_per_step_logits"] = per_step_logits
            examples_1["fused_metric_ce"] = metric_ce

            del features_1
            del features_2

            return examples_1

        dataset = DatasetDict({})
        dataset["train"] = raw_dataset["train"].map(
            fused_dataset_map_add_new,
            batched=True,
            batch_size=args.batch_size,
            with_indices=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=True,
            fn_kwargs={"fuse_dataset": fused_dataset["train"]},
            keep_in_memory=True,
            desc="Replace model.",
        )

        dataset.save_to_disk(args.dataset_save_dir)