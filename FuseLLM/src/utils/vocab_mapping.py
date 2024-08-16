"""Mapping vocabs from different models."""

import argparse
import json
import multiprocessing

import editdistance
import numpy as np
import tqdm
from datasets import DatasetDict, Features, load_dataset, load_from_disk
from src.utils.others import TOKENIZER_TO_SPECIAL_TOKEN, get_logger, get_tokenizer

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mapping vocabs from different pretrain language models."
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
        "--dataset_dir", type=str, required=True, help="The local dir to load data."
    )
    parser.add_argument(
        "--vocab_mapping_save_dir",
        type=str,
        required=True,
        help="The local dir to save processed data.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache dir.")
    parser.add_argument(
        "--model_max_length", type=int, default=2048, help="The model max length."
    )
    parser.add_argument(
        "--vocab_mapping_type",
        type=str,
        default="default",
        help="The vocab mapping type.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="The window size to calculate co-occurrences.",
    )
    parser.add_argument(
        "--num_process", type=int, default=1, help="The number of process."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Data processing args: {args}")

    base_tokenizer, _ = get_tokenizer(
        args.base_model_name_or_path, args.cache_dir, args.model_max_length
    )
    blending_tokenizer, _ = get_tokenizer(
        args.blending_model_name_or_path, args.cache_dir, args.model_max_length
    )

    base_tokens = list(base_tokenizer.get_vocab().keys())
    blending_tokens = list(blending_tokenizer.get_vocab().keys())
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[base_tokenizer.__class__]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        blending_tokenizer.__class__
    ]

    def find_best_mapping(
        x,
        base_tokens,
        blending_model_special_token,
        base_model_special_token,
        best_one=True,
    ):
        tmp_x = x.replace(blending_model_special_token, base_model_special_token)
        if tmp_x in base_tokens:
            return tmp_x, tmp_x
        else:
            if best_one:
                return tmp_x, min(
                    [(y, editdistance.eval(tmp_x, y)) for y in base_tokens],
                    key=lambda d: d[1],
                )[0]
            else:
                token_and_distance = [
                    (y, editdistance.eval(tmp_x, y)) for y in base_tokens
                ]
                min_distance = min(item[1] for item in token_and_distance)
                shortest_distance_tokens = [
                    item[0] for item in token_and_distance if item[1] == min_distance
                ]
                return tmp_x, shortest_distance_tokens

    if args.vocab_mapping_type == "default":
        blending_to_base_mapping = dict()

        with multiprocessing.Pool(64) as pool:
            mapping_args = [
                (x, base_tokens, blending_model_special_token, base_model_special_token)
                for x in blending_tokens
            ]
            results = list(
                tqdm.tqdm(
                    pool.starmap(find_best_mapping, mapping_args),
                    total=len(blending_tokens),
                )
            )

        for tmp_x, best_mapping in results:
            blending_to_base_mapping[tmp_x] = best_mapping

    elif args.vocab_mapping_type == "co_occurrence":
        """
        1. Calculate basic one-to-many mapping based on editdistance
        2. Ranking based on co-occurrence to get one-to-one mapping
        """
        # 1. Calculate basic one-to-many mapping based on editdistance
        blending_to_base_mapping = dict()

        with multiprocessing.Pool(64) as pool:
            mapping_args = [
                (x, base_tokens, blending_model_special_token, base_model_special_token)
                for x in blending_tokens
            ]
            results = list(
                tqdm.tqdm(
                    pool.starmap(find_best_mapping, mapping_args),
                    total=len(blending_tokens),
                )
            )

        for tmp_x, best_mapping in results:
            blending_to_base_mapping[tmp_x] = best_mapping

        most_sim_blending_to_base_mapping = dict()
        with multiprocessing.Pool(64) as pool:
            mapping_args = [
                (
                    x,
                    base_tokens,
                    blending_model_special_token,
                    base_model_special_token,
                    False,
                )
                for x in blending_tokens
            ]
            results = list(
                tqdm.tqdm(
                    pool.starmap(find_best_mapping, mapping_args),
                    total=len(blending_tokens),
                )
            )

        for tmp_x, basic_mapping in results:
            most_sim_blending_to_base_mapping[tmp_x] = basic_mapping

        dataset = load_from_disk(args.dataset_dir)[
            "validation"
        ]  # The training set is too large!

        text_corpus = [dataset[i]["text"] for i in range(len(dataset))]

        def calculate_cooccurrence(chunk, tokenizer, window_size):
            tknz_text = tokenizer(
                list(chunk),
                add_special_tokens=False,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            input_ids = tknz_text["input_ids"]
            vocab_size = len(tokenizer.get_vocab())
            co_occurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

            for row in tqdm.tqdm(input_ids):
                for i, token_id in enumerate(row):
                    for j in range(
                        max(0, i - window_size), min(i + window_size, len(row))
                    ):
                        if i != j:
                            co_occurrence_matrix[token_id, row[j]] += 1
                            co_occurrence_matrix[row[j], token_id] += 1

            return co_occurrence_matrix

        def tokenize_and_calculate_cooccurrence(
            text_corpus, tokenizer, window_size, n_processes
        ):
            with multiprocessing.Pool(n_processes) as pool:
                chunks = np.array_split(text_corpus, n_processes)
                co_occurrence_matrices = pool.starmap(
                    calculate_cooccurrence,
                    [(chunk, tokenizer, window_size) for chunk in chunks],
                )

            vocab_size = len(tokenizer.get_vocab())
            co_occurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
            for matrix in co_occurrence_matrices:
                co_occurrence_matrix += matrix

            return co_occurrence_matrix

        base_model_co_occurrence_matrix = tokenize_and_calculate_cooccurrence(
            text_corpus, base_tokenizer, args.window_size, args.num_process
        )
        blending_model_co_occurrence_matrix = tokenize_and_calculate_cooccurrence(
            text_corpus, blending_tokenizer, args.window_size, args.num_process
        )
        base_vocab = base_tokenizer.get_vocab()
        base_id_to_vocab = {v: k for k, v in base_vocab.items()}
        blending_vocab = blending_tokenizer.get_vocab()
        blending_id_to_vocab = {v: k for k, v in blending_vocab.items()}
        tmp_blending_tokens = [
            x.replace(blending_model_special_token, base_model_special_token)
            for x in blending_tokens
        ]
        base_to_blending_mapping = {v: k for k, v in blending_to_base_mapping.items()}
        blending_tokens_mapping_to_base_tokens = [
            blending_to_base_mapping[x] for x in tmp_blending_tokens
        ]
        common_tokens = list(
            set(blending_tokens_mapping_to_base_tokens) & set(base_tokens)
        )
        blending_index = np.array(
            [
                blending_vocab[
                    base_to_blending_mapping[c].replace(
                        base_model_special_token, blending_model_special_token
                    )
                ]
                for c in common_tokens
            ]
        )
        base_index = np.array([base_vocab[c] for c in common_tokens])
        clip_blending_model_co_occurrence_matrix = blending_model_co_occurrence_matrix[
            :, blending_index
        ]
        clip_base_model_co_occurrence_matrix = base_model_co_occurrence_matrix[
            :, base_index
        ]

        def refined_mapping(key, value):
            if key in value:
                best_base_token = key  # best mapping
            elif len(value) == 1:
                best_base_token = value[0]  # only one mapping
            else:
                blending_id = blending_vocab[
                    key.replace(base_model_special_token, blending_model_special_token)
                ]
                blending_vector = clip_blending_model_co_occurrence_matrix[blending_id]
                base_ids = np.array([base_vocab[base_token] for base_token in value])
                base_vectors = clip_base_model_co_occurrence_matrix[base_ids]
                dot_product = np.dot(blending_vector, base_vectors.T)
                blending_norm = np.linalg.norm(blending_vector)
                base_norms = np.linalg.norm(base_vectors, axis=1)
                similarities = dot_product / (blending_norm * base_norms + 1e-6)
                assert len(similarities) == len(value)
                best_base_token = value[int(np.argmax(similarities))]
            return key, best_base_token

        updated_cnt = 0
        updated_dict = dict()
        for key, value in tqdm.tqdm(most_sim_blending_to_base_mapping.items()):
            if blending_to_base_mapping[key] == key:
                continue
            _, new_value = refined_mapping(key, value)
            if new_value != blending_to_base_mapping[key]:
                updated_cnt += 1
                updated_dict[key] = {
                    "default": blending_to_base_mapping[key],
                    "co-occurrence": new_value,
                }
                blending_to_base_mapping[key] = new_value
        logger.info(f"Co-occurrence updated {updated_cnt} tokens.")
        with open(
            args.vocab_mapping_save_dir.replace(".json", "_updated_log.json"), "w"
        ) as fout:
            json.dump(updated_dict, fout)
    else:
        raise NotImplementedError
    cnt = 0
    for k, v in blending_to_base_mapping.items():
        if k == v:
            cnt += 1
    logger.info(
        f"Total tokens in blending vocab: {len(blending_tokenizer.get_vocab())},"
        f"Total tokens in blending to base mapping: {len(blending_to_base_mapping)},"
        f"Total best matched tokens: {cnt}."
    )
    with open(args.vocab_mapping_save_dir, "w") as fout:
        json.dump(blending_to_base_mapping, fout)
