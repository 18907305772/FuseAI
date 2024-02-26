"""Align token and vocab."""
import json
import numpy as np
from transformers import LlamaTokenizer, LlamaTokenizerFast, tokenization_utils_base, AutoTokenizer
from datasets import load_from_disk, DatasetDict, Dataset
import editdistance
import argparse
from typing import List, Dict
from tqdm import tqdm
import os


def dict_to_list(examples):
    return [{key: examples[key][i] for key in examples} for i in range(len(examples[next(iter(examples))]))]


def list_to_dict(examples):
    return {key: [d[key] for d in examples] for key in examples[0].keys()}


TOKENIZER_TO_SPECIAL_TOKEN = {LlamaTokenizer: '▁',
                              LlamaTokenizerFast: 'Ġ'}


def sigmoid(x):
    """Compute the sigmoid."""
    return 1. / (1 + np.exp(-x))


def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def dtw(series_1,
        series_2,
        norm_func=np.linalg.norm
        ):
    """
    Use dynamic time wrapping to align to tokenizers, modified from:
    https://github.com/talcs/simpledtw/blob/master/simpledtw.py
    """
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(matrix[i, j + 1], matrix[i + 1, j], matrix[i, j])
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


def token_align_mapping(base_model_tokenizer: tokenization_utils_base.PreTrainedTokenizerBase,
                        blending_model_tokenizer: tokenization_utils_base.PreTrainedTokenizerBase,
                        token_alignment_matrix: np.ndarray = None):
    """
    get one-to-one base to blending mapping based on token alignment matrix
    """
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[base_model_tokenizer.__class__] if "Nous" not in base_model_tokenizer.name_or_path else '▁'
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[blending_model_tokenizer.__class__] if "Nous" not in blending_model_tokenizer.name_or_path else '▁'

    assert blending_model_tokenizer.unk_token != None
    map_token_id = blending_model_tokenizer.unk_token_id  # map base to unk for bad case

    def dist_fn(a, b):
        """calculate editdistance between two tokens, a is from blending model, b is from base model"""
        aa = a.replace(base_model_special_token, '')
        bb = b.replace(blending_model_special_token, '')
        w = 1
        if aa in bb or bb in aa:
            w = 0.1
        dist = editdistance.eval(aa, bb)
        return dist * w

    base_to_blending = [map_token_id for _ in range(len(token_alignment_matrix))]

    for i in range(len(token_alignment_matrix)):
        base_token = base_model_tokenizer.convert_ids_to_tokens(i)
        if base_token == None:  # vocab size may not match?
            continue

        non_zero_ids = np.nonzero(token_alignment_matrix[i])[0]
        if len(non_zero_ids) != 0:
            dists = []
            for j in non_zero_ids:
                blending_token = blending_model_tokenizer.convert_ids_to_tokens(j.item())
                dists.append(dist_fn(base_token, blending_token))
            dist_weights = [sigmoid(token_alignment_matrix[i][j]) for j in non_zero_ids]
            weighted_dists = [dist * d_w for d_w, dist in zip(dist_weights, dists)]
            base_to_blending[i] = int(non_zero_ids[np.argmin(weighted_dists)])
    base_to_blending[base_model_tokenizer.bos_token_id] = blending_model_tokenizer.bos_token_id
    base_to_blending[base_model_tokenizer.eos_token_id] = blending_model_tokenizer.eos_token_id

    total_match = 0
    for i in range(len(token_alignment_matrix)):
        print(f"base: {base_model_tokenizer.convert_ids_to_tokens(i).replace(base_model_special_token, '')}, blending: {blending_model_tokenizer.convert_ids_to_tokens(base_to_blending[i]).replace(blending_model_special_token, '')}")
        if base_to_blending[i] != map_token_id:
            total_match += 1
    print(f"totat match: {total_match}, match_rate: {total_match / len(base_to_blending):.2f}")
    return base_to_blending


def transform_step_token(base_model_tokenizer, base_model_input_ids, blending_model_tokenizer, blending_model_input_ids):
    """
    token alignment: use dtw to perform token alignment for two sequence.
    """
    base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
    blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(blending_model_input_ids)
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[base_model_tokenizer.__class__] if "Nous" not in base_model_tokenizer.name_or_path else '▁'
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[blending_model_tokenizer.__class__] if "Nous" not in blending_model_tokenizer.name_or_path else '▁'

    def dist_fn(a, b):
        """calculate editdistance between two tokens, a is from blending model, b is from base model"""
        aa = a.replace(blending_model_special_token, '')
        bb = b.replace(base_model_special_token, '')
        w = 1
        if aa in bb or bb in aa:
            w = 0.1
        dist = editdistance.eval(aa, bb)
        return dist * w

    _, _, _, base_to_blending, _ = dtw(blending_model_tokens, base_model_tokens, norm_func=dist_fn)
    return base_model_tokens, blending_model_tokens, base_model_special_token, blending_model_special_token, base_to_blending


def transform_step_logits(base_model_tokenizer: tokenization_utils_base.PreTrainedTokenizerBase,
                          blending_model_tokenizer: tokenization_utils_base.PreTrainedTokenizerBase,
                          base_model_vocab: Dict[str, int],
                          base_model_input_ids: List[int],
                          blending_model_input_ids: List[int],
                          blending_model_per_step_logits: List[List[float]],
                          blending_model_per_step_indices: List[List[int]],
                          use_token_alignment_matrix: bool = False,
                          token_alignment_matrix: np.ndarray = None,
                          blending_to_base: List[int] = None,
                          ):
    """
    distribution alignment: align blending model per step logits & indices with base model.
    """
    base_model_tokens, blending_model_tokens, base_model_special_token, blending_model_special_token, base_to_blending\
        = transform_step_token(base_model_tokenizer, base_model_input_ids, blending_model_tokenizer, blending_model_input_ids)
    aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = [], []
    for i, blending_idx in enumerate(base_to_blending):
        aligned_blending_model_per_step_logit = []
        aligned_blending_model_per_step_index = []
        if len(blending_idx) == 1:  # one base token map to one blending token
            j = blending_idx[0]
            base_token = base_model_tokens[i]
            blending_token = blending_model_tokens[j].replace(blending_model_special_token, base_model_special_token)
            if base_token in blending_token:  # one to one, M to one
                # the logits and indices at this step
                for blending_logit, blending_index in \
                        zip(blending_model_per_step_logits[j], blending_model_per_step_indices[j]):
                    # the token corresponds to the logit and indices
                    blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0]\
                        .replace(blending_model_special_token, base_model_special_token)
                    if blending_t in base_model_vocab:
                        aligned_index = base_model_vocab[blending_t]  # the index of the token in base model vocab
                        if aligned_index not in aligned_blending_model_per_step_index:
                            aligned_blending_model_per_step_index.append(aligned_index)
                            aligned_blending_model_per_step_logit.append(blending_logit)
                    else:
                        blending_t = base_model_tokenizer.convert_ids_to_tokens([blending_to_base[blending_index]])[0]
                        aligned_index = base_model_vocab[blending_t]
                        if aligned_index not in aligned_blending_model_per_step_index:
                            aligned_blending_model_per_step_index.append(aligned_index)
                            aligned_blending_model_per_step_logit.append(blending_logit)
            else:  # find error aligned mapping, use the one-hot logits
                aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                aligned_blending_model_per_step_logit.append(1.0)
        else:  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
            if not use_token_alignment_matrix:
                base_token = base_model_tokens[i]
                aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                aligned_blending_model_per_step_logit.append(1.0)
            else:
                base_token: str = base_model_tokens[i]
                blending_tokens: List[str] = [blending_model_tokens[j].replace(blending_model_special_token, base_model_special_token) for j in blending_idx]
                for j, blending_t in zip(blending_idx, blending_tokens):
                    if base_token != base_model_special_token and base_token == blending_t:  # go v.s. [xx, go, xx, xx]
                        blending_idx = [j]
                        break
                if len(blending_idx) != 1:
                    for j, blending_t in zip(blending_idx, blending_tokens):
                        if base_token != base_model_special_token and base_token in blending_t:  # go v.s. [xx, going, xx, xx]
                            blending_idx = [j]
                            break
                if len(blending_idx) == 1:
                    j = blending_idx[0]
                    for blending_logit, blending_index in \
                            zip(blending_model_per_step_logits[j], blending_model_per_step_indices[j]):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0] \
                            .replace(blending_model_special_token, base_model_special_token)
                        if blending_t in base_model_vocab:
                            aligned_index = base_model_vocab[blending_t]  # the index of the token in base model vocab
                            if aligned_index not in aligned_blending_model_per_step_index:
                                aligned_blending_model_per_step_index.append(aligned_index)
                                aligned_blending_model_per_step_logit.append(blending_logit)
                        else:
                            blending_t = base_model_tokenizer.convert_ids_to_tokens([blending_to_base[blending_index]])[0]
                            aligned_index = base_model_vocab[blending_t]
                            if aligned_index not in aligned_blending_model_per_step_index:
                                aligned_blending_model_per_step_index.append(aligned_index)
                                aligned_blending_model_per_step_logit.append(blending_logit)
                else:
                    def find_map_idx(s, list_s):
                        indices = []
                        for i in range(len(list_s)):
                            current_substring = list_s[i]
                            if s.startswith(current_substring):
                                indices.append(i)
                                s = s[len(current_substring):]
                        return indices if not s else []
                    mapped_ids = find_map_idx(base_token, blending_tokens)
                    if len(mapped_ids) > 0:
                        blending_idx = [blending_idx[m_id] for m_id in mapped_ids]
                        blending_token = "".join([blending_model_tokens[j].replace(blending_model_special_token, base_model_special_token) for j in blending_idx])
                        if base_token == blending_token:  # find the aligned mapping, use the corresponding logits
                            # the logits and indices at this step
                            fusion_weight = softmax([token_alignment_matrix[base_model_input_ids[i]][blending_model_input_ids[j]] for j in blending_idx])
                            for idx, j in enumerate(blending_idx):  # multiple distributions
                                for blending_logit, blending_index in \
                                        zip(blending_model_per_step_logits[j], blending_model_per_step_indices[j]):
                                    # the token corresponds to the logit and indices
                                    blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0] \
                                        .replace(blending_model_special_token, base_model_special_token)
                                    if blending_t in base_model_vocab:
                                        aligned_index = base_model_vocab[blending_t]  # the index of the token in base model vocab
                                        if aligned_index not in aligned_blending_model_per_step_index:
                                            aligned_blending_model_per_step_index.append(aligned_index)
                                            aligned_blending_model_per_step_logit.append(blending_logit * fusion_weight[idx])
                                        else:  # multiple blending to one base should use the max logits
                                            cur_aligned_index_idx = aligned_blending_model_per_step_index.index(aligned_index)
                                            aligned_blending_model_per_step_logit[cur_aligned_index_idx] = max(aligned_blending_model_per_step_logit[cur_aligned_index_idx], blending_logit * fusion_weight[idx])
                                    else:
                                        blending_t = base_model_tokenizer.convert_ids_to_tokens([blending_to_base[blending_index]])[0]
                                        aligned_index = base_model_vocab[blending_t]
                                        if aligned_index not in aligned_blending_model_per_step_index:
                                            aligned_blending_model_per_step_index.append(aligned_index)
                                            aligned_blending_model_per_step_logit.append(blending_logit * fusion_weight[idx])
                                        else:  # multiple blending to one base should use the max logits
                                            cur_aligned_index_idx = aligned_blending_model_per_step_index.index(aligned_index)
                                            aligned_blending_model_per_step_logit[cur_aligned_index_idx] = max(aligned_blending_model_per_step_logit[cur_aligned_index_idx], blending_logit * fusion_weight[idx])
                    else:
                        aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                        aligned_blending_model_per_step_logit.append(1.0)
        aligned_blending_model_per_step_indices.append(aligned_blending_model_per_step_index)
        aligned_blending_model_per_step_logits.append(aligned_blending_model_per_step_logit)
    return aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Token alignment.")
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
        help="The local dir to load data with logits."
    )
    parser.add_argument(
        "--blending_dataset_dir",
        type=str,
        required=True,
        help="The local dir to load data with logits."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The cache dir."
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="model max length.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to do data loading."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="The batch size."
    )
    parser.add_argument(
        "--do_token_alignment",
        action="store_true",
        help="do token alignment."
    )
    parser.add_argument(
        "--token_alignment_matrix_file",
        type=str,
        default=None,
        help="file of token alignment matrix."
    )
    parser.add_argument(
        "--blending_to_base_file",
        type=str,
        default=None,
        help="file of blending to base."
    )
    parser.add_argument(
        "--do_distribution_alignment",
        action="store_true",
        help="do distribution alignment."
    )
    parser.add_argument(
        "--blending_model_index",
        type=int,
        default=0,
        help="The index of blending model."
    )
    parser.add_argument(
        "--metric_level",
        type=str,
        default="sequence",
        help="sequence or token level."
    )
    parser.add_argument(
        "--use_token_alignment_matrix",
        action="store_true",
        help="use token alignment matrix for distribution alignment."
    )
    parser.add_argument(
        "--not_align",
        action="store_true",
        help="whether to use alignment."
    )
    parser.add_argument(
        "--dataset_save_dir",
        type=str,
        help="The local dir to save processed data."
    )
    args = parser.parse_args()
    return args


def main():
    """
    stage 1: perform token alignment with dtw on the training corpus, get the token mapping matrix;
        - tbd: use EM to optimize
    stage 2: use the mapping matrix as fusion weight for distribution alignment
    """
    args = parse_args()
    print("Align token and vocab.")
    print(f"Data processing args: {args}")

    base_model_logits_datasets = load_from_disk(args.base_dataset_dir)
    blending_model_logits_datasets = load_from_disk(args.blending_dataset_dir)

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        trust_remote_code=False,
        use_fast=False,
    )
    base_tokenizer.pad_token = base_tokenizer.unk_token

    blending_tokenizer = AutoTokenizer.from_pretrained(
        args.blending_model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        trust_remote_code=False,
        use_fast=False,
    )
    blending_tokenizer.pad_token = blending_tokenizer.unk_token

    base_vocab = base_tokenizer.get_vocab()
    blending_vocab = blending_tokenizer.get_vocab()

    token_mapping_matrices = dict()
    if args.do_token_alignment:
        def token_alignment(examples_1, indices, dataset_2):
            features_1 = dict_to_list(examples_1)
            features_2 = [dataset_2[idx] for idx in indices]
            base_to_blending_list = []
            for feature_1, feature_2 in zip(features_1, features_2):
                _, _, _, _, base_to_blending = transform_step_token(base_model_tokenizer=base_tokenizer,
                                                                    base_model_input_ids=feature_1['input_ids'],
                                                                    blending_model_tokenizer=blending_tokenizer,
                                                                    blending_model_input_ids=feature_2['input_ids']
                                                                    )
                for i in range(len(base_to_blending)):
                    for j in range(len(base_to_blending[i])):
                        base_to_blending[i][j] = feature_2['input_ids'][base_to_blending[i][j]]
                base_to_blending_list.append(base_to_blending)
            examples_1["base_to_blending_mapping"] = base_to_blending_list
            return examples_1

        base_model_logits_with_token_mapping_datasets = DatasetDict({})
        for k in ["train"]:
            base_model_logits_with_token_mapping_datasets[k] = base_model_logits_datasets[k].map(
                token_alignment,
                batched=True,
                batch_size=args.batch_size,
                with_indices=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=True,
                fn_kwargs={"dataset_2": blending_model_logits_datasets[k]},
                keep_in_memory=False,
                desc="Get token mapping.",
            )
            if os.path.exists(args.token_alignment_matrix_file):
                token_mapping_matrix = np.load(args.token_alignment_matrix_file)
                print(f"Using existed token aligment matrix:\n{token_mapping_matrices}")
            else:
                token_mapping_matrix = np.zeros((len(base_vocab), len(blending_vocab)))
            for idx in tqdm(range(len(base_model_logits_with_token_mapping_datasets[k]))):
                base_to_blending_mapping = base_model_logits_with_token_mapping_datasets[k][idx]["base_to_blending_mapping"]
                base_input_ids = base_model_logits_with_token_mapping_datasets[k][idx]["input_ids"]
                for i, base_id in enumerate(base_input_ids):
                    token_mapping_matrix[base_id, base_to_blending_mapping[i]] += 1
            np.save(args.token_alignment_matrix_file, token_mapping_matrix)
            token_mapping_matrices[k] = token_mapping_matrix
            blending_to_base = token_align_mapping(blending_tokenizer, base_tokenizer, np.transpose(token_mapping_matrix))
            with open(args.blending_to_base_file, "w") as f:
                json.dump(blending_to_base, f, ensure_ascii=False)
    else:
        if args.use_token_alignment_matrix:
            for k in ["train"]:
                token_mapping_matrix = np.load(args.token_alignment_matrix_file)
                token_mapping_matrices[k] = token_mapping_matrix
                if os.path.exists(args.blending_to_base_file):
                    with open(args.blending_to_base_file, "r") as f:
                        blending_to_base = json.load(f)
                else:
                    blending_to_base = token_align_mapping(blending_tokenizer, base_tokenizer, np.transpose(token_mapping_matrix))
                    with open(args.blending_to_base_file, "w") as f:
                        json.dump(blending_to_base, f, ensure_ascii=False)
        else:
            token_mapping_matrix = None
            blending_to_base = None

    if args.do_distribution_alignment:
        def distribution_alignment(examples_1, indices, dataset_2):
            features_1 = dict_to_list(examples_1)
            features_2 = [dataset_2[idx] for idx in indices]
            aligned_per_step_logits_list, aligned_per_step_indices_list = [], []
            per_step_logits_list, per_step_indices_list = [], []
            metric_ce_aligned = []
            for feature_1, feature_2 in zip(features_1, features_2):
                feature_1["per_step_logits"] = feature_1["per_step_logits"][:len(feature_1['input_ids'])]
                feature_1["per_step_indices"] = feature_1["per_step_indices"][:len(feature_1['input_ids'])]
                feature_2["per_step_logits"] = feature_2["per_step_logits"][:len(feature_2['input_ids'])]
                feature_2["per_step_indices"] = feature_2["per_step_indices"][:len(feature_2['input_ids'])]
                if args.metric_level == "token":
                    feature_1["per_step_metric_ce"] = feature_1["per_step_metric_ce"][
                                                      :len(feature_1['input_ids'])]  # The last one is zero
                    feature_2["per_step_metric_ce"] = feature_2["per_step_metric_ce"][
                                                      :len(feature_2['input_ids'])]  # The last one is zero
                if args.not_align:
                    aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = \
                        feature_2["per_step_logits"], feature_2['per_step_indices']
                else:
                    aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = transform_step_logits(
                        base_model_tokenizer=base_tokenizer,
                        blending_model_tokenizer=blending_tokenizer,
                        base_model_vocab=base_tokenizer.get_vocab(),
                        base_model_input_ids=feature_1['input_ids'],
                        blending_model_input_ids=feature_2['input_ids'],
                        blending_model_per_step_logits=feature_2['per_step_logits'],
                        blending_model_per_step_indices=feature_2['per_step_indices'],
                        use_token_alignment_matrix=args.use_token_alignment_matrix,
                        token_alignment_matrix=token_mapping_matrix,
                        blending_to_base=blending_to_base,
                    )
                aligned_per_step_logits_list.append(aligned_blending_model_per_step_logits)
                aligned_per_step_indices_list.append(aligned_blending_model_per_step_indices)
                per_step_logits_list.append(feature_1["per_step_logits"])
                per_step_indices_list.append(feature_1["per_step_indices"])
                if args.metric_level == "sequence":
                    metric_ce_aligned.append(feature_2["metric_ce"])
                else:
                    metric_ce_aligned.append(feature_2["per_step_metric_ce"])
            examples_1["per_step_logits"] = per_step_logits_list
            examples_1["per_step_indices"] = per_step_indices_list
            examples_1[f"per_step_aligned_logits_{args.blending_model_index}"] = aligned_per_step_logits_list
            examples_1[f"per_step_aligned_indices_{args.blending_model_index}"] = aligned_per_step_indices_list
            if args.metric_level == "sequence":
                examples_1[f"metric_ce_aligned_{args.blending_model_index}"] = metric_ce_aligned
                if "per_step_metric_ce" in examples_1:
                    del examples_1["per_step_metric_ce"]
            else:
                examples_1[f"per_step_metric_ce_aligned_{args.blending_model_index}"] = metric_ce_aligned
                if "metric_ce" in examples_1:
                    del examples_1["metric_ce"]
            return examples_1

        base_model_blending_model_logits_datasets = DatasetDict({})
        for k in base_model_logits_datasets.keys():
            base_model_blending_model_logits_datasets[k] = base_model_logits_datasets[k].map(
                distribution_alignment,
                batched=True,
                batch_size=args.batch_size,
                with_indices=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=True,
                fn_kwargs={"dataset_2": blending_model_logits_datasets[k]},
                keep_in_memory=False,
                desc="Align blending model's logits with base model's logits.",
            )
        base_model_blending_model_logits_datasets.save_to_disk(args.dataset_save_dir)


if __name__ == "__main__":
    main()