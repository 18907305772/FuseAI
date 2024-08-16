"""Align token and vocab."""
import json
from multiprocessing import Pool, cpu_count
import numpy as np
import transformers
import random
from scipy import sparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from transformers import LlamaTokenizer, LlamaTokenizerFast, tokenization_utils_base, AutoTokenizer,Qwen2Tokenizer,GemmaTokenizer
from datasets import load_from_disk, DatasetDict, Dataset
import datasets
import editdistance
import argparse
from typing import List, Dict
from tqdm import tqdm
import os

SPECIAL_TOKEN_LLAMA3_TO_MISTRAL = {
    "âĢĻ":"’",
    'âĢĶ':'—',
    'âĢĿ':'”',
    'č':'\r',
    'ĉ':'<0x09>',
    "Ċ":"<0x0A>"
}

def dict_to_list(examples):
    return [{key: examples[key][i] for key in examples} for i in range(len(examples[next(iter(examples))]))]


def list_to_dict(examples):
    return {key: [d[key] for d in examples] for key in examples[0].keys()}


TOKENIZER_TO_SPECIAL_TOKEN = {transformers.LlamaTokenizer: '▁',
                              transformers.PreTrainedTokenizerFast: 'Ġ',
                              transformers.tokenization_utils_fast.PreTrainedTokenizerFast: 'Ġ',
                              transformers.GPTNeoXTokenizerFast: 'Ġ',
                              transformers.CodeGenTokenizer: 'Ġ',
                              transformers.models.codegen.tokenization_codegen.CodeGenTokenizer: 'Ġ',
                              transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer: 'Ġ',
                              transformers.models.gemma.tokenization_gemma.GemmaTokenizer: 'Ġ',
                              transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer: 'Ġ',
    }

def sigmoid(x):
    """Compute the sigmoid."""
    return 1. / (1 + np.exp(-x))


def softmax(x,T):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x/T)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def dtw(series_1,
        series_2,
        norm_func=np.linalg.norm
        ):
    """
    Use dynamic time wrapping to align to tokenizers
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
        move = np.argmin([option_diag, option_left, option_up])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()
    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


def find_best_mapping(args):
    base_token, base_model_tokenizer, blending_model_tokenizer, base_model_special_token, blending_model_special_token, blending_tokens = args
    base_id = base_model_tokenizer.get_vocab()[base_token]
    base_token = base_token.replace(base_model_special_token, blending_model_special_token)
    if base_token in blending_tokens:
        return base_id, blending_model_tokenizer.get_vocab()[base_token]
    else:
        blending_token_ids = np.array(list(blending_model_tokenizer.get_vocab().values()))
        blending_tokens_np = np.array(list(blending_model_tokenizer.get_vocab().keys()))
        edit_distances = editdistance.eval(base_token, blending_tokens_np)
        min_idx = np.argmin(edit_distances)
        return base_id, blending_token_ids[min_idx]

def replace_special_tokens(token_list):
    for special_token in SPECIAL_TOKEN_LLAMA3_TO_MISTRAL.keys():
        token_list = [token.replace(special_token,SPECIAL_TOKEN_LLAMA3_TO_MISTRAL[special_token]) if special_token in token else token for token in token_list]
    return token_list

def token_align_mapping(base_model_tokenizer: tokenization_utils_base.PreTrainedTokenizerBase,
                        blending_model_tokenizer: tokenization_utils_base.PreTrainedTokenizerBase,
                        algin_type: str = "default",
                        token_alignment_matrix: np.ndarray = None):
    """
    get one-to-one base to blending mapping based on token alignment matrix
    """
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[base_model_tokenizer.__class__]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[blending_model_tokenizer.__class__]

    map_token_id = blending_model_tokenizer.unk_token_id if blending_model_tokenizer.unk_token is not None else blending_model_tokenizer.eos_token_id

    if algin_type == "soft":
        base_tokens = list(base_model_tokenizer.get_vocab().keys())
        blending_tokens = list(blending_model_tokenizer.get_vocab().keys())
        base_to_blending = [0] * len(base_tokens)
        print(f"cpu count: {cpu_count()}, base vocab: {len(base_tokens)} blending vocab: {len(blending_tokens)}")
        mapping_args = [(x, base_model_tokenizer, blending_model_tokenizer, base_model_special_token, blending_model_special_token, blending_tokens) for x in base_tokens]

        with ProcessPoolExecutor(max_workers=32) as executor:
            results = list(tqdm(executor.map(find_best_mapping, mapping_args), total=len(base_tokens)))
            
        for base_id, blending_id in results:
            base_to_blending[base_id] = int(blending_id)
    else:
        def dist_fn(a, b):
            """calculate editdistance between two tokens, a is from blending model, b is from base model"""
            aa = a.replace(base_model_special_token, '')
            bb = b.replace(blending_model_special_token, '')
            for special_token in SPECIAL_TOKEN_LLAMA3_TO_MISTRAL:
                aa = aa.replace(special_token, SPECIAL_TOKEN_LLAMA3_TO_MISTRAL[special_token]) if special_token in aa else aa
                bb = bb.replace(special_token,SPECIAL_TOKEN_LLAMA3_TO_MISTRAL[special_token]) if special_token in bb else bb

            if len(aa) == 0 or len(bb) == 0:
                aa = a.replace(base_model_special_token, blending_model_special_token)
                bb = b

            w = 1
            if aa in bb or bb in aa:
                w = 0.1
            dist = editdistance.eval(aa, bb)
            return dist * w

        n_row = token_alignment_matrix.shape[0]
        base_to_blending = [map_token_id for _ in range(n_row)]


        for i in range(n_row):
            base_token = base_model_tokenizer.convert_ids_to_tokens(i)
            if base_token == None:
                continue

            assert isinstance(token_alignment_matrix,np.ndarray | sparse.coo_matrix), "token_alignment_matrix should be np.ndarray or sparse.coo_matrix"
            if isinstance(token_alignment_matrix, np.ndarray):
                non_zero_ids = np.nonzero(token_alignment_matrix[i])[0]  # dense
                row_i = token_alignment_matrix[i]
            elif isinstance(token_alignment_matrix, sparse.coo_matrix):
                non_zero_ids = np.nonzero(token_alignment_matrix.getrow(i).toarray()[0])[0]  # sparse
                row_i = token_alignment_matrix.getrow(i).toarray()[0]

            if len(non_zero_ids) != 0:
                dists = []
                for j in non_zero_ids:
                    blending_token = blending_model_tokenizer.convert_ids_to_tokens(j.item())
                    dists.append(dist_fn(base_token, blending_token))
                dist_weights = [sigmoid(row_i[j]) for j in non_zero_ids]
                weighted_dists = [dist * d_w for d_w, dist in zip(dist_weights, dists)]
                base_to_blending[i] = int(non_zero_ids[np.argmin(weighted_dists)])
        if base_model_tokenizer.bos_token_id is not None and blending_model_tokenizer.bos_token_id is not None:
            base_to_blending[base_model_tokenizer.bos_token_id] = blending_model_tokenizer.bos_token_id
        if base_model_tokenizer.eos_token_id is not None and blending_model_tokenizer.eos_token_id is not None:
            base_to_blending[base_model_tokenizer.eos_token_id] = blending_model_tokenizer.eos_token_id

    total_match = 0
    for i in range(n_row):
        print(
            f"base: {base_model_tokenizer.convert_ids_to_tokens(i).replace(base_model_special_token, '')}, blending: {blending_model_tokenizer.convert_ids_to_tokens(base_to_blending[i]).replace(blending_model_special_token, '')}")
        if base_to_blending[i] != map_token_id:
            total_match += 1
    print(f"totat match: {total_match}, match_rate: {total_match / len(base_to_blending):.2f}")
    return base_to_blending


def transform_step_token(base_model_tokenizer, base_model_input_ids, blending_model_tokenizer, blending_model_input_ids):
    """
    token alignment: use dtw to perform token alignment for two sequence.
    """

    base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
    base_model_tokens = replace_special_tokens(base_model_tokens)
    blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(blending_model_input_ids)
    blending_model_tokens = replace_special_tokens(blending_model_tokens)

    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[base_model_tokenizer.__class__]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[blending_model_tokenizer.__class__]

    def dist_fn(a, b):
        """calculate editdistance between two tokens, a is from blending model, b is from base model"""
        aa = a.replace(blending_model_special_token, '')
        bb = b.replace(base_model_special_token, '')

        if len(aa) == 0 or len(bb) == 0:
                aa = a.replace(blending_model_special_token, base_model_special_token)
                bb = b
        w = 1
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
                          token_alignment_matrix: sparse.coo_matrix = None,
                          blending_to_base: List[int] = None,
                          align_type:str = "default",
                          temperature:float = 1.0,
                          ):
    """
    distribution alignment: align blending model per step logits & indices with base model.
    """
    base_model_tokens, blending_model_tokens, base_model_special_token, blending_model_special_token, base_to_blending\
        = transform_step_token(base_model_tokenizer, base_model_input_ids, blending_model_tokenizer, blending_model_input_ids)
    base_model_labels = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)[1:]
    base_model_labels = base_model_labels + [base_model_labels[-1]]
    aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = [], []
    for i, blending_idx in enumerate(base_to_blending):
        aligned_blending_model_per_step_logit = []
        aligned_blending_model_per_step_index = []
        if len(blending_idx) == 1:  # one base token map to one blending token
            j = blending_idx[0]
            base_token = base_model_tokens[i]
            blending_token = blending_model_tokens[j].replace(blending_model_special_token, base_model_special_token)
            if align_type == "hard":
                if base_token == blending_token:  # find the aligned mapping, use the corresponding logits
                    # the logits and indices at this step
                    for blending_logit, blending_index in zip(blending_model_per_step_logits[j],
                                                              blending_model_per_step_indices[j]):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0].replace(
                            blending_model_special_token, base_model_special_token)
                        if blending_t in base_model_vocab:
                            aligned_index = base_model_vocab[blending_t]  # the index of the token in base model vocab
                            if aligned_index not in aligned_blending_model_per_step_index:
                                aligned_blending_model_per_step_index.append(aligned_index)
                                aligned_blending_model_per_step_logit.append(blending_logit)
                else:  # find error aligned mapping, use the one-hot logits
                    aligned_blending_model_per_step_index.append(base_model_vocab[base_model_labels[i]])
                    aligned_blending_model_per_step_logit.append(1.0)
            elif align_type == "soft":
                if (base_token == blending_token) or (base_model_vocab[base_token] == blending_to_base[j]):  # find the aligned mapping, use the corresponding logits
                    # the logits and indices at this step
                    for blending_logit, blending_index in zip(blending_model_per_step_logits[j],blending_model_per_step_indices[j]):
                        # the token corresponds to the logit and indices
                        blending_t = base_model_tokenizer.convert_ids_to_tokens([blending_to_base[blending_index]])[0]
                        aligned_index = base_model_vocab[blending_t]  # the index of the token in base model vocab
                        if aligned_index not in aligned_blending_model_per_step_index:
                            aligned_blending_model_per_step_index.append(aligned_index)
                            aligned_blending_model_per_step_logit.append(blending_logit)
                else:  # find error aligned mapping, use the one-hot logits
                    aligned_blending_model_per_step_index.append(base_model_vocab[base_model_labels[i]])
                    aligned_blending_model_per_step_logit.append(1.0)
            else:
                if blending_token.endswith(base_token) or base_token.endswith(blending_token):  # 1-1 or n-1: learn the last token
                    # the logits and indices at this step
                    for blending_logit, blending_index in zip(blending_model_per_step_logits[j], blending_model_per_step_indices[j]):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0].replace(blending_model_special_token, base_model_special_token)
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
                    aligned_blending_model_per_step_index.append(base_model_vocab[base_model_labels[i]])
                    aligned_blending_model_per_step_logit.append(1.0)
        elif align_type == "default":  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
            if not use_token_alignment_matrix:
                aligned_blending_model_per_step_index.append(base_model_vocab[base_model_labels[i]])
                aligned_blending_model_per_step_logit.append(1.0)
            else:
                base_token: str = base_model_tokens[i].replace(base_model_special_token, "")
                blending_tokens: List[str] = [blending_model_tokens[j].replace(blending_model_special_token, "") for j in blending_idx]
                for j, blending_t in zip(blending_idx, blending_tokens):
                    if base_token != base_model_special_token and (base_token.endswith(blending_t) or blending_t.endswith(base_token)):  # xxgo v.s. [xx, yygo, xx, xx]
                        blending_idx = [j]
                        break
                if len(blending_idx) == 1:
                    j = blending_idx[0]
                    for blending_logit, blending_index in \
                            zip(blending_model_per_step_logits[j], blending_model_per_step_indices[j]):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0].replace(blending_model_special_token, base_model_special_token)
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
                    blending_all = "".join(blending_tokens)
                    if len(mapped_ids) > 0 or blending_all.endswith(base_token)  or base_token.endswith(blending_all):
                        blending_idx = [blending_idx[m_id] for m_id in mapped_ids]
                        blending_token = "".join([blending_model_tokens[j].replace(blending_model_special_token, "")for j in blending_idx])
                        if base_token == blending_token:  # find the aligned mapping, use the corresponding logits
                            blending_id = blending_idx[-1]
                            for blending_logit, blending_index in zip(blending_model_per_step_logits[blending_id], blending_model_per_step_indices[blending_id]):
                                # the token corresponds to the logit and indices
                                blending_t = blending_model_tokenizer.convert_ids_to_tokens([blending_index])[0].replace(blending_model_special_token, base_model_special_token)
                                if blending_t in base_model_vocab:
                                    aligned_index = base_model_vocab[
                                        blending_t]  # the index of the token in base model vocab
                                    if aligned_index not in aligned_blending_model_per_step_index:
                                        aligned_blending_model_per_step_index.append(aligned_index)
                                        aligned_blending_model_per_step_logit.append(blending_logit)
                                else:
                                    blending_t = base_model_tokenizer.convert_ids_to_tokens([blending_to_base[blending_index]])[0]
                                    aligned_index = base_model_vocab[blending_t]
                                    if aligned_index not in aligned_blending_model_per_step_index:
                                        aligned_blending_model_per_step_index.append(aligned_index)
                                        aligned_blending_model_per_step_logit.append(blending_logit)
                    else :
                        aligned_blending_model_per_step_index.append(base_model_vocab[base_model_labels[i]])
                        aligned_blending_model_per_step_logit.append(1.0)
        else:
            aligned_blending_model_per_step_index.append(base_model_vocab[base_model_labels[i]])
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
        help="Path to pivot llm.",
    )
    parser.add_argument(
        "--base_model_use_fast",type=str, default=False
    )
    parser.add_argument(
        "--base_model_trust_remote_code", type=str, default=False
    )
    parser.add_argument(
        "--blending_model_use_fast",type=str, default=False
    )
    parser.add_argument(
        "--blending_model_trust_remote_code", type=str, default=False
    )
    parser.add_argument(
        "--blending_model_name_or_path",
        type=str,
        required=True,
        help="Path to source llm.",
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
        help="Whether to do token alignment."
    )
    parser.add_argument(
        "--align_type",
        type=str,
        default="default",
        help="Which align method to use."
    )
    parser.add_argument(
        "--token_alignment_matrix_file",
        type=str,
        default=None,
        help="File path of token alignment sparse matrix."
    )
    parser.add_argument(
        "--blending_to_base_file",
        type=str,
        default=None,
        help="File path of blending to base json file."
    )
    parser.add_argument(
        "--do_distribution_alignment",
        action="store_true",
        help="Whether to do distribution alignment."
    )
    parser.add_argument(
        "--metric_level",
        type=str,
        default="sequence",
        help="Sequence or token level."
    )
    parser.add_argument(
        "--use_token_alignment_matrix",
        action="store_true",
        help="Whether to use token alignment matrix for distribution alignment."
    )
    parser.add_argument(
        "--not_align",
        action="store_true",
        help="Whether to apply token alignment."
    )
    parser.add_argument(
        "--aligned_dataset_tknz_save_dir",
        type=str,
        help="The local dir to save processed data tknz."
    )
    parser.add_argument(
        "--aligned_dataset_save_dir",
        type=str,
        help="The local dir to save processed representations."
    )
    parser.add_argument(
        "--temperature",
        type=str,
        help="Temperature used for softmax."
    )
    parser.add_argument(
        "--dataset_sample_prop",
        type=float,
        default=None,
        help="The prop to sample dataset. Debug only."
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

    base_dataset_list = args.base_dataset_dir.split(",")
    blending_dataset_list = args.blending_dataset_dir.split(",")

    if len(base_dataset_list) == 1:
        base_model_logits_datasets = load_from_disk(base_dataset_list[0])
    else :
        base_model_logits_datasets = datasets.DatasetDict()
        base_model_logits_datasets["train"] = datasets.concatenate_datasets([datasets.load_from_disk(_)["train"] for _ in base_dataset_list])

    if len(blending_dataset_list) == 1 :
        blending_model_logits_datasets = load_from_disk(blending_dataset_list[0])
    else :
        blending_model_logits_datasets = datasets.DatasetDict()
        blending_model_logits_datasets["train"] = datasets.concatenate_datasets([datasets.load_from_disk(_)["train"] for _ in blending_dataset_list])

    if args.dataset_sample_prop is not None:
        print(f"Sample prop: {args.dataset_sample_prop}.")
        data_len = len(base_model_logits_datasets["train"])
        select_size = int(data_len * args.dataset_sample_prop)
        random_numbers = random.sample(range(data_len), select_size)
        print(f"dataset length:{data_len} sampling size:{select_size} sampling from index:{random_numbers}\n")
        base_model_logits_datasets["train"] = base_model_logits_datasets["train"].select(random_numbers)
        blending_model_logits_datasets["train"] = blending_model_logits_datasets["train"].select(random_numbers)

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        trust_remote_code=args.base_model_trust_remote_code,
        use_fast=args.base_model_use_fast,
    )
    
    if "qwen" in base_tokenizer.name_or_path.lower():
        base_tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    elif "llama-3"in base_tokenizer.name_or_path.lower():
        base_tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})
    else:
        base_tokenizer.pad_token = base_tokenizer.unk_token

    blending_tokenizer = AutoTokenizer.from_pretrained(
        args.blending_model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        trust_remote_code=args.blending_model_trust_remote_code,
        use_fast=args.blending_model_use_fast,
    )

    if "qwen" in blending_tokenizer.name_or_path.lower():
        blending_tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    elif "llama-3"in blending_tokenizer.name_or_path.lower():
        blending_tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})
    else:
        blending_tokenizer.pad_token = blending_tokenizer.unk_token

    base_vocab = base_tokenizer.get_vocab()
    blending_vocab = blending_tokenizer.get_vocab()

    token_mapping_matrices = dict()

    if args.do_token_alignment:
        if args.align_type == "default":
            def token_alignment(examples_1, indices, dataset_2):
                features_1 = dict_to_list(examples_1)
                features_2 = dataset_2.select([idx for idx in indices])
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
                if os.path.exists(args.aligned_dataset_tknz_save_dir):
                    base_model_logits_with_token_mapping_datasets[k] = datasets.load_from_disk(args.aligned_dataset_tknz_save_dir)[k]
                else:
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
                    base_model_logits_with_token_mapping_datasets.save_to_disk(args.aligned_dataset_tknz_save_dir)
                if os.path.exists(args.token_alignment_matrix_file):
                    token_mapping_matrix_csr = sparse.load_npz(args.token_alignment_matrix_file)
                    token_mapping_matrix = token_mapping_matrix_csr.toarray()
                    print(f"Using existed token aligment matrix shape:\n{token_mapping_matrix.shape}")
                else:
                    token_mapping_matrix = np.zeros((len(base_vocab), len(blending_vocab)))
                    for idx in tqdm(range(len(base_model_logits_with_token_mapping_datasets[k]))):
                        base_to_blending_mapping = base_model_logits_with_token_mapping_datasets[k][idx]["base_to_blending_mapping"]
                        base_input_ids = base_model_logits_with_token_mapping_datasets[k][idx]["input_ids"]
                        for i, base_id in enumerate(base_input_ids):
                            token_mapping_matrix[base_id, base_to_blending_mapping[i]] += 1
                    token_mapping_sparse_matrix = sparse.csr_matrix(token_mapping_matrix)
                    sparse.save_npz(args.token_alignment_matrix_file, token_mapping_sparse_matrix)

                token_mapping_matrices[k] = token_mapping_matrix
                blending_to_base = token_align_mapping(blending_tokenizer, base_tokenizer, args.align_type, np.transpose(token_mapping_matrix))
                with open(args.blending_to_base_file, "w") as f:
                    json.dump(blending_to_base, f, ensure_ascii=False)
        else:
            blending_to_base = token_align_mapping(blending_tokenizer, base_tokenizer, args.align_type)
            with open(args.blending_to_base_file, "w") as f:
                json.dump(blending_to_base, f, ensure_ascii=False)
    else:
        if args.use_token_alignment_matrix:
            for k in ["train"]:
                token_mapping_matrix_csr = sparse.load_npz(args.token_alignment_matrix_file)
                token_mapping_matrix = token_mapping_matrix_csr.tocoo()
                token_mapping_matrices[k] = token_mapping_matrix
                if os.path.exists(args.blending_to_base_file):
                    with open(args.blending_to_base_file, "r") as f:
                        blending_to_base = json.load(f)
                else:
                    blending_to_base = token_align_mapping(blending_tokenizer, base_tokenizer, np.transpose(token_mapping_matrix))
                    with open(args.blending_to_base_file, "w") as f:
                        json.dump(blending_to_base, f, ensure_ascii=False)
        elif args.align_type == "soft":
            if os.path.exists(args.blending_to_base_file):
                with open(args.blending_to_base_file, "r") as f:
                    blending_to_base = json.load(f)
            token_mapping_matrix = None
        else:
            token_mapping_matrix = None
            blending_to_base = None
    
    if args.do_distribution_alignment:
        def distribution_alignment(examples_1, indices, dataset_2):
            features_1 = dict_to_list(examples_1)
            features_2 = dataset_2.select([idx for idx in indices])
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
                        align_type=args.align_type,
                        temperature=float(args.temperature),
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
            examples_1["fused_per_step_logits"] = aligned_per_step_logits_list
            examples_1["fused_per_step_indices"] = aligned_per_step_indices_list
            if args.metric_level == "sequence":
                examples_1["fused_metric_ce"] = metric_ce_aligned
                if "per_step_metric_ce" in examples_1:
                    del examples_1["per_step_metric_ce"]
            else:
                examples_1["per_step_fused_metric_ce"] = metric_ce_aligned
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
        base_model_blending_model_logits_datasets.save_to_disk(args.aligned_dataset_save_dir)


if __name__ == "__main__":
    main()