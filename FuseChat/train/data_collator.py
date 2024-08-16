# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Dict, Sequence, Any, Union
import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import Seq2SeqTrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from dataclasses import dataclass

@dataclass
class DataCollatorForSFT:
    """
    Data collator that will dynamically pad the inputs and labels, then weighted sum and pad all logits.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 model: Optional[Any] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 label_pad_token_id: int = -100,
                 return_tensors: str = "pt",
                 training_args: Seq2SeqTrainingArguments = None,
                 ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.training_args = training_args
        self.vocab = self.tokenizer.get_vocab().keys()
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = self.max_length
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        delete_keys = dict()  # save other keys
        for k in features[0].keys():
            if k not in ["input_ids", "attention_mask", "labels"]:
                delete_keys[k] = []
        if len(delete_keys.keys()) > 0:
            for feature in features:
                for k in delete_keys.keys():
                    delete_keys[k].append(feature[k])
                    del feature[k]
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        if len(delete_keys.keys()) > 0:
            for k, v in delete_keys.items():
                features[k] = v
        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

@dataclass
class DataCollatorForFuse:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 model: Optional[Any] = None,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 label_pad_token_id: int = -100,
                 return_tensors: str = "pt",
                 training_args: Seq2SeqTrainingArguments = None,
                 ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.training_args = training_args
        self.vocab = self.tokenizer.get_vocab().keys()
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = self.max_length
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        delete_keys = dict()  # save other keys
        for k in features[0].keys():
            if k not in ["input_ids", "attention_mask", "labels"]:
                delete_keys[k] = []
        if len(delete_keys.keys()) > 0:
            for feature in features:
                for k in delete_keys.keys():
                    delete_keys[k].append(feature[k])
                    del feature[k]
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        if len(delete_keys.keys()) > 0:
            for k, v in delete_keys.items():
                features[k] = v
        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        # weighted sum and pad all logits
        batch_size = features["input_ids"].size(0)
        vocab_size = len(self.vocab)
        base_target_dist = torch.zeros(batch_size, self.max_length, vocab_size).to(torch.bfloat16)
        fused_target_dist = torch.zeros(batch_size, self.max_length, vocab_size).to(torch.bfloat16)
        for i in range(batch_size):
            base_seq_len = len(features["per_step_logits"][i])
            for j in range(self.max_length):
                if j < base_seq_len:
                    base_logits = torch.tensor(features["per_step_logits"][i][j], dtype=torch.bfloat16)
                    base_prob = softmax(base_logits / self.training_args.fuse_temperature, -1)
                    base_indices = torch.tensor(features["per_step_indices"][i][j])
                    base_target_dist[i][j] = base_target_dist[i][j].scatter_(-1, base_indices, base_prob)

                    if fused_target_dist is not None and len(features["fused_per_step_indices"][i][j]) > 0:
                        fused_logits = torch.tensor(features["fused_per_step_logits"][i][j], dtype=torch.bfloat16)
                        fused_prob = softmax(fused_logits / self.training_args.fuse_temperature, -1)
                        fused_indices = torch.tensor(features["fused_per_step_indices"][i][j])
                        fused_target_dist[i][j] = fused_target_dist[i][j].scatter_(-1, fused_indices, fused_prob)
                    elif fused_target_dist is not None:
                        fused_target_dist[i][j] = base_target_dist[i][j]  # bad case

                else:  # padding position
                    base_target_dist[i][j][self.pad_id] = 1.0
                    if fused_target_dist is not None:
                        fused_target_dist[i][j][self.pad_id] = 1.0

        features.pop("per_step_logits")
        features.pop("per_step_indices")
        if "fused_per_step_logits" in features:
            features.pop("fused_per_step_logits")
            features.pop("fused_per_step_indices")

        if self.training_args.fuse_with_ref_model is True:
            features["base_target_dist"] = base_target_dist
        else:
            features.pop("metric_ce")
        if fused_target_dist is not None:
            features["fused_target_dist"] = fused_target_dist
        elif "fused_metric_ce" in features:
            features.pop("fused_metric_ce")
        return features
