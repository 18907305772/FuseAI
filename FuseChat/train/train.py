# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence
import random;random.seed(42)

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import datasets

from model.model_adapter import get_conversation_template

from data_collator import DataCollatorForDistill
from trainer import DistillTrainer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    conv_temp: str = field(
        default="metamath", metadata={"help": "Conversation template."}
    )
    mask_instruction: bool = True
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn_transformers: bool = False
    # distill args
    do_distill: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to distill logits during training."}
    )
    distill_with_ref_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use ref model during distilling."}
    )
    distill_with_aligned_model_0: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use aligned model 0 duriing distilling."}
    )
    distill_with_aligned_model_1: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use aligned model 1 duriing distilling."}
    )
    distill_loss_type: Optional[str] = field(
        default="ce",
        metadata={"help": "The distill loss type, could be ce or kl."}
    )
    distill_teacher_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "The temperature used for teacher during distilling."}
    )
    lm_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "The weight of language loss during distilling."}
    )
    distill_greater_as_gt: Optional[bool] = field(
        default=False,
        metadata={"help": "Use logits from greater teacher as ground truth label."}
    )
    distill_greater_as_gt_type: Optional[str] = field(
        default="hard",
        metadata={"help": "hard or soft or hard_and_pair or soft_and_pair."}
    )
    distill_weighted_as_gt: Optional[bool] = field(
        default=False,
        metadata={"help": "Use logits from weighted teacher as ground truth label."}
    )
    distill_weighted_as_gt_type: Optional[str] = field(
        default="hard",
        metadata={"help": "hard or soft or hard_and_pair or soft_and_pair."}
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    conv_temp: list[str],
    mask_instruction: bool,
) -> Dict:
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv = get_conversation_template(conv_temp[i])
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conv.sep2 = tokenizer.eos_token
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    if mask_instruction:
        if "openchat" in conv_temp[0]:
            for idx, (conversation, target) in enumerate(zip(conversations, targets)):
                conv = get_conversation_template(conv_temp[idx])
                roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
                conv.sep2 = tokenizer.eos_token
                sep = conv.roles[1] + ": "
                total_len = int(target.ne(tokenizer.pad_token_id).sum())

                turns = conversation.split(conv.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_TOKEN_ID
                for i, turn in enumerate(turns):
                    if turn == "":
                        break
                    turn_len = len(tokenizer(turn).input_ids)
                    if i % 2 == 0:
                        target[cur_len: cur_len + turn_len] = IGNORE_TOKEN_ID
                        cur_len += turn_len
                    else:
                        part = sep
                        # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                        instruction_len = len(tokenizer(part).input_ids) - 2

                        if i != 0 and not tokenizer.legacy:
                            # The legacy and non-legacy modes handle special tokens differently
                            instruction_len -= 1

                        # Ignore the user instructions
                        target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                        cur_len += turn_len

                        if i != 0 and not tokenizer.legacy:
                            # The legacy and non-legacy modes handle special tokens differently
                            cur_len -= 1

                target[cur_len:] = IGNORE_TOKEN_ID

                if False:  # Inspect and check the correctness of masking
                    z = target.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    print(tokenizer.decode(input_ids[0]))
                    print(tokenizer.decode(z))
                    exit()

                if cur_len < tokenizer.model_max_length:
                    if cur_len != total_len:
                        # target[:] = IGNORE_TOKEN_ID  # TODO: Do not drop this target.
                        print(
                            f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                            f" #turn = {len(turns) - 1}. (ignored)"
                        )
        else:
            for idx, (conversation, target) in enumerate(zip(conversations, targets)):
                conv = get_conversation_template(conv_temp[idx])
                roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
                conv.sep2 = tokenizer.eos_token
                sep = conv.sep + conv.roles[1] + ": "
                total_len = int(target.ne(tokenizer.pad_token_id).sum())

                turns = conversation.split(conv.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_TOKEN_ID
                for i, turn in enumerate(turns):
                    if turn == "":
                        break
                    turn_len = len(tokenizer(turn).input_ids)

                    parts = turn.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                    if i != 0 and not tokenizer.legacy:
                        # The legacy and non-legacy modes handle special tokens differently
                        instruction_len -= 1

                    # Ignore the user instructions
                    target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                    if i != 0 and not tokenizer.legacy:
                        # The legacy and non-legacy modes handle special tokens differently
                        cur_len -= 1

                target[cur_len:] = IGNORE_TOKEN_ID

                if False:  # Inspect and check the correctness of masking
                    z = target.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    print(tokenizer.decode(input_ids[0]))
                    print(tokenizer.decode(z))
                    exit()

                if cur_len < tokenizer.model_max_length:
                    if cur_len != total_len:
                        # target[:] = IGNORE_TOKEN_ID  # TODO: Do not drop this target.
                        print(
                            f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                            f" #turn = {len(turns) - 1}. (ignored)"
                        )
    else:
        targets[targets == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, conv_temp: str, mask_instruction: bool = True):
        super(SupervisedDataset, self).__init__()

        print(f"Formatting inputs with '{conv_temp}' conversation template...")
        sources = [example["conversations"] for example in raw_data]
        conv_temp = [conv_temp for _ in range(len(raw_data))]
        data_dict = preprocess(sources, tokenizer, conv_temp, mask_instruction)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, conv_temp: str, mask_instruction: bool = True):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        print(f"Formatting inputs with '{conv_temp}' conversation template...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.conv_temp = conv_temp
        self.mask_instruction = mask_instruction
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, [self.conv_temp], self.mask_instruction)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    print(f"Loading data from {data_args.data_path}...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_json = random.sample(train_json, len(train_json))  # same as code from MetaMath
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, conv_temp=data_args.conv_temp, mask_instruction=data_args.mask_instruction)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, conv_temp=data_args.conv_temp, mask_instruction=data_args.mask_instruction)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def make_distill_data_module(
    tokenizer, data_args, training_args
) -> Dict:
    """make dataset and collator for distilling."""
    print(f"Loading data from {data_args.data_path}...")
    dataset_name_list = data_args.data_path.split(",")
    if len(dataset_name_list) == 1:
        raw_dataset = datasets.load_from_disk(dataset_name_list[0])
    else:
        raw_dataset = datasets.DatasetDict()
        if training_args.do_train:
            raw_dataset["train"] = datasets.concatenate_datasets([datasets.load_from_disk(_)['train'] for _ in dataset_name_list])
        if training_args.do_eval:
            raw_dataset["validation"] = datasets.concatenate_datasets([datasets.load_from_disk(_)['validation'] for _ in dataset_name_list])
    train_dataset = raw_dataset["train"].shuffle(seed=42)  # same as code from MetaMath
    data_collator = DataCollatorForDistill(tokenizer,
                                           padding="max_length",
                                           max_length=training_args.model_max_length,
                                           label_pad_token_id=IGNORE_TOKEN_ID,
                                           training_args=training_args,
                                           )
    if "validation" in raw_dataset:
        eval_dataset = raw_dataset["validation"]
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.generation_config = None  # use model.generation_config
    local_rank = training_args.local_rank

    trust_remote_code = False
    tknz_trust_remote_code = False
    use_fast = False

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=trust_remote_code
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    compute_dtype = (
        torch.bfloat16
        if training_args.bf16
        else (torch.float16 if training_args.fp16 else torch.float32)
    )

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        use_flash_attention_2=True if training_args.flash_attn_transformers else False,
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=tknz_trust_remote_code,
        use_fast=use_fast,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if training_args.do_distill:
        # Load data
        data_module = make_distill_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
        # Start trainner
        trainer = DistillTrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
    else:
        # Load data
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        # Start trainner
        trainer = Trainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if training_args.deepspeed is not None:
        if "zero_stage3" in training_args.deepspeed:
            trainer.save_model()  # deepspeed zero3 should use zero_to_fp32
        else:
            safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer, training_args.output_dir)


if __name__ == "__main__":
    train()
