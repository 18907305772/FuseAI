"""Other functions."""

import gc
import logging
import os
import sys
from typing import Dict, List

import editdistance
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)


logger = get_logger(__name__)


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.GPTNeoXTokenizerFast: "Ġ",
}


# get tokenizer
def get_tokenizer(model_name_or_path, cache_dir, model_max_length):
    kwargs = {
        "use_fast": False,
        "tokenizer_trust_remote_code": False,
        "model_trust_remote_code": False,
    }
    if "llama" in model_name_or_path.lower():
        kwargs["use_fast"] = False
        kwargs["tokenizer_trust_remote_code"] = False
        kwargs["model_trust_remote_code"] = False
    elif "mpt" in model_name_or_path.lower():
        kwargs["use_fast"] = True
        kwargs["tokenizer_trust_remote_code"] = True
        kwargs["model_trust_remote_code"] = True
    else:
        raise NotImplementedError
    logger.info("Loading tokenizer.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=kwargs["use_fast"],
        trust_remote_code=kwargs["tokenizer_trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError
    logger.info(
        f"bos_token: {tokenizer.bos_token}, {tokenizer.bos_token_id} "
        f"eos_token: {tokenizer.eos_token}, {tokenizer.eos_token_id} "
        f"unk_token: {tokenizer.unk_token}, {tokenizer.unk_token_id} "
        f"pad_token: {tokenizer.pad_token}, {tokenizer.pad_token_id} "
    )
    return tokenizer, kwargs


# get base or peft model
def get_base_model(args, trust_remote_code):
    logger.info("Loading base model.")
    if args.use_flash_attn and "mpt" in args.model_name_or_path.lower():
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, trust_remote_code=trust_remote_code
        )
        config.attn_config["attn_impl"] = "triton"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            trust_remote_code=trust_remote_code,
            config=config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            trust_remote_code=trust_remote_code,
        )
    return model


def find_all_linear_names(args, model):
    import bitsandbytes as bnb

    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_last_checkpoint_for_lora(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(
                os.path.join(checkpoint_dir, filename)
            ) and filename.startswith("checkpoint"):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint-{max_step}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def get_accelerate_model(args, checkpoint_dir, trust_remote_code):
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer
    from transformers import BitsAndBytesConfig

    n_gpus = torch.cuda.device_count()
    max_memory = f"{args.max_memory_MB}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {"": max_memory[local_rank]}

    compute_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )
    logger.info("Loading base model.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(
            torch.float32
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        ),
        trust_remote_code=trust_remote_code,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if checkpoint_dir is not None:
        logger.info("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(
            model, os.path.join(checkpoint_dir, "adapter_model"), is_trainable=True
        )
    else:
        logger.info("Adding lora module.")
        modules = find_all_linear_names(args, model)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


# save base or peft model
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info("Saving PEFT checkpoint...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def release_model_and_tensor(model):
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def dict_to_list(examples):
    return [
        {key: examples[key][i] for key in examples}
        for i in range(len(examples[next(iter(examples))]))
    ]


def list_to_dict(examples):
    return {key: [d[key] for d in examples] for key in examples[0].keys()}


class AttrDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"No such attribute: {key}")


def dtw(series_1, series_2, norm_func=np.linalg.norm):
    """Use dynamic time wrapping to align to tokenizers, modified from:
    https://github.com/talcs/simpledtw/blob/master/simpledtw.py"""
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
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


def transform_step_logits(
    base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    base_model_vocab: Dict[str, int],
    base_model_input_ids: List[int],
    blending_model_input_ids: List[int],
    blending_model_per_step_logits: List[List[float]],
    blending_model_per_step_indices: List[List[int]],
    vocab_align_type: str = "hard",
    blending_to_base_mapping: Dict[str, str] = None,
):
    """Align blending model per step logits & indices with base model."""
    base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
    blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
        blending_model_input_ids
    )
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        base_model_tokenizer.__class__
    ]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        blending_model_tokenizer.__class__
    ]

    def dist_fn(a, b):
        """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
        aa = a.replace(blending_model_special_token, "")
        bb = b.replace(base_model_special_token, "")
        dist = editdistance.eval(aa, bb)
        return dist

    _, _, _, base_to_blending, _ = dtw(
        blending_model_tokens, base_model_tokens, norm_func=dist_fn
    )
    aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = (
        [],
        [],
    )
    for i, blending_idx in enumerate(base_to_blending):
        aligned_blending_model_per_step_logit = []
        aligned_blending_model_per_step_index = []
        if len(blending_idx) == 1:  # one base token map to one blending token
            j = blending_idx[0]
            base_token = base_model_tokens[i]
            blending_token = blending_model_tokens[j].replace(
                blending_model_special_token, base_model_special_token
            )
            if (
                (
                    blending_model_tokenizer.__class__
                    == transformers.GPTNeoXTokenizerFast
                    or blending_model_tokenizer.__class__
                    == transformers.GPT2TokenizerFast
                )
                and i == 0
                and base_token.startswith(base_model_special_token)
                and not blending_token.startswith(base_model_special_token)
            ):
                blending_token = (
                    base_model_special_token + blending_token
                )  # special case for mpt
            if vocab_align_type == "hard":
                if (
                    base_token == blending_token
                ):  # find the aligned mapping, use the corresponding logits
                    # the logits and indices at this step
                    for blending_logit, blending_index in zip(
                        blending_model_per_step_logits[j],
                        blending_model_per_step_indices[j],
                    ):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                            [blending_index]
                        )[0].replace(
                            blending_model_special_token, base_model_special_token
                        )
                        if blending_t in base_model_vocab:
                            aligned_index = base_model_vocab[
                                blending_t
                            ]  # the index of the token in base model vocab
                            if (
                                aligned_index
                                not in aligned_blending_model_per_step_index
                            ):
                                aligned_blending_model_per_step_index.append(
                                    aligned_index
                                )
                                aligned_blending_model_per_step_logit.append(
                                    blending_logit
                                )
                else:  # find error aligned mapping, use the one-hot logits
                    aligned_blending_model_per_step_index.append(
                        base_model_vocab[base_token]
                    )
                    aligned_blending_model_per_step_logit.append(1.0)
            elif vocab_align_type == "soft":
                if (base_token == blending_token) or (
                    blending_token in blending_to_base_mapping
                    and base_token == blending_to_base_mapping[blending_token]
                ):  # find the aligned mapping, use the corresponding logits
                    # the logits and indices at this step
                    for blending_logit, blending_index in zip(
                        blending_model_per_step_logits[j],
                        blending_model_per_step_indices[j],
                    ):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                            [blending_index]
                        )[0].replace(
                            blending_model_special_token, base_model_special_token
                        )
                        blending_t = blending_to_base_mapping[blending_t]
                        if blending_t in base_model_vocab:
                            aligned_index = base_model_vocab[
                                blending_t
                            ]  # the index of the token in base model vocab
                            if (
                                aligned_index
                                not in aligned_blending_model_per_step_index
                            ):
                                aligned_blending_model_per_step_index.append(
                                    aligned_index
                                )
                                aligned_blending_model_per_step_logit.append(
                                    blending_logit
                                )
                        else:
                            logger.warning(
                                f"blending_t: {blending_t} not in base_model_vocab!"
                            )
                else:  # find error aligned mapping, use the one-hot logits
                    aligned_blending_model_per_step_index.append(
                        base_model_vocab[base_token]
                    )
                    aligned_blending_model_per_step_logit.append(1.0)
            else:
                logger.warning(
                    f"The vocab_align_type: '{vocab_align_type}' is not support!"
                )
                raise NotImplementedError
        else:  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
            base_token = base_model_tokens[i]
            aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
            aligned_blending_model_per_step_logit.append(1.0)
        aligned_blending_model_per_step_indices.append(
            aligned_blending_model_per_step_index
        )
        aligned_blending_model_per_step_logits.append(
            aligned_blending_model_per_step_logit
        )
    return (
        aligned_blending_model_per_step_logits,
        aligned_blending_model_per_step_indices,
    )
