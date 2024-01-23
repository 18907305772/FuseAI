"""Prepare args, tokenizer, model."""
import argparse
import transformers
from .others import (
    get_logger,
    get_tokenizer,
    get_accelerate_model,
    get_base_model,
    get_last_checkpoint_for_lora,
)
from .config import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments
)

logger = get_logger(__name__)


def prepare_args():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    if args.training_mode == "full":
        args.optim = "adamw_torch"
    elif args.training_mode == "qlora":
        args.optim = "paged_adamw_32bit"
    else:
        logger.warning(f"Now {args.training_mode} is not supported.")
        raise NotImplementedError
    logger.info(f"Training/Evaluation Args: {args}")
    return model_args, data_args, training_args, args


def load_tokenizer_and_model(args):
    tokenizer, kwargs = get_tokenizer(args.model_name_or_path, args.cache_dir, args.model_max_length)
    if args.training_mode == "full":
        model = get_base_model(args, trust_remote_code=kwargs["model_trust_remote_code"])
    elif args.training_mode == "qlora":
        checkpoint_dir, completed_training = get_last_checkpoint_for_lora(args.output_dir)
        model = get_accelerate_model(args, checkpoint_dir=checkpoint_dir,
                                     trust_remote_code=kwargs["model_trust_remote_code"])
    else:
        logger.warning(f"Now {args.training_mode} is not supported.")
        raise NotImplementedError
    model.config.use_cache = False
    return tokenizer, model
