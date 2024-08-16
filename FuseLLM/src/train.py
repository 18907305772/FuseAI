"""Full parameters & QLoRA Training."""

import json
import math
import os
import pathlib

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import EvalPrediction, Seq2SeqTrainer, set_seed
from utils.common import (
    load_tokenizer_and_model,
    prepare_args,
)
from utils.data_collator import DataCollatorForDistill, DataCollatorForSeq2Seq
from utils.others import (
    IGNORE_TOKEN_ID,
    SavePeftModelCallback,
    get_logger,
    safe_save_model_for_hf_trainer,
)
from utils.trainer import DistillTrainer

logger = get_logger(__name__)

local_rank = None


def train():
    global local_rank
    model_args, data_args, training_args, args = prepare_args()
    if args.use_flash_attn and "llama" in args.model_name_or_path.lower():
        from utils.llama_flash_attn_monkey_patch import (
            replace_llama_attn_with_flash_attn,
        )

        replace_llama_attn_with_flash_attn()
    if args.deepspeed is not None and "zero_stage3" in args.deepspeed:
        logger.info("Must use zero_to_fp32.py to save model!")
    local_rank = args.local_rank
    set_seed(args.seed)
    tokenizer, model = load_tokenizer_and_model(args)
    dataset_name_list = args.dataset_name.split(",")
    logger.info(f"Loading {len(dataset_name_list)} dataset/datasets.")
    if len(dataset_name_list) == 1:
        raw_dataset = load_from_disk(dataset_name_list[0])
    else:
        raw_dataset = DatasetDict()
        if args.do_train:
            raw_dataset["train"] = concatenate_datasets(
                [load_from_disk(_)["train"] for _ in dataset_name_list]
            )
        if args.do_eval:
            raw_dataset["validation"] = concatenate_datasets(
                [load_from_disk(_)["validation"] for _ in dataset_name_list]
            )
        if args.do_eval:
            raw_dataset["test"] = concatenate_datasets(
                [load_from_disk(_)["test"] for _ in dataset_name_list]
            )
    dataset = raw_dataset
    if args.do_train:
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    if args.do_eval:
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if args.do_distill:
        data_collator = DataCollatorForDistill(
            tokenizer,
            padding="max_length",
            max_length=args.model_max_length,
            label_pad_token_id=IGNORE_TOKEN_ID,
            training_args=training_args,
        )
        trainer = DistillTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if args.do_eval else None,
            data_collator=data_collator,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            padding="max_length",
            max_length=args.model_max_length,
            label_pad_token_id=IGNORE_TOKEN_ID,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if args.do_eval else None,
            data_collator=data_collator,
        )
    if args.training_mode == "qlora":
        trainer.add_callback(SavePeftModelCallback)
    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        if args.training_mode == "full" and list(
            pathlib.Path(args.output_dir).glob("checkpoint-*")
        ):
            train_result = trainer.train(resume_from_checkpoint=True)
        else:
            # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
            # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
            train_result = trainer.train()
        model.config.use_cache = True
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if args.training_mode == "full":
            if args.deepspeed is not None and "zero_stage3" in args.deepspeed:
                trainer.save_model()
            else:
                safe_save_model_for_hf_trainer(
                    trainer=trainer, output_dir=args.output_dir
                )
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if args.do_train or args.do_eval:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
