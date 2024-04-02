"""All the config."""

from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "The max train samples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "The max eval samples."}
    )
    max_predict_samples: Optional[int] = field(
        default=None, metadata={"help": "The max predict samples."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=64,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    # common args
    training_mode: Optional[str] = field(
        default="full", metadata={"help": "The training mode: full or qlora."}
    )
    use_flash_attn: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use flash attention."}
    )
    cache_dir: Optional[str] = field(default=None)
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). "
            "0 means that the data will be loaded in the main process."
        },
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    optim: str = field(
        default="adamw_torch", metadata={"help": "adamw_torch or paged_adamw_32bit."}
    )
    # qlora args
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    max_memory_MB: int = field(default=40000, metadata={"help": "Free memory per gpu."})
    # distill args
    do_distill: Optional[bool] = field(
        default=False, metadata={"help": "Whether to distill logits during training."}
    )
    distill_with_ref_model: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use ref model during distilling."}
    )
    distill_with_aligned_model_0: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use aligned model 0 duriing distilling."},
    )
    distill_with_aligned_model_1: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use aligned model 1 duriing distilling."},
    )
    distill_loss_type: Optional[str] = field(
        default="ce", metadata={"help": "The distill loss type, could be ce or kl."}
    )
    distill_teacher_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "The temperature used for teacher during distilling."},
    )
    lm_loss_weight: Optional[float] = field(
        default=1.0, metadata={"help": "The weight of language loss during distilling."}
    )
    distill_greater_as_gt: Optional[bool] = field(
        default=False,
        metadata={"help": "Use logits from greater teacher as ground truth label."},
    )
    distill_greater_as_gt_type: Optional[str] = field(
        default="hard", metadata={"help": "hard or hard_and_decay or soft."}
    )
    distill_weighted_as_gt: Optional[bool] = field(
        default=False,
        metadata={"help": "Use logits from weighted teacher as ground truth label."},
    )
    distill_weighted_as_gt_type: Optional[str] = field(
        default="hard", metadata={"help": "hard or hard_and_decay or soft."}
    )


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_length: Optional[int] = field(default=4096)
    max_new_tokens: Optional[int] = field(default=None)
    min_new_tokens: Optional[int] = field(default=None)

    # Generation strategy
    do_sample: Optional[bool] = field(default=True)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=0.6)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=0.9)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)
