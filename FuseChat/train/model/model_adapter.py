"""Model adapter registration."""

import math
import os
import re
import sys
from typing import Dict, List, Optional
import warnings

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache

import accelerate
import psutil
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)

from conversation import Conversation, get_conv_template

def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""
    import torch

    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


# Check an environment variable to check if we should be sharing Peft model
# weights.  When false we treat all Peft models as separate.
peft_share_base_weights = (
    os.environ.get("PEFT_SHARE_BASE_WEIGHTS", "false").lower() == "true"
)


ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-instant-1",
)


class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=self.use_fast_tokenizer,
                revision=revision,
                trust_remote_code=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, revision=revision, trust_remote_code=True
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        except NameError:
            model = AutoModel.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


# A global registry for all model adapters
# TODO (lmzheng): make it a priority queue.
model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")


def raise_warning_for_incompatible_cpu_offloading_configuration(
    device: str, load_8bit: bool, cpu_offloading: bool
):
    if cpu_offloading:
        if not load_8bit:
            warnings.warn(
                "The cpu-offloading feature can only be used while also using 8-bit-quantization.\n"
                "Use '--load-8bit' to enable 8-bit-quantization\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
        if not "linux" in sys.platform:
            warnings.warn(
                "CPU-offloading is only supported on linux-systems due to the limited compatability with the bitsandbytes-package\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
        if device != "cuda":
            warnings.warn(
                "CPU-offloading is only enabled when using CUDA-devices\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
    return cpu_offloading


def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template."""
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)


def remove_parent_directory_name(model_path):
    """Remove parent directory name."""
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    return model_path.split("/")[-1]


peft_model_cache = {}


class PeftModelAdapter:
    """Loads any "peft" model and it's base model."""

    def match(self, model_path: str):
        """Accepts any model path with "peft" in the name"""
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            return True
        return "peft" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        """Loads the base model then the (peft) adapter weights"""
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(model_path)
        base_model_path = config.base_model_name_or_path
        if "peft" in base_model_path:
            raise ValueError(
                f"PeftModelAdapter cannot load a base model with 'peft' in the name: {config.base_model_name_or_path}"
            )

        # Basic proof of concept for loading peft adapters that share the base
        # weights.  This is pretty messy because Peft re-writes the underlying
        # base model and internally stores a map of adapter layers.
        # So, to make this work we:
        #  1. Cache the first peft model loaded for a given base models.
        #  2. Call `load_model` for any follow on Peft models.
        #  3. Make sure we load the adapters by the model_path.  Why? This is
        #  what's accessible during inference time.
        #  4. In get_generate_stream_function, make sure we load the right
        #  adapter before doing inference.  This *should* be safe when calls
        #  are blocked the same semaphore.
        if peft_share_base_weights:
            if base_model_path in peft_model_cache:
                model, tokenizer = peft_model_cache[base_model_path]
                # Super important: make sure we use model_path as the
                # `adapter_name`.
                model.load_adapter(model_path, adapter_name=model_path)
            else:
                base_adapter = get_model_adapter(base_model_path)
                base_model, tokenizer = base_adapter.load_model(
                    base_model_path, from_pretrained_kwargs
                )
                # Super important: make sure we use model_path as the
                # `adapter_name`.
                model = PeftModel.from_pretrained(
                    base_model, model_path, adapter_name=model_path
                )
                peft_model_cache[base_model_path] = (model, tokenizer)
            return model, tokenizer

        # In the normal case, load up the base model weights again.
        base_adapter = get_model_adapter(base_model_path)
        base_model, tokenizer = base_adapter.load_model(
            base_model_path, from_pretrained_kwargs
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        """Uses the conv template of the base model"""
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(model_path)
        if "peft" in config.base_model_name_or_path:
            raise ValueError(
                f"PeftModelAdapter cannot load a base model with 'peft' in the name: {config.base_model_name_or_path}"
            )
        base_model_path = config.base_model_name_or_path
        base_adapter = get_model_adapter(base_model_path)
        return base_adapter.get_default_conv_template(config.base_model_name_or_path)


class VicunaAdapter(BaseModelAdapter):
    "Model adapter for Vicuna models (e.g., lmsys/vicuna-7b-v1.5)" ""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "vicuna" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        self.raise_warning_for_old_weights(model)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "v0" in remove_parent_directory_name(model_path):
            return get_conv_template("one_shot")
        return get_conv_template("vicuna_v1.1")

    def raise_warning_for_old_weights(self, model):
        if isinstance(model, LlamaForCausalLM) and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fastchat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.3: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommended).\n"
            )


class OpenChat35Adapter(BaseModelAdapter):
    """The model adapter for OpenChat 3.5 (e.g. openchat/openchat_3.5)"""

    def match(self, model_path: str):
        if "openchat_3.5" in model_path.lower() or "fusechat" in model_path.lower() or "starling" in model_path.lower():
            return True
        return False

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("openchat_3.5")


class PretrainAdapter(BaseModelAdapter):
    """The model adapter for Pretrain"""

    def match(self, model_path: str):
        return "pretrain" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("pretrain")


class AiroborosAdapter(BaseModelAdapter):
    """The model adapter for jondurbin/airoboros-*"""

    def match(self, model_path: str):
        if re.search(r"airoboros|spicyboros", model_path, re.I):
            return True
        return False

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "-3." in model_path or "-3p" in model_path:
            return get_conv_template("airoboros_v3")
        if "spicyboros" in model_path or re.search(r"-(2\.[2-9]+)", model_path):
            return get_conv_template("airoboros_v2")
        return get_conv_template("airoboros_v1")

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        if "mpt" not in model_path.lower():
            return super().load_model(model_path, from_pretrained_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_seq_len=8192,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        return model, tokenizer




class GoogleT5Adapter(BaseModelAdapter):
    """The model adapter for google/Flan based models, such as Salesforce/codet5p-6b, lmsys/fastchat-t5-3b-v1.0, flan-t5-*, flan-ul2"""

    def match(self, model_path: str):
        return any(
            model_str in model_path.lower()
            for model_str in ["flan-", "fastchat-t5", "codet5p"]
        )

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = T5Tokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer


class KoalaAdapter(BaseModelAdapter):
    """The model adapter for Koala"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "koala" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("koala_v1")


class AlpacaAdapter(BaseModelAdapter):
    """The model adapter for Alpaca"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "alpaca" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")


class ChatGLMAdapter(BaseModelAdapter):
    """The model adapter for THUDM/chatglm-6b, THUDM/chatglm2-6b"""

    def match(self, model_path: str):
        return "chatglm" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, **from_pretrained_kwargs
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "chatglm2" in model_path.lower():
            return get_conv_template("chatglm2")
        return get_conv_template("chatglm")


class DollyV2Adapter(BaseModelAdapter):
    """The model adapter for databricks/dolly-v2-12b"""

    def match(self, model_path: str):
        return "dolly-v2" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("dolly_v2")


class OasstPythiaAdapter(BaseModelAdapter):
    """The model adapter for OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"""

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "oasst" in model_path and "pythia" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("oasst_pythia")

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer


class OasstLLaMAAdapter(BaseModelAdapter):
    """The model adapter for OpenAssistant/oasst-sft-7-llama-30b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        model_path = model_path.lower()
        if "openassistant-sft-7-llama-30b-hf" in model_path:
            return True
        return "oasst" in model_path and "pythia" not in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("oasst_llama")


class PythiaAdapter(BaseModelAdapter):
    """The model adapter for any EleutherAI/pythia model"""

    def match(self, model_path: str):
        return "pythia" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer


class StableLMAdapter(BaseModelAdapter):
    """The model adapter for StabilityAI/stablelm-tuned-alpha-7b"""

    def match(self, model_path: str):
        return "stablelm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("stablelm")


class MPTAdapter(BaseModelAdapter):
    """The model adapter for MPT series (mosaicml/mpt-7b-chat, mosaicml/mpt-30b-chat)"""

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "mpt" in model_path and not "airoboros" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_seq_len=8192,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "mpt-7b-chat" in model_path:
            return get_conv_template("mpt-7b-chat")
        elif "mpt-30b-chat" in model_path:
            return get_conv_template("mpt-30b-chat")
        elif "mpt-30b-instruct" in model_path:
            return get_conv_template("mpt-30b-instruct")
        else:
            print(
                "Warning: Loading base MPT model with `zero_shot` conversation configuration.  "
                "If this is not desired, inspect model configurations and names."
            )
            return get_conv_template("zero_shot")


class BaizeAdapter(BaseModelAdapter):
    """The model adapter for project-baize/baize-v2-7b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "baize" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("baize")


class OpenBuddyAdapter(BaseModelAdapter):
    """The model adapter for OpenBuddy/openbuddy-7b-v1.1-bf16-enc"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "openbuddy" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("openbuddy")


class PhoenixAdapter(BaseModelAdapter):
    """The model adapter for FreedomIntelligence/phoenix-inst-chat-7b"""

    def match(self, model_path: str):
        return "phoenix" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phoenix")


class ReaLMAdapter(BaseModelAdapter):
    """The model adapter for FreedomIntelligence/ReaLM-7b"""

    def match(self, model_path: str):
        return "ReaLM" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ReaLM-7b-v1")


class ChatGPTAdapter(BaseModelAdapter):
    """The model adapter for ChatGPT"""

    def match(self, model_path: str):
        return model_path in ("gpt-3.5-turbo", "gpt-4")

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("chatgpt")


class ClaudeAdapter(BaseModelAdapter):
    """The model adapter for Claude"""

    def match(self, model_path: str):
        return model_path in ANTHROPIC_MODEL_LIST

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("claude")


class BardAdapter(BaseModelAdapter):
    """The model adapter for Bard"""

    def match(self, model_path: str):
        return model_path == "bard"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bard")


class PaLM2Adapter(BaseModelAdapter):
    """The model adapter for PaLM2"""

    def match(self, model_path: str):
        return model_path == "palm-2"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bard")


class BiLLaAdapter(BaseModelAdapter):
    """The model adapter for Neutralzz/BiLLa-7B-SFT"""

    def match(self, model_path: str):
        return "billa" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("billa")


class RedPajamaINCITEAdapter(BaseModelAdapter):
    """The model adapter for togethercomputer/RedPajama-INCITE-7B-Chat"""

    def match(self, model_path: str):
        return "redpajama-incite" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("redpajama-incite")


class H2OGPTAdapter(BaseModelAdapter):
    """The model adapter for h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "h2ogpt" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("h2ogpt")


class RobinAdapter(BaseModelAdapter):
    """The model adapter for LMFlow/Full-Robin-7b-v2"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "robin" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("Robin")


class SnoozyAdapter(BaseModelAdapter):
    """The model adapter for nomic-ai/gpt4all-13b-snoozy"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "gpt4all" in model_path and "snoozy" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("snoozy")


class WizardLMAdapter(BaseModelAdapter):
    """The model adapter for WizardLM/WizardLM-13B-V1.0"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "wizardlm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "13b" in model_path or "30b" in model_path or "70b" in model_path:
            return get_conv_template("vicuna_v1.1")
        else:
            # TODO: use the recommended template for 7B
            # (https://huggingface.co/WizardLM/WizardLM-13B-V1.0)
            return get_conv_template("one_shot")


class ManticoreAdapter(BaseModelAdapter):
    """The model adapter for openaccess-ai-collective/manticore-13b-chat-pyg"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "manticore" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("manticore")


class GuanacoAdapter(BaseModelAdapter):
    """The model adapter for timdettmers/guanaco-33b-merged"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "guanaco" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        # Fix a bug in tokenizer config
        tokenizer.eos_token_id = model.config.eos_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("zero_shot")


class ChangGPTAdapter(BaseModelAdapter):
    """The model adapter for lcw99/polyglot-ko-12.8b-chang-instruct-chat"""

    def match(self, model_path: str):
        model_path = model_path.lower()
        return "polyglot" in model_path and "chang" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("polyglot_changgpt")


class CamelAdapter(BaseModelAdapter):
    """The model adapter for camel-ai/CAMEL-13B-Combined-Data"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "camel" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("vicuna_v1.1")


class TuluAdapter(BaseModelAdapter):
    """The model adapter for allenai/tulu-30b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "tulu" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("tulu")


class FalconAdapter(BaseModelAdapter):
    """The model adapter for tiiuae/falcon-40b"""

    def match(self, model_path: str):
        return "falcon" in model_path.lower() and "chat" not in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        # Strongly suggest using bf16, which is recommended by the author of Falcon
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        # In Falcon tokenizer config and special config there is not any pad token
        # Setting `pad_token_id` to 9, which corresponds to special token '>>SUFFIX<<'
        tokenizer.pad_token_id = 9
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("falcon")


class FalconChatAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "falcon" in model_path.lower() and "chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("falcon-chat")


class TigerBotAdapter(BaseModelAdapter):
    """The model adapter for TigerResearch/tigerbot-7b-sft"""

    def match(self, model_path: str):
        return "tigerbot" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("tigerbot")


class BaichuanAdapter(BaseModelAdapter):
    """The model adapter for Baichuan models (e.g., baichuan-inc/Baichuan-7B)"""

    def match(self, model_path: str):
        return "baichuan" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        # for Baichuan-13B-Chat
        if "chat" in model_path.lower():
            if "baichuan2" in model_path.lower():
                return get_conv_template("baichuan2-chat")
            return get_conv_template("baichuan-chat")
        return get_conv_template("zero_shot")


class XGenAdapter(BaseModelAdapter):
    """The model adapter for Salesforce/xgen-7b"""

    def match(self, model_path: str):
        return "xgen" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        model.config.eos_token_id = 50256
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("xgen")


class NousHermesAdapter(BaseModelAdapter):
    """The model adapter for NousResearch/Nous-Hermes-13b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "nous-hermes" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")


class InternLMChatAdapter(BaseModelAdapter):
    """The model adapter for internlm/internlm-chat-7b"""

    def match(self, model_path: str):
        return "internlm2" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("internlm2")


class StarChatAdapter(BaseModelAdapter):
    """The model adapter for HuggingFaceH4/starchat-beta"""

    def match(self, model_path: str):
        return "starchat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("starchat")


class MistralAdapter(BaseModelAdapter):
    """The model adapter for Mistral AI models"""

    def match(self, model_path: str):
        return "mistral" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("mistral")


class Llama2Adapter(BaseModelAdapter):
    """The model adapter for Llama-2 (e.g., meta-llama/Llama-2-7b-hf)"""

    def match(self, model_path: str):
        return "llama-2" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")


class CuteGPTAdapter(BaseModelAdapter):
    """The model adapter for CuteGPT"""

    def match(self, model_path: str):
        return "cutegpt" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<end>")
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("cutegpt")


class OpenOrcaAdapter(BaseModelAdapter):
    """Model adapter for Open-Orca models which may use different prompt templates
    - (e.g. Open-Orca/OpenOrcaxOpenChat-Preview2-13B, Open-Orca/Mistral-7B-OpenOrca)
    - `OpenOrcaxOpenChat-Preview2-13B` uses their "OpenChat Llama2 V1" prompt template.
        - [Open-Orca/OpenOrcaxOpenChat-Preview2-13B #Prompt Template](https://huggingface.co/Open-Orca/OpenOrcaxOpenChat-Preview2-13B#prompt-template)
    - `Mistral-7B-OpenOrca` uses the [OpenAI's Chat Markup Language (ChatML)](https://github.com/openai/openai-python/blob/main/chatml.md)
        format, with <|im_start|> and <|im_end|> tokens added to support this.
        - [Open-Orca/Mistral-7B-OpenOrca #Prompt Template](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca#prompt-template)
    """

    use_fast_tokenizer = False

    def match(self, model_path: str):
        if "mistral-7b-openorca" in model_path.lower():
            return get_conv_template("mistral-7b-openorca")
        elif "openorca" in model_path.lower():
            return get_conv_template("open-orca")
        else:
            return False

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        ).eval()
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("open-orca")


class WizardCoderAdapter(BaseModelAdapter):
    """The model adapter for WizardCoder (e.g., WizardLM/WizardCoder-Python-34B-V1.0)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "wizardcoder" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        # Same as Alpaca, see :
        # https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/inference_wizardcoder.py#L60
        return get_conv_template("alpaca")


class QwenChatAdapter(BaseModelAdapter):
    """The model adapter for Qwen/Qwen-7B-Chat
    To run this model, you need to ensure additional flash attention installation:
    ``` bash
    git clone https://github.com/Dao-AILab/flash-attention
    cd flash-attention && pip install .
    pip install csrc/layer_norm
    pip install csrc/rotary
    ```

    Since from 2.0, the following change happened
    - `flash_attn_unpadded_func` -> `flash_attn_varlen_func`
    - `flash_attn_unpadded_qkvpacked_func` -> `flash_attn_varlen_qkvpacked_func`
    - `flash_attn_unpadded_kvpacked_func` -> `flash_attn_varlen_kvpacked_func`
    You may need to revise the code in: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/modeling_qwen.py#L69
    to from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    """

    def match(self, model_path: str):
        return "qwen-chat" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        from transformers.generation import GenerationConfig

        revision = from_pretrained_kwargs.get("revision", "main")
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        # NOTE: if you use the old version of model file, please remove the comments below
        # config.use_flash_attn = False
        config.fp16 = True
        generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        ).eval()
        if hasattr(model.config, "use_dynamic_ntk") and model.config.use_dynamic_ntk:
            model.config.max_sequence_length = 16384
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        tokenizer.eos_token_id = config.eos_token_id
        tokenizer.bos_token_id = config.bos_token_id
        tokenizer.pad_token_id = generation_config.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen-7b-chat")

class QwenAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "qwen" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        from transformers.generation import GenerationConfig

        revision = from_pretrained_kwargs.get("revision", "main")
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        # NOTE: if you use the old version of model file, please remove the comments below
        # config.use_flash_attn = False
        config.fp16 = True
        generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        ).eval()
        if hasattr(model.config, "use_dynamic_ntk") and model.config.use_dynamic_ntk:
            model.config.max_sequence_length = 16384
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        tokenizer.eos_token_id = config.eos_token_id
        tokenizer.bos_token_id = config.bos_token_id
        tokenizer.pad_token_id = generation_config.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen")

class YiAdapter(BaseModelAdapter):
    """The model adapter for Yi/Yi-1.5 models"""

    def match(self, model_path: str):
        return "yi-" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("yi-")

class Llama3Adapter(BaseModelAdapter):
    """The model adapter for Llama-3 (e.g., meta-llama/Meta-Llama-3-8B-Instruct)"""

    def match(self, model_path: str):
        return "llama-3" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-3")

class OpenchatLlama3Adapter(BaseModelAdapter):
    """The model adapter for Llama-3 (e.g., meta-llama/Meta-Llama-3-8B-Instruct)"""

    def match(self, model_path: str):
        return "openchat-3.6" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("openchat-3.6-llama3")

class BGEAdapter(BaseModelAdapter):
    """The model adapter for BGE (e.g., BAAI/bge-large-en-v1.5)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "bge" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModel.from_pretrained(
            model_path,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        if hasattr(model.config, "max_position_embeddings") and hasattr(
            tokenizer, "model_max_length"
        ):
            model.config.max_sequence_length = min(
                model.config.max_position_embeddings, tokenizer.model_max_length
            )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


class E5Adapter(BaseModelAdapter):
    """The model adapter for E5 (e.g., intfloat/e5-large-v2)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "e5-" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModel.from_pretrained(
            model_path,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        if hasattr(model.config, "max_position_embeddings") and hasattr(
            tokenizer, "model_max_length"
        ):
            model.config.max_sequence_length = min(
                model.config.max_position_embeddings, tokenizer.model_max_length
            )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


class AquilaChatAdapter(BaseModelAdapter):
    """The model adapter for BAAI/AquilaChat-7B"""

    def match(self, model_path: str):
        return "aquila" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("aquila-chat")


class Lamma2ChineseAdapter(BaseModelAdapter):
    """The model adapter for FlagAlpha/LLama2-Chinese sft"""

    def match(self, model_path: str):
        return "llama2-chinese" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama2-chinese")


class VigogneAdapter(BaseModelAdapter):
    """The model adapter for vigogne (e.g., bofenghuang/vigogne-2-7b-chat)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return bool(re.search(r"vigogne|vigostral", model_path, re.I))

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=True,
            revision=revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        ).eval()
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "chat" in model_path.lower():
            if "vigostral" in model_path.lower():
                return get_conv_template("vigogne_chat_v3")
            return get_conv_template("vigogne_chat_v2")
        return get_conv_template("vigogne_instruct")


class OpenLLaMaOpenInstructAdapter(BaseModelAdapter):
    """The model adapter for OpenLLaMa-Open-Instruct (e.g., VMware/open-llama-7b-open-instruct)"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return (
            "open-llama" in model_path.lower() and "open-instruct" in model_path.lower()
        )

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=True,
            revision=revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        ).eval()
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("alpaca")


class CodeLlamaAdapter(BaseModelAdapter):
    """The model adapter for CodeLlama (e.g., codellama/CodeLlama-34b-hf)"""

    def match(self, model_path: str):
        return "codellama" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")


class PhindCodeLlamaAdapter(CodeLlamaAdapter):
    """The model adapter for Phind-CodeLlama (e.g., Phind/Phind-CodeLlama-34B-v2)"""

    def match(self, model_path: str):
        return "phind" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phind")

class Phi3Adapter(CodeLlamaAdapter):
    """The model adapter for Phind-CodeLlama (e.g., Phind/Phind-CodeLlama-34B-v2)"""

    def match(self, model_path: str):
        return "phi-3" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phi-3")


class Llama2ChangAdapter(Llama2Adapter):
    """The model adapter for Llama2-ko-chang (e.g., lcw99/llama2-ko-chang-instruct-chat)"""

    def match(self, model_path: str):
        return "llama2-ko-chang" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("polyglot_changgpt")


class ZephyrAdapter(BaseModelAdapter):
    """The model adapter for Zephyr (e.g. HuggingFaceH4/zephyr-7b-alpha)"""

    def match(self, model_path: str):
        return "zephyr" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("zephyr")


class XwinLMAdapter(BaseModelAdapter):
    """The model adapter for Xwin-LM V0.1 and V0.2 series of models(e.g., Xwin-LM/Xwin-LM-70B-V0.1)"""

    # use_fast_tokenizer = False

    def match(self, model_path: str):
        return "xwin-lm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("vicuna_v1.1")

class GemmaAdapter(BaseModelAdapter):
    """The model adapter for google/gemma"""

    def match(self, model_path: str):
        return "gemma" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("gemma")

# Note: the registration order matters.
# The one registered earlier has a higher matching priority.
register_model_adapter(PeftModelAdapter)
register_model_adapter(VicunaAdapter)
register_model_adapter(OpenChat35Adapter)
register_model_adapter(PretrainAdapter)
register_model_adapter(AiroborosAdapter)
register_model_adapter(GoogleT5Adapter)
register_model_adapter(KoalaAdapter)
register_model_adapter(AlpacaAdapter)
register_model_adapter(ChatGLMAdapter)
register_model_adapter(DollyV2Adapter)
register_model_adapter(OasstPythiaAdapter)
register_model_adapter(OasstLLaMAAdapter)
register_model_adapter(StableLMAdapter)
register_model_adapter(BaizeAdapter)
register_model_adapter(OpenBuddyAdapter)
register_model_adapter(PhoenixAdapter)
register_model_adapter(BardAdapter)
register_model_adapter(PaLM2Adapter)
register_model_adapter(ChatGPTAdapter)
register_model_adapter(ClaudeAdapter)
register_model_adapter(MPTAdapter)
register_model_adapter(BiLLaAdapter)
register_model_adapter(RedPajamaINCITEAdapter)
register_model_adapter(H2OGPTAdapter)
register_model_adapter(RobinAdapter)
register_model_adapter(SnoozyAdapter)
register_model_adapter(WizardLMAdapter)
register_model_adapter(ManticoreAdapter)
register_model_adapter(GuanacoAdapter)
register_model_adapter(CamelAdapter)
register_model_adapter(ChangGPTAdapter)
register_model_adapter(TuluAdapter)
register_model_adapter(FalconChatAdapter)
register_model_adapter(FalconAdapter)
register_model_adapter(TigerBotAdapter)
register_model_adapter(BaichuanAdapter)
register_model_adapter(XGenAdapter)
register_model_adapter(NousHermesAdapter)
register_model_adapter(PythiaAdapter)
register_model_adapter(InternLMChatAdapter)
register_model_adapter(StarChatAdapter)
register_model_adapter(Llama2Adapter)
register_model_adapter(MistralAdapter)
register_model_adapter(CuteGPTAdapter)
register_model_adapter(OpenOrcaAdapter)
register_model_adapter(WizardCoderAdapter)
register_model_adapter(QwenChatAdapter)
register_model_adapter(Llama3Adapter)
register_model_adapter(OpenchatLlama3Adapter)
register_model_adapter(AquilaChatAdapter)
register_model_adapter(BGEAdapter)
register_model_adapter(E5Adapter)
register_model_adapter(Lamma2ChineseAdapter)
register_model_adapter(VigogneAdapter)
register_model_adapter(OpenLLaMaOpenInstructAdapter)
register_model_adapter(ReaLMAdapter)
register_model_adapter(PhindCodeLlamaAdapter)
register_model_adapter(CodeLlamaAdapter)
register_model_adapter(Llama2ChangAdapter)
register_model_adapter(ZephyrAdapter)
register_model_adapter(XwinLMAdapter)
register_model_adapter(Phi3Adapter)
register_model_adapter(QwenAdapter)
register_model_adapter(YiAdapter)
register_model_adapter(GemmaAdapter)

# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)