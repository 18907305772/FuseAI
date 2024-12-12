import os
from abc import ABC, abstractmethod
from typing import List

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./hf_home")

import torch
from stop_sequencer import StopSequencer
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<eos>",
    "<end_of_turn>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#"
]

# prompt from evalplus repo
instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

def make_raw_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
    return task_prompt


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        direct_completion: bool = True,
        dtype: str = "bfloat16",  # default
        trust_remote_code: bool = False,
        dataset: str = None,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.direct_completion = direct_completion
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        if direct_completion:
            if dataset.lower() == "humaneval":
                self.eos += ["\ndef", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
            elif dataset.lower() == "mbpp":
                self.eos += ['\n"""', "\nassert"]

    @abstractmethod
    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class VLlmDecoder(DecoderBase):
    def __init__(self, name: str, tensor_parallel_size = 1, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": tensor_parallel_size, #int(os.getenv("VLLM_N_GPUS", "1"))
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": 0.95
        }
        print(kwargs)
        self.llm = LLM(model=name, max_model_len=4096, **kwargs)

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)
        # print(prompt)

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs


class VLlmAWQDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", "1")),
            "dtype": torch.float16,
            "trust_remote_code": self.trust_remote_code,
            "quantization": "AWQ",
        }

        self.llm = LLM(model=name, max_model_len=2048, **kwargs)

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs


class AWQChatML(VLlmAWQDecoder):
    def __init__(self, name: str, tensor_parallel_size, **kwargs) -> None:
        kwargs["direct_completion"] = False
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        input = f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?
```python
{prompt}
```
<|im_end|>
<|im_start|>assistant
```python
"""
        return VLlmDecoder.codegen(self, input, do_sample, num_samples)

class Gemma2Chat(VLlmDecoder):
    def __init__(self, name: str, tensor_parallel_size, **kwargs) -> None:
        kwargs["direct_completion"] = False
        super().__init__(name, tensor_parallel_size, **kwargs)
        self.eos += ["\n```"]
        self.tokenizer = self.llm.get_tokenizer()

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        input = make_raw_chat_prompt(prompt, instruction_prefix, response_prefix, self.tokenizer)
        # deal with double bos
        if input.startswith(self.tokenizer.bos_token):
            input = input[len(self.tokenizer.bos_token):]
        return VLlmDecoder.codegen(self, input, do_sample, num_samples)


class Llama3Chat(VLlmDecoder):
    def __init__(self, name: str, tensor_parallel_size, **kwargs) -> None:
        kwargs["direct_completion"] = False
        super().__init__(name, tensor_parallel_size, **kwargs)
        self.eos += ["\n```"]
        self.tokenizer=self.llm.get_tokenizer()

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        input = make_raw_chat_prompt(prompt,instruction_prefix,response_prefix,self.tokenizer)

        return VLlmDecoder.codegen(self, input, do_sample, num_samples)



class ChatML(VLlmDecoder):
    def __init__(self, name: str, tensor_parallel_size, **kwargs) -> None:
        kwargs["direct_completion"] = False
        super().__init__(name, tensor_parallel_size, **kwargs)
        self.eos += ["\n```"]

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        input = f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?
```python
{prompt}
```
<|im_end|>
<|im_start|>assistant
```python
"""
        return VLlmDecoder.codegen(self, input, do_sample, num_samples)


def make_model(
    model_type: str,
    model_size: str,
    model_path: str,
    batch_size: int = 1,
    temperature: float = 0.8,
    dataset: str = None,
    tensor_parallel_size = 1
):
    if model_type == "codeqwen" or model_type == "qwen2":
        if "chat" in model_size.lower():
            if "awq" in model_size.lower():
                return AWQChatML(
                    batch_size=batch_size,
                    name=model_path,
                    temperature=temperature,
                    max_new_tokens=2048,
                    tensor_parallel_size = tensor_parallel_size
                )
            else:
                return ChatML(
                    batch_size=batch_size,
                    name=model_path,
                    temperature=temperature,
                    max_new_tokens=2048,
                    tensor_parallel_size = tensor_parallel_size
                )
        else:
            return VLlmDecoder(
                batch_size=batch_size,
                name=model_path,
                temperature=temperature,
                dataset=dataset,
                tensor_parallel_size = tensor_parallel_size
            )
    elif model_type=="llama3":
        if "chat" in model_size.lower():
            return Llama3Chat(
                batch_size=batch_size,
                name=model_path,
                temperature=temperature,
                max_new_tokens=2048,
                tensor_parallel_size = tensor_parallel_size
            )
        else:
            return VLlmDecoder(
                batch_size=batch_size,
                name=model_path,
                temperature=temperature,
                dataset=dataset,
                tensor_parallel_size = tensor_parallel_size
            )
    elif model_type=="gemma2":
        if "chat" in model_size.lower():
            return Gemma2Chat(
                batch_size=batch_size,
                name=model_path,
                temperature=temperature,
                max_new_tokens=2048,
                tensor_parallel_size = tensor_parallel_size
            )
        else:
            return VLlmDecoder(
                batch_size=batch_size,
                name=model_path,
                temperature=temperature,
                dataset=dataset,
                tensor_parallel_size = tensor_parallel_size
            )
    else:
        raise ValueError(f"Invalid model name: {model_type}@{model_size}")
