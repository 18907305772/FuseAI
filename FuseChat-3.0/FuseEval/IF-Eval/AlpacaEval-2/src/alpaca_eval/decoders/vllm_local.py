import logging
from typing import Sequence

import numpy as np
from vllm import LLM, SamplingParams

from .. import utils

__all__ = ["vllm_local_completions"]

llm = None
llmModelName = None


def vllm_local_completions(
    prompts: Sequence[str],
    model_name: str,
    max_new_tokens: int,
    do_sample: bool = False,
    batch_size: int = 1,
    model_kwargs=None,
    **kwargs,
) -> dict[str, list]:
    """Decode locally using vllm transformers pipeline.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model (repo on hugging face hub)  to use for decoding.

    do_sample : bool, optional
        Whether to use sampling for decoding.

    batch_size : int, optional
        Batch size to use for decoding. This currently does not work well with to_bettertransformer.

    model_kwargs : dict, optional
        Additional kwargs to pass to from_pretrained.

    kwargs :
        Additional kwargs to SamplingParams
    """
    global llm, llmModelName
    tp = 1
    if "tp" in model_kwargs:
        tp = model_kwargs["tp"]
    tokenizer_mode = model_kwargs.get("tokenizer_mode", "auto")
    trust_remote_code = model_kwargs.get("trust_remote_code", False)
    model_root=model_kwargs["model_root_path"]
    model_name=model_root+model_name
    print(f"testing model {model_name}\n")
    if llm is None:
        logging.info("vllm: loading model: %s, tp=%d, trust_remote_code=%d", model_name, tp, trust_remote_code)
        llm = LLM(
            model=model_name,
            tokenizer=model_name,
            tokenizer_mode=tokenizer_mode,
            tensor_parallel_size=tp,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            max_model_len=4096
        )
        llmModelName = model_name
    if model_name != llmModelName:
        assert False, "vllm_local_completions can only be used with a single model"

    sampling_params = SamplingParams(max_tokens=max_new_tokens)
    if "n" in kwargs:
        sampling_params.n = kwargs["n"]
    if "best_of" in kwargs:
        sampling_params.best_of = kwargs["best_of"]
    if "temperature" in kwargs:
        sampling_params.temperature = kwargs["temperature"]
    if "top_p" in kwargs:
        sampling_params.top_p = kwargs["top_p"]
    if "top_k" in kwargs:
        sampling_params.top_k = kwargs["top_k"]
    if "stop_token_ids" in kwargs:
        sampling_params.stop_token_ids = kwargs["stop_token_ids"]
    if "stop" in kwargs:
        sampling_params.stop = kwargs["stop"]
    if do_sample:
        sampling_params.use_beam_search = True
    if "length_penalty" in kwargs:
        sampling_params.length_penalty = kwargs["length_penalty"]
    if "early_stopping" in kwargs:
        sampling_params.early_stopping = kwargs["early_stopping"]
    print(sampling_params)
    completions = []
    with utils.Timer() as t:
        outputs = llm.generate(prompts, sampling_params)
        print("example prompt_token_ids:")
        print(outputs[0].prompt_token_ids)
        for j in range(0, len(prompts)):
            if sampling_params.n > 1:
                completion_list=[]
                for k in range(sampling_params.n):
                    completion_list.append(outputs[j].outputs[k].text)
                completions.append(completion_list)
            else:
                completions.append(outputs[j].outputs[0].text)
    price = [np.nan] * len(completions)
    avg_time = [t.duration / len(prompts)] * len(completions)
    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)
