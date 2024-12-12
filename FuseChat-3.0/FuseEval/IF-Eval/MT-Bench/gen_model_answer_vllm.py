"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import sys
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams

from common import load_questions, temperature_config

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def run_eval(
    model_path,
    question_file,
    question_begin,
    question_end,
    answer_file,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                num_gpus_per_model,
                questions[i : i + chunk_size],
                answer_file,
                num_choices
            )
        )

    if use_ray:
        ray.get(ans_handles)


def get_model_answers(
    model_path,
    tensor_parallel_size,
    questions,
    answer_file,
    num_choices,
):
    llm = LLM(model=model_path,
              tensor_parallel_size=tensor_parallel_size,
              gpu_memory_utilization=0.95,
              enforce_eager=True,
              disable_custom_all_reduce=True,
              max_model_len=4096)
    tokenizer = llm.get_tokenizer()

    model_id=model_path.split("/")[-1]

    if "mistral" in model_path.lower() or "gemma" in model_path.lower():
        filter_bos = True
        bos_token=tokenizer.bos_token
    else:
        filter_bos = False
        bos_token = None



    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        if temperature < 1e-4:
            do_sample = False
        else:
            do_sample = True

        sampling_params = SamplingParams(temperature=temperature,
                                         max_tokens=4096,
                                         )
        sampling_params.use_beam_search = do_sample



        choices = []
        for i in range(num_choices):
            turns = []
            convs = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                convs.append({'role': 'user', 'content': qs})
                conversations = tokenizer.apply_chat_template(convs, tokenize=False,add_generation_prompt=True)
                if filter_bos and conversations.startswith(bos_token):
                    conversations=conversations[len(bos_token):]
                outputs = llm.generate(conversations, sampling_params)
                output = outputs[0].outputs[0].text
                convs.append({'role': 'assistant', 'content': output})
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a", encoding='utf-8') as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json,ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r", encoding='utf-8') as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w", encoding='utf-8') as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outs.jsonl",
        help="The model file to store.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/model_answer/{args.output_file}.jsonl"

    run_eval(
        model_path=args.model_path,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
    )

    reorg_answer_file(answer_file)
