"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import glob

import shortuuid
import torch
from tqdm import tqdm

from livebench.common import (
    reorg_answer_file,
    get_categories_tasks,
    get_hf_dataset,
    get_tasks_from_hf_category,
    load_questions,
    load_questions_jsonl,
    LIVE_BENCH_DATA_SUPER_PATH,
)
from livebench.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
from vllm import LLM, SamplingParams
from collections import defaultdict

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def run_eval(
    model_path,
    model_id,
    tp_size,
    questions,
    num_gpus_per_model,
    num_gpus_total,
):
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
                model_id,
                tp_size,
                questions[i : i + chunk_size]
            )
        )

    if use_ray:
        ray.get(ans_handles)


def get_model_answers(
    model_path,
    model_id,
    tensor_parallel_size,
    questions
):

    # grouped_questions = defaultdict(list)
    # for question, answer_file in questions:
    #     grouped_questions[answer_file].append(question)
    # grouped_questions = dict(grouped_questions)
    # for answer_file, questions in grouped_questions.items():
    #     print(f"Answer File: {answer_file}. Test cases: {len(questions)}")

    llm = LLM(model=model_path,
              tensor_parallel_size=tensor_parallel_size,
              gpu_memory_utilization=0.95,
              enforce_eager=True,
              # disable_custom_all_reduce=True,
              max_model_len=4096)
    tokenizer = llm.get_tokenizer()

    # model_id=model_path.split("/")[-1]

    if "mistral" in model_path.lower() or "gemma" in model_path.lower():
        filter_bos = True
        bos_token=tokenizer.bos_token
    else:
        filter_bos = False
        bos_token = None

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=4096,
                                     )

    # for answer_file, cur_questions in tqdm(grouped_questions.items()):
    #     conversations=[]
    #     for question in cur_questions:
    #         cons = tokenizer.apply_chat_template([{'role': 'user', 'content': question["turns"][0]}], tokenize=False, add_generation_prompt=True)
    #         if filter_bos and conversations.startswith(bos_token):
    #             cons = cons[len(bos_token):]
    #         conversations.append(cons)
    #     outputs = llm.generate(conversations, sampling_params)
    #     converted_ans=[]
    #     for question,output in zip(cur_questions, outputs):
    #         ans_json = {
    #             "question_id": question["question_id"],
    #             "answer_id": shortuuid.uuid(),
    #             "model_id": model_id,
    #             "choices": [{"index": 0, "turns": [output.outputs[0].text]}],
    #             "tstamp": time.time(),
    #         }
    #         converted_ans.append(ans_json)
    #
    #     os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    #     with open(os.path.expanduser(answer_file), "a", encoding='utf-8') as fout:
    #         for ans_json in converted_ans:
    #             fout.write(json.dumps(ans_json,ensure_ascii=False) + "\n")
    conversations = []
    for question,ans_file in questions:
        cons = tokenizer.apply_chat_template([{'role': 'user', 'content': question["turns"][0]}], tokenize=False,
                                             add_generation_prompt=True)
        if filter_bos and cons.startswith(bos_token):
            cons = cons[len(bos_token):]
        conversations.append(cons)
    outputs = llm.generate(conversations, sampling_params)

    for question_mes, output in zip(questions, outputs):
        question, save_file=question_mes
        ans_json = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": [{"index": 0, "turns": [output.outputs[0].text]}],
            "tstamp": time.time(),
        }
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(os.path.expanduser(save_file), "a", encoding='utf-8') as fout:
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="live_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=4096,
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
        "--question-source", type=str, default="huggingface", help="The source of the questions. 'huggingface' will draw questions from huggingface. 'jsonl' will use local jsonl files to permit tweaking or writing custom questions."
    )
    parser.add_argument(
        "--livebench-release-option", type=str, default='2024-08-31', help="Livebench release to use. Provide a single date option, current options are {'2024-08-31' (august update), '2024-07-26' (july update), '2024-06-24' (original release)}. Will handle excluding deprecated questions for selected release."
    )
    args = parser.parse_args()

    valid_livebench_releases = set(['2024-07-26', '2024-06-24', '2024-08-31'])

    if args.livebench_release_option not in valid_livebench_releases:
        raise ValueError(f"Bad release {args.livebench_release_option}.")
        
    release_set = set([
        r for r in valid_livebench_releases if r <= args.livebench_release_option
    ])

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    questions_all = []
    answer_files  = []

    if args.question_source == "huggingface":
        categories, tasks = get_categories_tasks(args.bench_name)

        for category_name, task_names in tasks.items():
            for task_name in task_names:
                questions = load_questions(categories[category_name], release_set, task_name, args.question_begin, args.question_end)

                task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                answer_file = f"data/{task_full_name}/model_answer/{args.model_id}.jsonl"

                questions_all.extend(
                    [
                        (q, answer_file)
                        for q in questions
                    ]
                )

                answer_files.append(answer_file)

    elif args.question_source == "jsonl":
        list_of_question_files = []
        original_question_file = f"data/{args.bench_name}/question.jsonl"
        if os.path.exists(original_question_file):
            list_of_question_files = [original_question_file]
        else:
            list_of_question_files = glob.glob(f"data/{args.bench_name}/**/question.jsonl", recursive=True)

        for question_file in list_of_question_files:
            print(question_file)
            questions = load_questions_jsonl(question_file, release_set, args.question_begin, args.question_end)

            bench_name = os.path.dirname(question_file).replace("data/","")
            answer_file = f"data/{bench_name}/model_answer/{args.model_id}.jsonl"

            questions_all.extend(
                [
                    (q, answer_file)
                    for q in questions
                ]
            )

            if len(questions) > 0:
                answer_files.append(answer_file)

    else:
        raise ValueError(f"Bad question source {args.question_source}.")

    questions_all = [
        q for q in questions_all if q[0]['livebench_removal_date'] == "" or q[0]['livebench_removal_date'] > args.livebench_release_option
    ]

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        tp_size=args.num_gpus_per_model,
        questions=questions_all,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
    )

    for answer_file in answer_files:
        reorg_answer_file(answer_file)
