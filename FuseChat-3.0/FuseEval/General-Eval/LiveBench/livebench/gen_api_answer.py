"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures
import glob
import openai
import shortuuid
import tqdm

from livebench.common import (
    reorg_answer_file,
    get_categories_tasks,
    get_hf_dataset,
    get_tasks_from_hf_category,
    load_questions,
    load_questions_jsonl,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_palm,
    chat_completion_google_generativeai,
    chat_completion_vertex,
    chat_completion_mistral,
    chat_completion_cohere,
    chat_completion_deepseek,
    chat_completion_together,
    chat_completion_nvidia,
    chat_completion_inference_openai,
    chat_completion_openrouter,
    LIVE_BENCH_DATA_SUPER_PATH,
)
from livebench.model.model_adapter import (
    get_conversation_template, 
    ANTHROPIC_MODEL_LIST,
    GOOGLE_GENERATIVEAI_MODEL_LIST, 
    VERTEX_MODEL_LIST,
    MISTRAL_MODEL_LIST, 
    COHERE_MODEL_LIST, 
    DEEPSEEK_MODEL_LIST,
    TOGETHER_MODEL_LIST,
    NVIDIA_MODEL_LIST,
    INFERENCE_OPENAI_MODEL_LIST,
    OPENROUTER_MODEL_LIST,
)

def get_answer(
    question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str, api_dict: dict=None
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    else:
        temperature = 0.0

    choices = []
    chat_state = None  # for palm-2 model
    for i in range(num_choices):
        conv = get_conversation_template(model)

        turns = []
        for j in range(len(question["turns"])):
            conv.append_message(conv.roles[0], question["turns"][j])
            conv.append_message(conv.roles[1], None)

            if api_dict is not None:
                output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict=api_dict)
            elif model in ANTHROPIC_MODEL_LIST:
                output = chat_completion_anthropic(model, conv, temperature, max_tokens)
            elif model == "palm-2-chat-bison-001":
                chat_state, output = chat_completion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            elif model in VERTEX_MODEL_LIST:
                output = chat_completion_vertex(model, conv, temperature, max_tokens)
            elif model in GOOGLE_GENERATIVEAI_MODEL_LIST:
                output = chat_completion_google_generativeai(model, conv, temperature, max_tokens)
            elif model in MISTRAL_MODEL_LIST:
                output = chat_completion_mistral(model, conv, temperature, max_tokens)
            elif model in COHERE_MODEL_LIST:
                output = chat_completion_cohere(model, conv, temperature, max_tokens)
            elif model in DEEPSEEK_MODEL_LIST:
                output = chat_completion_deepseek(model, conv, temperature, max_tokens)
            elif model in TOGETHER_MODEL_LIST:
                output = chat_completion_together(model, conv, temperature, max_tokens)
            elif model in NVIDIA_MODEL_LIST:
                output = chat_completion_nvidia(model, conv, temperature, max_tokens)
            elif model in INFERENCE_OPENAI_MODEL_LIST:
                output = chat_completion_inference_openai(model, conv, temperature, max_tokens)
            elif model in OPENROUTER_MODEL_LIST:
                output = chat_completion_openrouter(model, conv, temperature, max_tokens)
            else:
                output = chat_completion_openai(model, conv, temperature, max_tokens)

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


def run_questions(parallel, questions, model, num_choices, max_tokens, answer_file, api_dict):
    if parallel == 1:
        for question in tqdm.tqdm(questions):
            get_answer(
                question,
                model,
                num_choices,
                max_tokens,
                answer_file,
                api_dict=api_dict,
            )
        if len(questions) > 0:
            reorg_answer_file(answer_file)
    else:

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            for question in questions:
                future = executor.submit(
                    get_answer,
                    question,
                    model,
                    num_choices,
                    max_tokens,
                    answer_file,
                    api_dict=api_dict,
                )
                futures.append(future)      

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()
        if len(questions) > 0:
            reorg_answer_file(answer_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="live_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="If provided, will be used as the base of an openai API request, along with the environment variable LIVEBENCH_API_KEY",
    )
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="The maximum number of new generated tokens.",
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
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
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

    if args.api_base is not None:
        api_key = os.environ.get("LIVEBENCH_API_KEY", "EMPTY")

        api_dict = {
            "api_key": api_key,
            "api_base": args.api_base,
        }
    else:
        api_dict = None

    if args.question_source == "huggingface":
        categories, tasks = get_categories_tasks(args.bench_name)

        for category_name, task_names in tasks.items():
            for task_name in task_names:
                questions = load_questions(categories[category_name], release_set, task_name, args.question_begin, args.question_end)

                questions = [
                    q for q in questions if q['livebench_removal_date'] == "" or q['livebench_removal_date'] > args.livebench_release_option
                ]

                task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                answer_file = f"data/{task_full_name}/model_answer/{args.model}.jsonl"

                print(f"Questions from {task_full_name}")
                print(f"Output to {answer_file}")

                run_questions(
                    parallel=args.parallel,
                    questions=questions, 
                    model=args.model, 
                    num_choices=args.num_choices,
                    max_tokens=args.max_tokens, 
                    answer_file=answer_file, 
                    api_dict=api_dict
                )

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

            questions = [
                q for q in questions if q['livebench_removal_date'] == "" or q['livebench_removal_date'] > args.livebench_release_option
            ]

            bench_name = os.path.dirname(question_file).replace("data/","")
            answer_file = f"data/{bench_name}/model_answer/{args.model}.jsonl"

            print(f"Questions from {question_file}")
            print(f"Output to {answer_file}")

            run_questions(
                parallel=args.parallel,
                questions=questions, 
                model=args.model, 
                num_choices=args.num_choices,
                max_tokens=args.max_tokens, 
                answer_file=answer_file, 
                api_dict=api_dict
            )

    else:
        raise ValueError(f"Bad question source {args.question_source}.")
    