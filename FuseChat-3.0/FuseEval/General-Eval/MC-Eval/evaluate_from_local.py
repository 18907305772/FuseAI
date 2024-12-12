import csv
import json
import argparse
import os
import torch
import numpy as np
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForCausalLM
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
from distutils.util import strtobool
import logging
import sys
from datasets import load_dataset


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 8192
max_new_tokens = 2048



def load_mmlu_pro():
    dataset = load_dataset("data/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df

def load_mmlu():
    dataset = load_dataset("data/mmlu_no_train","all")
    dataset = dataset.rename_column("subject", "category")
    dataset = dataset.rename_column("choices", "options")
    # print(dataset)
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df

def load_gpqa_diamond():
    dataset = load_dataset("data/gpqa_formatted/diamond")
    dataset = dataset.rename_column("Question", "question")
    # print(dataset)
    test_df = dataset["train"]
    test_df = test_df.map(lambda example: {"category": "gpqa_diamond"},remove_columns=["Canary String"])

    test_df = preprocess(test_df)
    return test_df, None

def load_gpqa_main():
    dataset = load_dataset("data/gpqa_formatted/main")
    dataset = dataset.rename_column("Question", "question")
    # print(dataset)
    test_df = dataset["train"]
    test_df = test_df.map(lambda example: {"category": "gpqa_main"},remove_columns=["Canary String"])

    test_df = preprocess(test_df)
    return test_df, None

load_function_map = {
    "mmlu": load_mmlu,
    "mmlu_pro": load_mmlu_pro,
    "gpqa_diamond":load_gpqa_diamond,
    "gpqa_main":load_gpqa_main
}

def load_model():
    try:
        llm = LLM(model=args.model,
                  gpu_memory_utilization=float(args.gpu_util),
                  enforce_eager=True,
                  tensor_parallel_size=torch.cuda.device_count(),
                  max_model_len=max_model_length if "gpqa" in args.selected_dataset else 4096 ,
                  trust_remote_code=True)
        # tokenizer=llm.get_tokenizer()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print("vllm unsupported models", e)
        return None, None
    return llm, tokenizer


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    model_name += f"_{input_args.selected_dataset}"
    if input_args.chat_template:
        model_name +="_chat_template"
    if input_args.zero_shot:
        model_name +="_zero_shot"
        scoring_method += "-zero-shot"
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        # refer to zero-shot cot prompt provided in meta-llama/Llama-3.1-8B-Instruct-evals
        prompt+=("- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nUse this step-by-step format:\n"
                "## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\nRegardless of the approach, always conclude with:\n"
                "The best answer is [the_answer_letter].\nwhere the [the_answer_letter] is the correct letter choice.\n\n")
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    # few shot
    if not args.zero_shot and val_df is not None:
        with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
            for line in fi.readlines():
                prompt += line
        subject = curr["category"]
        val_df = select_by_category(val_df, subject)
        val_df = val_df[: k]
        prompt = prompt.replace("{$}", subject) + "\n"
        for example in val_df:
            prompt += format_cot_example(example, including_answer=True)
    # zero-shot
    else:
        prompt="Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n"
        prompt += format_cot_example(curr, including_answer=False)

    return prompt


def check_exist(res, q_id):
    for each in res:
        if q_id == each["question_id"]:
            if "pred" in each:
                # logging.debug("exist, skip it")
                return True
            else:
                logging.debug("no result in exist result error")
                return False
        else:
            continue
    return False


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None



def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if "answer_index" in each:
                mapped_value=each["answer_index"]
            else:
                mapped_value = each["answer"]
            if x == mapped_value:
                corr += 1
            else:
                wrong += 1
        elif ("answer_index" in each and each["pred"] == each["answer"]) or ("answer_index" not in each and each["pred"] == chr(each["answer"]+65)):
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path, exists_result=None):
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                     stop=["Question:"])
    llm = model
    if not exists_result:
        res = []
    else:
        res = exists_result
    print("load exists result length", len(res))
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)

            if args.chat_template:
                convs=[{"content":prompt,"role":"user"}]
                prompt = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
                # For Qwen2.5 series, delete default system prompt
                if "<|im_start|>system" in prompt:
                    prompt = "<|im_start|>user" + prompt.split("<|im_start|>user", 1)[1]
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
            if k < 0:
                break
        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong

def main():
    model,tokenizer = load_model()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)


    load_function = load_function_map.get(args.selected_dataset)
    if load_function:
        full_test_df, full_val_df = load_function()
    else:
        raise ValueError(f"Unknown dataset: {args.selected_dataset}")

    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject) if full_val_df is not None else None
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        if os.path.exists(output_path):
            with open(output_path, "r") as fi:
                exists_result = json.load(fi)
        else:
            exists_result = []
        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df,
                                                test_df, output_path, exists_result)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_dataset", "-sd", type=str, default="mmlu_pro",
                        choices=["mmlu_pro", "mmlu", "gpqa_diamond","gpqa_main"])
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--ngpu", "-g", type=int, default=1)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--chat_template", "-ct", action="store_true")
    parser.add_argument("--zero_shot", "-zs", action="store_true")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()


