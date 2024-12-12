import argparse
import jsonlines
import os
import json
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch

temp_config = {
    "专业能力": 0.1,
    "数学计算": 0.1,
    "基本任务": 0.1,
    "逻辑推理": 0.1,
    "中文理解": 0.1,
    "文本写作": 0.7,
    "角色扮演": 0.7,
    "综合问答": 0.7
}

if __name__ == '__main__':
    """
    singleround inference 
    input question doc format:
        question_doc = {
            "question_id": int,
            "category": str,
            "subcategory": str,
            "question": str,
        }
    output answer file format
         {
            "question_id": int,
            "category": str,
            "subcategory": str,
            "model_id": str,
            "question": str,
            "answer": str
        }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Evaluated Model Name")
    parser.add_argument("--question-file", type=str, default="data/data_v1.1_release.jsonl")
    parser.add_argument("--save-dir", type=str,default="data/model_answer")
    parser.add_argument("--first-n", type=int, help="Debug Option")
    args = parser.parse_args()

    model_path=args.model
    model_name=model_path.split("/")[-1]
    answer_file=f"{args.save_dir}/{model_name}.jsonl"

    print(">>> inference model: ", model_path)

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    print(f">>> Output to {answer_file}")

    ## load questions
    docs = []
    with jsonlines.open(args.question_file, "r") as f:
        for doc in f:
            docs.append(doc)
        f.close()

    if args.first_n:
        docs = docs[: args.first_n]
    print(f">>> running {len(docs)} docs")

    category_data = {category: [] for category in temp_config}
    for data in docs:
        category = data.get("category")
        # 如果category在权重字典中，则将其添加到对应的列表中
        if category in category_data:
            category_data[category].append(data)

    llm = LLM(model=model_path,
              tensor_parallel_size=torch.cuda.device_count(),
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

    for category in category_data:
        cur_datas=category_data[category]
        temperature=temp_config[category]

        sampling_params = SamplingParams(temperature=temperature,
                                         max_tokens=4096,
                                         )

        conversations=[]
        for data in cur_datas:
            conversation = tokenizer.apply_chat_template([{'role': 'user', 'content': data['question']}],
                                                          tokenize=False, add_generation_prompt=True)
            if filter_bos and conversation.startswith(bos_token):
                conversation = conversation[len(bos_token):]
            conversations.append(conversation)
        outputs = llm.generate(conversations, sampling_params)

        for data, output in zip(cur_datas,outputs):
            data["model_id"] = model_id
            data["answer"] = output.outputs[0].text
            data["answer_id"] = str(data["question_id"]) + "_" + model_id
            with open(os.path.expanduser(answer_file), "a",encoding='utf-8') as fout:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

