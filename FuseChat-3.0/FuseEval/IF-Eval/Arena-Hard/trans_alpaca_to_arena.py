import argparse
import pandas as pd
import time
import shortuuid
import json
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir", type=str, default="outputs"
)
parser.add_argument(
    "--model-name", type=str, default="Meta-Llama-3.1-8B-Instruct"
)

args = parser.parse_args()

file_name=args.model_name

questions = []
with open("data/question.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        questions.append(data["question_id"])

files = f"{args.input_dir}/model_outputs.json"
ans_list = []


with open(files, 'r', encoding='utf-8') as f:
    df = pd.read_json(files)
    outputs=df["output"]
    question_ids = df["dataset"].tolist()


candidate_list=[]
inputs=[]

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

token_count=0

ans_list=[]
for i in range(len(questions)):
    output_idx=question_ids.index(questions[i])
    choices=[]
    turns=[]
    token_len=len(encoding.encode(outputs[output_idx],disallowed_special=()))
    turns.append({"content": outputs[output_idx], "token_len": token_len})
    token_count+=token_len
    choices.append({"index": 0, "turns": turns})

    ans_json = {
        "question_id": questions[i],
        "answer_id": shortuuid.uuid(),
        "model_id": file_name,
        "choices": choices,
        "tstamp": time.time(),
    }
    ans_list.append(ans_json)


with open(f"{args.input_dir}/{args.model_name}.jsonl", "w", encoding='utf-8') as fout:
    for item in ans_list:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"average token len:{token_count/len(questions)}")