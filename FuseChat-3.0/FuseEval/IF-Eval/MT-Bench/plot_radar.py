import json
import pandas as pd
import csv
import argparse
pd.set_option('display.max_colwidth', 300)

MT_BENCH_CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]
VICUNA_BENCH_CATEGORIES = ["Generic", "Knowledge", "Roleplay", "Common-Sense", "Fermi", "Counterfactual", "Coding & Math", "Writing"]

BENCH_NAME = "mt_bench"

CATEGORIES_MAPPING = {"mt_bench": MT_BENCH_CATEGORIES,
                      "vicuna_bench": VICUNA_BENCH_CATEGORIES}

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str,default="gpt-4-0125-preview_judge_single.jsonl")
parser.add_argument("--output-file", type=str, default="output_scores.csv")
args = parser.parse_args()


def get_model_df():
    cnt = 0
    q2result = []
    fin = open(args.input_file, "r")
    for line in fin:
        obj = json.loads(line)
        index = (obj["question_id"]-81) // 10 if BENCH_NAME == "mt_bench" else (obj["question_id"] - 1) // 10
        obj["category"] = CATEGORIES_MAPPING[BENCH_NAME][index]
        q2result.append(obj)
    df = pd.DataFrame(q2result)
    return df


df = get_model_df()
all_models = df["model"].unique()
all_models = [model_name for model_name in all_models if "Meta-Llama-3-8B" not in model_name]
# print(all_models)
scores_all = []
for model in all_models:
    for cat in CATEGORIES_MAPPING[BENCH_NAME]:
        # filter category/model, and score format error (<1% case)
        res = df[(df["category"] == cat) & (df["model"] == model) & (df["score"] >= 0)]

        turn1 = res[res["turn"] == 1]
        turn2 = res[res["turn"] == 2]
        score = res["score"].mean()
        score1 = turn1["score"].mean()
        score2 = turn2["score"].mean()

        scores_all.append({"model": model, "category": cat, "score": score,"turn1":score1,"turn2":score2})


scores_target = scores_all

df_score = pd.DataFrame(scores_target)

print(df_score)

df_score.to_csv(args.output_file, index=False)
# print(df_score["score"].mean(),df_score["turn1"].mean(),df_score["turn2"].mean())