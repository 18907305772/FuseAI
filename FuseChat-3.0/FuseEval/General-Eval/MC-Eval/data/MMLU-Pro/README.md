---
language:
- en
license: mit
size_categories:
- 10K<n<100K
task_categories:
- question-answering
pretty_name: MMLU-Pro
tags:
- evaluation
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
dataset_info:
  features:
  - name: question_id
    dtype: int64
  - name: question
    dtype: string
  - name: options
    sequence: string
  - name: answer
    dtype: string
  - name: answer_index
    dtype: int64
  - name: cot_content
    dtype: string
  - name: category
    dtype: string
  - name: src
    dtype: string
  splits:
  - name: validation
    num_bytes: 61129
    num_examples: 70
  - name: test
    num_bytes: 8715503
    num_examples: 12032
  download_size: 25356206
  dataset_size: 8776632
---

# MMLU-Pro Dataset (stable version)

MMLU-Pro dataset is a more **robust** and **challenging** massive multi-task understanding dataset tailored to more rigorously benchmark large language models' capabilities. This dataset contains 12K complex questions across various disciplines. 

**We have finished the final round of data cleaning to remove all the potential false negative options! The current is our stable version! Feel free to use it.**

|[**Github**](https://github.com/TIGER-AI-Lab/MMLU-Pro) | [**🏆Leaderboard**](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro) | [**📖Paper**](https://arxiv.org/abs/2406.01574) |


## 1. What's new about MMLU-Pro

Compared to the original MMLU, there are three major differences:

- The original MMLU dataset only contains 4 options, MMLU-Pro increases it to 10 options. The increase in options will make the evaluation more realistic and challenging. The random guessing will lead to a much lower score.
- The original MMLU dataset contains mostly knowledge-driven questions without requiring much reasoning. Therefore, PPL results are normally better than CoT. In our dataset, we increase the problem difficulty and integrate more reasoning-focused problems. In MMLU-Pro, CoT can be 20% higher than PPL. 
- By increasing the distractor numbers, we significantly reduce the probability of correct guess by chance to boost the benchmark’s robustness. Specifically, with 24 different prompt styles tested, the sensitivity of model scores to prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro

![image/png](https://cdn-uploads.huggingface.co/production/uploads/636a35eff8d9af4aea181608/EOSnJQx3o3PTn_vnKWrxQ.png)


## 2. Dataset Summary

- **Questions and Options:** Each question within the dataset typically has **ten** multiple-choice options, except for some that were reduced during the manual review process to remove unreasonable choices. This increase from the original **four** options per question is designed to enhance complexity and robustness, necessitating deeper reasoning to discern the correct answer among a larger pool of potential distractors.
  
- **Sources:** The dataset consolidates questions from several sources:
  - **Original MMLU Questions:** Part of the dataset comes from the original MMLU dataset. We remove the trivial and ambiguous questions.
  - **STEM Website:** Hand-picking high-quality STEM problems from the Internet.
  - **TheoremQA:** High-quality human-annotated questions requiring theorems to solve.
  - **SciBench:** Science questions from college exams.

- **Disciplines Covered by the Newly Added Data:** The subjects that have been enhanced with questions from the STEM Website, TheoremQA, and SciBench are biology, business, chemistry, computer science, economics, engineering, math, physics, and psychology.

| Discipline        | Number of Questions | From Original MMLU | Newly Added |
|:------------------|:--------------------|:-------------------|:------------|
| Math              | 1351                | 846                | 505         |
| Physics           | 1299                | 411                | 888         |
| Chemistry         | 1132                | 178                | 954         |
| Law               | 1101                | 1101               | 0           |
| Engineering       | 969                 | 67                 | 902         |
| Other             | 924                 | 924                | 0           |
| Economics         | 844                 | 444                | 400         |
| Health            | 818                 | 818                | 0           |
| Psychology        | 798                 | 493                | 305         |
| Business          | 789                 | 155                | 634         |
| Biology           | 717                 | 219                | 498         |
| Philosophy        | 499                 | 499                | 0           |
| Computer Science  | 410                 | 274                | 136         |
| History           | 381                 | 381                | 0           |
| **Total**         | **12032**           | 6810	           | 5222        |


![image/png](https://cdn-uploads.huggingface.co/production/uploads/636a35eff8d9af4aea181608/M7mJcKstlVHo6p7P4Cu1j.png)

## 3. Dataset Construction

![image/png](https://cdn-uploads.huggingface.co/production/uploads/636a35eff8d9af4aea181608/kP6hA-T7ldXxOvqTJf42X.png)

- **Initial Filtering:** The construction process began with a comprehensive review of the original MMLU dataset to identify and retain only those questions that meet a higher threshold of difficulty and relevance.
  
- **Question Collection and Integration:** Additional questions were carefully selected from STEM websites, theoremQA, and scibench based on their ability to challenge the analytical capabilities of advanced models. The selection criteria focused on the complexity of the problems and the quality of the questions.
  
- **Option Augmentation:** To further enhance the dataset, we employed GPT-4 to augment the number of choices per question from **four** to **ten**. This process was not merely about adding more options but involved generating plausible distractors that require discriminative reasoning to navigate.
  
- **Expert Review:** Each question and its associated options underwent rigorous scrutiny by a panel of over ten experts. These experts ensured that the questions were not only challenging and comprehensive but also accurate and fair. This step was crucial to maintain the integrity and utility of the dataset as a benchmarking tool.


## 4. Leaderboard

For the updated leaderboard, please refer to https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro. You can submit your evaluation there. Some of the results are run by us while some of the results are obtained by others. Normally we use 5-shot, some models like Gemini use 0-shot.

If you want to reproduce our results, please check out https://github.com/TIGER-AI-Lab/MMLU-Pro for the evaluation scripts. We also cache our model predictions in https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main/eval_results. 


## 5. CoT vs Direct Evaluation

Unlike the original MMLU, which favors PPL evaluation. MMLU-Pro requires CoT reasoning to achieve better results.

|Models                       | Prompting | Overall | Biology | Business | Chemistry | ComputerScience  | Economics | Engineering | Health | History | Law    | Math   | Philosophy | Physics | Psychology | Other  |
|:----------------------------|:----------|:--------|:--------|:---------|:----------|:-----------------|:----------|-------------|:-------|:--------|:-------|:-------|:-----------|:--------|:-----------|:-------|
| GPT-4o                      | CoT       | 0.7255	| 0.8675  | 0.7858	 | 0.7393	 | 0.7829	        | 0.808	    | 0.55	      | 0.7212 | 0.7007	 | 0.5104 |	0.7609 | 0.7014     | 0.7467  |	0.7919	   | 0.7748 |

The non-CoT results are reported in the following table. As you can see, the performance dropped by as much as 19% without chain-of-thought reasoning. It reflects the challenging nature of our dataset.

|Models                       | Prompting | Overall | Biology | Business | Chemistry | ComputerScience  | Economics | Engineering | Health | History | Law   | Math  | Philosophy | Physics | Psychology | Other |
|:----------------------------|:----------|:--------|:--------|:---------|:----------|:-----------------|:-----------|------------|:-------|:--------|:------|:------|:-----------|:--------|:-----------|:------|
| GPT-4o                      | Direct    | 0.5346  | 0.8102  | 0.392    | 0.3447    | 0.5813           | 0.6899    | 0.3981      | 0.6933 | 0.6949  | 0.542 | 0.3427| 0.6614     | 0.3971  | 0.7628     | 0.6391|

## 6. MMLU v.s. MMLU-Pro Results

| Models                        | Original MMLU Score | MMLU Pro Score | Drop       |
|:------------------------------|:--------------------|:---------------|:-----------|
| GPT-4o                        | 0.887               | 0.7255         | 0.1615     |
| Claude-3-Opus                 | 0.868               | 0.6845         | 0.1835     |
| Claude-3-Sonnet               | 0.815               | 0.5511         | 0.2639     |
| Gemini 1.5 Flash              | 0.789               | 0.5912         | 0.1978     |
| Llama-3-70B-Instruct          | 0.820               | 0.5620         | 0.258      |


We can observe that some models like GPT-4o only drop by 16% while some models like Mixtral-8x7B drop more than 30%.


## 7. Dataset Maintenance

There are mistakes in the dataset. If you find anyone, please paste the question_id to the issue page, we will modify it accordingly. Our team is commmitted to maintain this dataset in the long run to ensure its quality!
