---
language:
- en
license: mit
task_categories:
- question-answering
pretty_name: MMLU loader with no auxiliary train set
dataset_info:
  config_name: all
  features:
  - name: question
    dtype: string
  - name: subject
    dtype: string
  - name: choices
    sequence: string
  - name: answer
    dtype:
      class_label:
        names:
          '0': A
          '1': B
          '2': C
          '3': D
  splits:
  - name: test
    num_bytes: 6967453
    num_examples: 14042
  - name: validation
    num_bytes: 763484
    num_examples: 1531
  - name: dev
    num_bytes: 125353
    num_examples: 285
  download_size: 3987384
  dataset_size: 7856290
configs:
- config_name: all
  data_files:
  - split: test
    path: all/test-*
  - split: validation
    path: all/validation-*
  - split: dev
    path: all/dev-*
---
This dataset contains a copy of the `cais/mmlu` HF dataset but without the `auxiliary_train` split that takes a long time to generate again each time when loading multiple subsets of the dataset.

Please visit https://huggingface.co/datasets/cais/mmlu for more information on the MMLU dataset.