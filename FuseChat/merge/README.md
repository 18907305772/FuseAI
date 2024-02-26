## compared methods
For `ties`, `slerp`, `da` and `dare` merge methods,

1. install the package [`mergekit`](https://github.com/arcee-ai/mergekit).

2. make the scripts in `merge_configs` available and run `merge_configs/merge.sh`.

## our methods
For the `VaRM` (Variation Ratio Merge) method, 

1. run `VaRM/analysis.sh` to obtain the `analysis.json` of each LLM, which contains the parameters required for merging such as the variation ratio of parameters before and after fine-tuning each target LLM. 

2. run `VaRM/merge.sh` to get the fused LLM.