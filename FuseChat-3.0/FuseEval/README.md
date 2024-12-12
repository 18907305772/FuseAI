# FuseEval

The evaluation of instruction-tuned models mainly focuses on the model performance of instruction following, natural language understanding, general question answering, reasoning, mathematics, coding, etc. For the evaluation of FuseChat-3.0, we include 14 benchmarks and organize them into four categories:

- **Instruction Following** Tasks: AlpacaEval-2, Arena-Hard, MTbench, AlignBench v1.1 (Chinese).
- **General** Tasks: LiveBench-0831, MMLU-Pro, MMLU-redux, GPQA-Diamond.
- **Mathematics** Tasks: GSM8K, MATH, AMC 23.
- **Coding** Tasks: HumanEval, MBPP, LiveCodeBench 2408-2411.

```bash
released
|-- IF-Eval         # Instruction Following Evaluation
    |-- MT-Bench
    |-- Arena-Hard
    |-- AlignBench
    |-- AlpacaEval-2
|-- General-Eval         # General Evaluation
    |-- MC-Eval        # MMLU-Pro, MMLU-redux, GPQA-Diamond.
    |-- LiveBench
|-- Math-Eval         # Math Evaluation
|-- Code-Eval         # Code Evaluation
    |-- LiveCodeBench
    |-- eval_plus    # HumanEval, MBPP
```