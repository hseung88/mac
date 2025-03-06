## MeZO-A3dam: Memory-Efficient Zeroth-Order Adam with Adaptivity Adjustmnets for Fine-tuning LLMs


This repository implements the MeZO-A3dam: Memory-Efficient Zeroth-Order Adam with Adaptivity Adjustments for Fine-tuning LLMs for fine-tuning pre-trained huggingface LMs. We also include MeZO and MeZO-SVRG as baselines. This repository is written in PyTorch and PyTorch-Lightning.

## Installation

To install the relevant python environment use the command

```bash
  conda create --name zo-a3dam python=3.10
  conda activate zo-a3dam
  python -m pip install -r requirements.txt
```
    
## File Overview

The script supports the following models:
1. 'distilbert-base-cased'
2. 'roberta-large'
3. 'gpt2-xl'
4. 'facebook/opt-2.7b'
5. 'facebook/opt-6.7b'

The script supports the following GLUE and SuperGLUE tasks:
1. MNLI
2. QNLI
3. SST-2
4. CoLA
5. RTE
6. BoolQ
7. WiC

Supported fine-tuning alrogithms are {'ZO', 'ZOSVRG', 'ZOAdam'}. 'ZO' for MeZO, 'ZOSVRG' for MeZO-SVRG, and 'ZOAdam' for MeZO-Adam and MeZO-A3dam (with passing adaptivity argument).

## Code Execution
```bash run.sh <CUDA_VISIBLE_DEVICE> <MODEL_NAME> <ALGORITHM> <TASK> <ADAPTIVITY>```

For example,
```bash run.sh 0 distilbert ZOAdam sst2 1e-03```

## Acknowledgements

Note that this repository has been build upon the MeZO-SVRG's official implementation (https://github.com/amazon-science/mezo_svrg).