# Requirements and Installation

* PyTorch version >= 1.10.0
* Python version == 3.8
* **Install fairseq** and develop locally:

```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

# Neural Machine Translation

* We require a few additional Python dependencies for preprocessing:

```bash
pip install fastBPE sacremoses subword_nmt
pip install tensorboardX
```

## Training a new model
### IWSLT'14 German to English (Transformer)

First download and preprocess the data:

```bash
# Download and prepare the data
cd examples/translation
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

Replace/add required files:

* trainer.py --> fairseq/trainer.py 
* fairseq_task.py --> fairseq/tasks/fairseq_task.py
* adaact_one.py --> fairseq/optim/adaact_one.py
* opt_utils2.py / torch_utils.py / utils_.py --> fairseq/optim/utils/*

Next we'll train a Transformer translation model over this data:

```bash
# Adam:
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
    
# R1:
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adaactr1 --momentum 0.9 --stat_decay 0.95 --damping 0.001 \
    --weight-decay 0.01 --update 1 --sgd_momentum_type heavyball \
    --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --dropout 0.3 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```