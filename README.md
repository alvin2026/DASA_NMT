# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8


# Getting Started

* Download and preprocess the minimum dataset

``` bash
cd DASA_NMT
bash ./examples/translation/prepare-iwslt14.sh
```


``` bash
python scripts/build_dep_bias.py \
  --bpe-src train.en.bpe \
  --out-dir dep_bias/train \
  --lang en

python scripts/build_dep_bias.py \
  --bpe-src valid.en.bpe \
  --out-dir dep_bias/valid \
  --lang en

python scripts/build_dep_bias.py \
  --bpe-src test.en.bpe \
  --out-dir dep_bias/test \
  --lang en

```

``` bash
  fairseq-preprocess \
  --source-lang en --target-lang de \
  --trainpref train --validpref valid --testpref test \
  --destdir data-bin/mydata \
  --workers 8
```

* Model training

``` bash
  fairseq-train data-bin/mydata \
  --user-dir dasa_fairseq \
  --task translation_dasa \
  --dep-bias-dir dep_bias \
  --arch transformer_dasa_base \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --dropout 0.3 --label-smoothing 0.1 \
  --dasa-mode log --dasa-lambda 0.5 \
  --max-tokens 4096 --fp16 \
  --dasa-mode mul \
  --save-dir ckpt/dasa
```

* Inference
``` bash

  fairseq-generate data-bin/mydata \
  --user-dir dasa_fairseq \
  --task translation_dasa \
  --dep-bias-dir dep_bias \
  --path ckpt/dasa/checkpoint_best.pt \
  --beam 5 --remove-bpe \
  --results-path results/dasa
```
