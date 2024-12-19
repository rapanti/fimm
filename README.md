# FIMM
Finetune PyTorch Image Models with TIMM

This project provides a simple way to finetune PyTorch Image Models with TIMM.

## Installation
To install FIMM (`fimm`), you can simply use `pip`:

```bash
pip install fimm
```

### Install from source
To install from source, you can clone this repo and install with `pip`:

```bash
git clone https://github.com/rapanti/fimm
pip install -e fimm  # -e for editable mode
```

## Usage
To use FIMM, you can simply run the follwing command to train or finetune a model:
```bash
train --data-dir /path/to/dataset --model resnet50 --experiment resnet50 # this trains a resnet50 model from scratch
train --data-dir /path/to/dataset --model resnet50 --experiment resnet50 --pretrained  # this finetunes a resnet50 model
```

To validate the performance of a model, you can simply run the following command:
```bash
validate --data-dir /path/to/eval/dataset --model resnet50 --checkpoint output/train/resnet50/model_best.pth.tar # this tests the resnet50 model
```

