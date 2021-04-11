# Nosie Robust Style Transfer
The code of project, Noise Robust Arbitrary Style Transfer via Knowledge Distillation, for PolyU Subject COMP6706 Advanced Topics in Visual Computing Spring 2021.

## Requirments

The codes are fully tested on Ubuntu 18.04 and Python 3.7. The codes should be able to run on other operation systems,e.g. Windows, and Python 3+ versions.

Please install requirements by `pip install -r requirements.txt`.

## Instruction
You may simly follow the instruction in the original repo:
https://github.com/naoto0804/pytorch-AdaIN.

## Pre-trained Models
Download [decoder.pth](https://drive.google.com/file/d/1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr/view?usp=sharing)/[vgg_normalized.pth](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) and put them under `models/`.

## Datasets for Training

MS-COCO: https://cocodataset.org/#download

Wikiart: https://www.kaggle.com/c/painter-by-numbers

## Use of Tensorboard

To use tensorboard, Please install the package by `pip install tensorboard`.

All traininig logs are saved in `logs/`, To run the tensorboard, enter the following command in the terminal:

`tensorboard --logdir logs`

Then, you can visit it by the following URL:

`http://localhost:6006/`
