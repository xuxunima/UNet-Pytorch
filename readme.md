# Pytorch version for UNet
This repository is a simple implementation of the [UNet](https://arxiv.org/abs/1505.04597), and trained with two datasets. One is the
[isbi challenge](http://brainiac2.mit.edu/isbi_challenge/) dataset, and another is Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge).

## Overview
### ISBI
The original data contains 30 512*512 images, and I used 25 for train and just 5 for valdation. After 30 epochs training, the dice is about 0.94. Because 30 images are not enough to train such a neural network, you can implement data augmentation to get more data.
The trained [model](https://github.com/xuxunima/UNet-Pytorch/releases/tag/model2) can be found here.


### Caravan
The caravan dataset contains 5000 images with labels, and I used 0.95 of them to train, and the remained for validation. After 10 epochs training, 
the dice is about 0.9912, also you can try more data augmentation. The trained [model](https://github.com/xuxunima/UNet-Pytorch/releases/tag/model) train be found here.

## Usage
You can easily train the model by `python train_car.py --gpu` or `python train_membrance.py --gpu`

If you want to train with your dataset, just need to write a new `Dataset` like the class in `data_utils/data_generator.py`

For prediction just use `python predict.py --gpu -l model/path` 

For isbi or caravan you just need to change a little in `predict.py`, such as in_channel for the net. And for caravan use the `predict_net()` function and for isbi use the `predict_mem_net()`.

## Note
This repo now can just support one class segmentation, I will update multi class in the future.

## Reference
This repo refers to [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
