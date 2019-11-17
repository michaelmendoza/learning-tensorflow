# Learning Tensorflow
![Supports TFv2.0](https://img.shields.io/badge/Supports-tensorflow%20v2.0-blue.svg)

This tutorial was created for learning tensorflow by example. Currently this repo contains examples for a simple single-layer neural network, a multi-layered perception neural network, and a convolutional neural network. Tensorflow implementations in this repo work with a variety of data sets. Included are simple examples using keras sequential API and more advanced examples using the imperative style of creating networks with model subclassing API.

### Getting Started
Installation requires python 3 with tensorflow. The easiest method to install the requisite libraries is to install the [conda package manager](https://conda.io/miniconda.html). Then run the following command to install necessary libraries and create a virtual environment call `tf`:

```
conda create -n tf python=3.6 numpy matplotlib scikit-image tqdm 
```

If you have a CUDA-enabled GPU install tensorflow-gpu: `pip install tensorflow-gpu`
Otherwise use: `pip install tensorflow`

Activate this virtual environment with `source activate tf` (Mac) or `activate tf` (PC).

### Notes

**Update**: Updated most examples for tensorflow 2.0! Code still using Tensorflow 1.0 will be denoted with 'v1'

If you are using older version of Tensorflow like 1.12, please look [here](https://github.com/michaelmendoza/learning-tensorflow/blob/tf-v1.12/README.md)

# Examples
Examples of tensorflow implementations for Classification, Segmentation, Regression and Modeling Fourier Transform

## Basics

Python basics ([tutorial](notebooks/0a%20-%20Python%20Basics.ipynb))

## Regression
Linear regression from scatch with Tensorflow 2.0 ([tutorial](/notebooks/0b%20-%20Regression%20from%20Scratch%20with%20Tensorflow.ipynb))

Non-linear regression with Tensorflow 2.0 and Keras API ([tutorial](notebooks/0c%20-%20Regression%20with%20Tensorflow%20and%20Keras%20API.ipynb))

## Classification
A variety of neural network implementations for MNIST, and CFAR-10 datasets for classification

### MNIST
Classifying using MNIST Dataset

- Basic Neural Network from scatch with Tensorflow 2.0 ([tutorial](notebooks/1a%20-%20Simple%20Neural%20Network.ipynb))
- Basic Neural Network with simple Keras APIs ([tutorial](notebooks/1b%20-%20Simple%20Neural%20Network%20with%20Keras.ipynb))
- Multi-layer Neural Nework ([simple](examples/mnist/mnist1.py), [advanced](examples/mnist/mnist1_imperative.py)) - A simple (multi-layer preception) network for classifying MNIST dataset 
- Convolutional Neural Nework ([simple](examples/mnist/mnist2.py), [advanced](examples/mnist/mnist2_imperative.py)) - A convolutional network for classifying MNIST dataset 

### CIFAR-10
- Basic Neural Network ([code](examples/cifar/cifar0.py)) - A simple (single layer preception) network for classifying CIFAR-10 dataset 
- Multi-layer Neural Nework ([code](examples/cifar/cifar1.py)) - A simple (multi-layer preception) network for classifying CIFAR-10 dataset 
- Convolutional Neural Nework ([code](examples/cifar/cifar2.py)) - A convolutional network for classifying CIFAR-10 dataset
- Convolutional Neural Nework ([code](examples/cifar/cifar3.py)) - A convolutional network (6-conv, 3 max pool, 2 fully-connected layers) with Dropout for classifying CIFAR-10 dataset 
- VGG network ([code](examples/cifar/cifar4.py), [paper](https://arxiv.org/pdf/1409.1556v6.pdf)) - A very deep convolutional network for large-scale image recongition

## Segmentation
Tensorflow implementation for simple color segmentation using a Unet ([tutorial](notebooks/Segmentation.ipynb))

## Modeling Fourier Transform / FFT
Neural network implementation for learning a fourier transform ([tensorflow v1](examples/fft/fft.py))
