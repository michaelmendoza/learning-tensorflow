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

(Note: This code is updated with python 3.6.9 using tensorflow-gpu version 2.0)

# Examples
Examples of tensorflow implementations for Classification, Segmentation, Regression and Modeling Fourier Transform

## Classification
A variety of neural network implementations for MNIST, and CFAR-10 datasets for classification

### MNIST
- Basic Neural Network ([tutorial](notebooks/mnist/0_Single_Layer_Network_Tutorial.ipynb), [simple](examples/mnist/mnist0.py), [advanced](examples/mnist/mnist0_imperative.py)) - A simple (single layer preception) network for classifying MNIST dataset 
- Multi-layer Neural Nework ([simple](examples/mnist/mnist1.py), [advanced](examples/mnist/mnist1_imperative.py)) - A simple (multi-layer preception) network for classifying MNIST dataset 
- Convolutional Neural Nework ([simple](examples/mnist/mnist2.py), [advanced](examples/mnist/mnist2_imperative.py)) - A convolutional network for classifying MNIST dataset 

### CIFAR-10
- Basic Neural Network ([tensorflow](examples/cifar/basic_net.py), [keras](examples/cifar/keras_basic.py)) - A simple (single layer preception) network for classifying CIFAR-10 dataset 
- Multi-layer Neural Nework ([tensorflow](examples/cifar/mlp_net.py), [keras](examples/cifar/keras_mlp.py)) - A simple (multi-layer preception) network for classifying CIFAR-10 dataset 
- Convolutional Neural Nework ([tensorflow](examples/cifar/conv_net.py), [keras](examples/cifar/keras_conv.py)) - A convolutional network for classifying CIFAR-10 dataset
- Convolutional Neural Nework ([keras](examples/cifar/keras_nine_layer_conv.py)) - A convolutional network (6-conv, 3 max pool, 2 fully-connected layers) with Dropout for classifying CIFAR-10 dataset 
- VGG network ([keras](examples/cifar/keras_vgg.py), [paper](https://arxiv.org/pdf/1409.1556v6.pdf)) - A very deep convolutional network for large-scale image recongition

## Segmentation
Tensorflow implementation for simple color segmentation ([tensorflow](examples/color/segmentation.py))

## Regression
Neural network implementations for linear ([tensorflow](examples/regression/linear_regression.py)) and non-linear regressions ([tensorflow](examples/regression/non_linear_regression.py))

## Modeling Fourier Transform / FFT
Neural network implementation for learning a fourier transform ([tensorflow](examples/fft/fft.py))

