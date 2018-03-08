# Learning Tensorflow
This tutorial was created for learning tensorflow by example. Currently this repo contains examples for a simple single-layer neural network, a multi-layered perception neural network, and a convolutional neural network. Tensorflow implementations in this repo work with a variety of data sets. Keras implmentations are also included as a comparison for some cases. 

# Install
This code is tested with python 2.7/3.2 using tensorflow 1.3 and keras 2.0. Installation instructions for Windows can be found [here](install.md). For installation on Mac/Linux search for information on the main tensorflow [website](https://www.tensorflow.org/install/).

# Examples
Examples of tensorflow implementations for Classification, Segmentation, Regression and Modeling Fourier Transform

## Classification
A variety of neural network implementations for MNIST, and CFAR-10 datasets for classification

### MNIST
- Basic Neural Network ([tutorial](notebooks/mnist/0_Single_Layer_Network_Tutorial.ipynb), [tensorflow](examples/mnist/basic-net.py)) - A simple (single layer preception) network for classifying MNIST dataset 
- Multi-layer Neural Nework ([tensorflow](examples/mnist/mlp-net.py)) - A simple (multi-layer preception) network for classifying MNIST dataset 
- Convolutional Neural Nework ([tensorflow](examples/mnist/conv-net.py)) - A convolutional network for classifying MNIST dataset 

### CIFAR-10
- Basic Neural Network ([tensorflow](examples/cifar/basic-net.py), [keras](examples/cifar/keras-basic.py)) - A simple (single layer preception) network for classifying CIFAR-10 dataset 
- Multi-layer Neural Nework ([tensorflow](examples/cifar/mlp-net.py), [keras](examples/cifar/keras-mlp.py)) - A simple (multi-layer preception) network for classifying CIFAR-10 dataset 
- Convolutional Neural Nework ([tensorflow](examples/cifar/conv-net.py), [keras](examples/cifar/keras-conv.py)) - A convolutional network for classifying CIFAR-10 dataset
- Convolutional Neural Nework ([keras](examples/cifar/keras-nine-layer-conv.py)) - A convolutional network (6-conv, 3 max pool, 2 fully-connected layers) with Dropout for classifying CIFAR-10 dataset 
- VGG network ([keras](examples/cifar/keras-vgg.py), [paper](https://arxiv.org/pdf/1409.1556v6.pdf)) - A very deep convolutional network for large-scale image recongition

## Segmentation
Tensorflow implementation for simple color segmentation ([tensorflow](examples/color/segmentation.py))

## Regression
Neural network implementations for linear ([tensorflow](examples/regression/linear_regression.py)) and non-linear regressions ([tensorflow](examples/regression/non-linear_regression.py))

## Modeling Fourier Transform / FFT
Neural netowrk implementation for learning a fourier transform ([tensorflow](examples/fft/fft.py))

