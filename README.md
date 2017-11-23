# Learning Tensorflow

This tutorial was created for learning tensorflow by example. Currently this repo contains examples for a simple single-layer neural network, a multi-layered perception neural network, and a convolutional neural network. Tensorflow implementations in this repo work with a variety of data sets. Keras implmentations are also included as a comparison for some cases.

# Examples

## Neural Networks
A variety of neural network implementations for MNIST, and CFAR-10 datasets

### MNIST
- Basic Neural Network - A simple (single layer preception) network for classifying MNIST dataset ([tensorflow](examples/mnist/basic-net.py))
- Multi-layer Neural Nework - A simple (multi-layer preception) network for classifying MNIST dataset ([tensorflow](examples/mnist/mlp-net.py))
- Convolutional Neural Nework - A convolutional network for classifying MNIST dataset ([tensorflow](examples/mnist/conv-net.py))

### CIFAR-10
- Basic Neural Network - A simple (single layer preception) network for classifying CIFAR-10 dataset ([tensorflow](examples/cifar/basic-net.py), [keras](examples/cifar/keras-basic.py))
- Multi-layer Neural Nework - A simple (multi-layer preception) network for classifying CIFAR-10 dataset ([tensorflow](examples/cifar/mlp-net.py))
- Convolutional Neural Nework - A convolutional network for classifying CIFAR-10 dataset ([tensorflow](examples/cifar/conv-net.py), [keras](examples/cifar/keras-conv.py))
- Convolutional Neural Nework - A convolutional network (6-conv, 3 max pool, 2 fully-connected layers) with Dropout for classifying CIFAR-10 dataset ([keras](examples/cifar/keras-nine-layer-conv.py))