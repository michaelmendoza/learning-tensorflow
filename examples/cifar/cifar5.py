
'''
Keras Code for a VGG network (see (2015) Simonyan K, Zisserman A. Very Deep Convolutional Networks 
for Large-Scale Image Recognitionhttps://arxiv.org/pdf/1409.1556v6.pdf). 

Ref: # https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
 
# Import Dataset
from data_loader import DataLoader
cifar = DataLoader()

# Training Parameters
batch_size = 32
epochs = 10

# Network Parameters
_WIDTH = 32; _HEIGHT = 32; _CHANNELS = 3 
NUM_INPUTS = _WIDTH * _HEIGHT * _CHANNELS 
NUM_OUTPUTS = 10

# Resnet 
def res_net_block(input_data, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation=None)(x)
    x = layers.layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x

inputs = keras.Input(shape = (_HEIGHT, _WIDTH, _CHANNELS))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D(3)(x) # Why 3?

res_block_count = 10
for _ in range(res_block_count):
    x = res_net_block(x, 64)

x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')x()
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_OUTPUTS, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Define Loss and Optimizier
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# Reshape data
cifar.x_train = cifar.x_train.reshape(-1, _WIDTH, _HEIGHT, _CHANNELS)
cifar.x_test = cifar.x_test.reshape(-1, _WIDTH, _HEIGHT, _CHANNELS)

# Train the model, iterating on the data in batches of 32 samples
model.fit(cifar.x_train, cifar.y_train, epochs=epochs, batch_size=batch_size)

# Evaluate
print('')
print('Evaluate:')
loss_and_metrics = model.evaluate(cifar.x_test, cifar.y_test, verbose=1)
print('')
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (loss_and_metrics[0], loss_and_metrics[1]))
