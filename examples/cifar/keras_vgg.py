
'''
Keras Code for a VGG network (see (2015) Simonyan K, Zisserman A. Very Deep Convolutional Networks 
for Large-Scale Image Recognitionhttps://arxiv.org/pdf/1409.1556v6.pdf). 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
 
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

# VGG Network Architecture 
model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(_WIDTH, _HEIGHT, _CHANNELS)))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
 
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(NUM_OUTPUTS, activation='softmax'))

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
