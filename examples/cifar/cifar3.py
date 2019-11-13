
'''
Keras Code for a convolutional neural network with 6 conv layers, 3 max pool layers, and 2 full-connected layers. 
This code also uses dropout to prevent over-generalization.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import tensorflow as tf 
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
NUM_C1 = 32
NUM_C2 = 64
NUM_C3 = 128
NUM_H1 = 1024
NUM_H2 = 1024

# Network Architecture 
model = Sequential()
model.add(Conv2D(NUM_C1, (3, 3), padding="same", activation="relu", input_shape=(_WIDTH, _HEIGHT, _CHANNELS)))
model.add(Conv2D(NUM_C1, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.25)) 

model.add(Conv2D(NUM_C2, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(NUM_C2, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Conv2D(NUM_C3, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(NUM_C3, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(NUM_H1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_H2, activation='relu'))
model.add(Dropout(0.5))
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
