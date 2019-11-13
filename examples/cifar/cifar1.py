
'''
Basic Keras Code for a multi-layer neural network
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
 
# Import Dataset
from data_loader import DataLoader
cifar = DataLoader()

# Training Parameters
batch_size = 128
epochs = 10

# Network Parameters
_WIDTH = 32; _HEIGHT = 32; _CHANNELS = 3 
NUM_INPUTS = _WIDTH * _HEIGHT * _CHANNELS 
NUM_OUTPUTS = 10
NUM_H1 = 512
NUM_H2 = 256

# Network Architecture
model = Sequential()
model.add(Dense(NUM_H1, activation='relu', input_dim=NUM_INPUTS))
model.add(Dense(NUM_H2, activation='relu'))
model.add(Dense(NUM_OUTPUTS, activation='softmax'))

# Define Loss and Optimizier
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# Train the model, iterating on the data in batches of 32 samples
model.fit(cifar.x_train, cifar.y_train, epochs=epochs, batch_size=batch_size)

# Evaluate
print('')
print('Evaluate:')
loss_and_metrics = model.evaluate(cifar.x_test, cifar.y_test, verbose=1)
print('')
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (loss_and_metrics[0], loss_and_metrics[1]))
