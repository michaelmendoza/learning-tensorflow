
'''
Basic Keras Code for a single-layer neural network
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
from keras.models import Sequential
from keras.layers import Dense
 
# Import Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Training Parameters
batch_size = 128
epochs = 10

# Network Parameters
NUM_INPUTS = 784
NUM_OUTPUTS = 10

model = Sequential()
model.add(Dense(NUM_OUTPUTS, activation='softmax', input_dim=NUM_INPUTS))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model, iterating on the data in batches of 32 samples
model.fit(mnist.train.images, mnist.train.labels, epochs=epochs, batch_size=batch_size)

# Evaluate
print('')
print('Evaluate:')
evaluation = model.evaluate(mnist.test.images, mnist.test.labels, verbose=1)
print('')
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))