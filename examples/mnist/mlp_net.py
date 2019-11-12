
'''
Basic Code (updated for TF 2.0) for a multi-layer neural network with two hidden layers
'''

from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Training Parameters
learning_rate = 0.1
num_epochs = 1000
display_step = 100

# Network Parameters
NUM_INPUTS = 784
NUM_OUTPUTS = 10
NUM_H1 = 512
NUM_H2 = 256

# Import Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train).astype(np.float32)
y_test = tf.keras.utils.to_categorical(y_test).astype(np.float32)
x_train = x_train.reshape(-1, NUM_INPUTS).astype(np.float32)
x_test = x_test.reshape (-1, NUM_INPUTS).astype(np.float32)

# Network Architecture
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = Dense(NUM_H1, activation='relu', input_dim=NUM_INPUTS)
        self.fc2 = Dense(NUM_H2, activation='relu')
        self.fcout = Dense(NUM_OUTPUTS, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fcout(x)

@tf.function
def train(model, inputs, outputs):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss_value = loss(predictions, outputs)
    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
    train_loss(loss_value)

@tf.function
def test(model, inputs, outputs):
    predictions = model(inputs)
    test_accuracy(outputs, predictions)

model = Model()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

data = []
epochs = range(num_epochs)
for epoch in epochs:
    train(model, x_train, y_train)
    test(model, x_test, y_test)

    data.append([epoch, test_accuracy.result()])

    if(epoch % display_step == 0):
        print('Epoch %2d: training loss=%2.5f test accuracy=%2.5f' % 
            (epoch, train_loss.result(), test_accuracy.result()))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    test_accuracy.reset_states()

# Plot Accuracy
data = np.array(data)
plt.plot(data.T[0], data.T[1], label="test accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for MINST Classification")
plt.show()
