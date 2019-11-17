
'''
Tensorflow Code (updated for TF 2.0) for a single layer neural network.
Uses imperative style of creating networks with model subclassing API.
'''

from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Training Parameters
learning_rate = 0.1
num_epochs = 100
display_step = 10

# Network Parameters
NUM_INPUTS = 784
NUM_OUTPUTS = 10

# Import Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train).astype(np.float32)
y_test = tf.keras.utils.to_categorical(y_test).astype(np.float32)
x_train = x_train.reshape(-1, NUM_INPUTS).astype(np.float32)
x_test = x_test.reshape (-1, NUM_INPUTS).astype(np.float32)

class Model(object):
    def __init__(self):
        self.W = tf.Variable(tf.zeros([NUM_INPUTS, NUM_OUTPUTS]), dtype=tf.float32) # Weights for layer
        self.b = tf.Variable(tf.zeros([NUM_OUTPUTS]), dtype=tf.float32)             # Bias for layer

    def __call__(self, x):
        return tf.nn.softmax(tf.matmul(x, self.W) + self.b)

@tf.function
def loss(y, y_target):
    return tf.reduce_mean(-tf.reduce_sum(y_target * tf.math.log(y), axis=[1]))

@tf.function
def accuracy(y, y_target):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_target,1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)
    return current_loss

@tf.function
def test(model, inputs, outputs):
    prediction = model(inputs)
    return accuracy(prediction, outputs)

model = Model()

# Train model and collect accuracy for plotting 
data = []
epochs = range(num_epochs)
for epoch in epochs:
    current_loss =  train(model, x_train, y_train, learning_rate=0.1)
    acc = accuracy(model(x_test), y_test)
    data.append([epoch, acc])

    if(epoch % display_step == 0):
        print('Epoch %2d: training loss=%2.5f test accuracy=%2.5f' % (epoch, current_loss, acc))

# Plot Accuracy
data = np.array(data)
plt.plot(data.T[0], data.T[1], label="test accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for MINST Classification")
plt.show()
