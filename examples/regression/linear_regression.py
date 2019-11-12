'''
Using tensorflow for simple linear regression.
Uses imperative style of creating networks with model subclassing API.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Create dataset
N = 400
data = lambda: None
data.x = np.linspace(0, 1, N)
data.y = 10 * np.exp(data.x) + 0.5 * 2 * np.random.rand(N)
plt.scatter(data.x, data.y)

# Training Parameters
learning_rate = 0.001
num_epochs = 100
display_step = 10

class Model(object):
    def __init__(self):
        self.W = tf.Variable(0.0, dtype=tf.float64) # Weights for layer
        self.b = tf.Variable(0.0, dtype=tf.float64) # Bias for layer

    def __call__(self, x):
        return self.W * x + self.b

@tf.function
def loss(y, y_target):
    return tf.reduce_mean(tf.square(y - y_target))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

model = Model()

# Train model and collect accuracy for plotting 
epochs = range(num_epochs)
for epoch in epochs:
    current_loss = loss(model(data.x), data.y)
    train(model, data.x, data.y, learning_rate=0.1)

    if(epoch % display_step == 0):
        print('Epoch %2d: training loss=%2.5f' % (epoch, current_loss))

# Plot Results
plt.plot(data.x, model.W * data.x + model.b, 'r')
plt.show()