
'''
Basic Code (updated for TF 2.0) for a convolutional neural network with 2 conv layers, a max pool layer, and 2 full-connected layers
'''

from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Training Parameters
num_epochs = 10
display_step = 1
batch_size = 32

# Network Parameters
WIDTH = 28; HEIGHT = 28; CHANNELS = 1
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS = 10

# Import Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train).astype(np.float32)
y_test = tf.keras.utils.to_categorical(y_test).astype(np.float32)
x_train = x_train.reshape(-1, HEIGHT, WIDTH, CHANNELS).astype(np.float32)
x_test = x_test.reshape (-1,  HEIGHT, WIDTH, CHANNELS).astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Network Architecture
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.c1 = Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu)
        self.c2 = Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu)
        self.mp = MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2))
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation=tf.nn.relu)
        self.dropout = Dropout(0.2)
        self.fc2 = Dense(128, activation=tf.nn.relu)
        self.fcout = Dense(NUM_OUTPUTS, activation=tf.nn.softmax)

    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.mp(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
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
    for x_train, y_train in train_ds:
        train(model, x_train, y_train)

    for x_test, y_test in test_ds:
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
