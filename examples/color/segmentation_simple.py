
'''
Tensorflow Code for a color segmentation network
'''

from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
import matplotlib
import matplotlib.pyplot as plt

# Generate Dataset
from data_loader import DataGenerator
data = DataGenerator()
data.print()

# Training Parameters
num_epochs = 10
display_step = 1
batch_size = 4

# Network Parameters
WIDTH = data.WIDTH; HEIGHT = data.HEIGHT; CHANNELS = data.CHANNELS
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS = 2

# Simple Model Architecture
def Model():
    xin = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), name='img')
    x = Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu)(xin)
    x = Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu)(x)
    xout = Conv2D(NUM_OUTPUTS, (1, 1), padding="same", activation=tf.nn.softmax)(x)
    return tf.keras.Model(inputs=xin, outputs=xout)

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

# Setup Unet model
model = Model()
model.summary()

# Set Losses and Optimizers 
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

epochs = range(num_epochs)
for epoch in epochs:
    for _ in range(data.batch_count(batch_size)):
        x_train, y_train = data.next_batch(batch_size)
        train(model, x_train, y_train)
        test(model, data.x_test, data.y_test)
    
    if(epoch % display_step == 0):
        print('Epoch %2d: training loss=%2.5f test accuracy=%2.5f' % 
            (epoch, train_loss.result(), test_accuracy.result()))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    test_accuracy.reset_states()

predictions = model(data.x_test)

index = 0
input_data = data.unwhiten_img(data.x_test[index])
truth_data = data.y_test[index][:,:,1] 
segmentation = np.greater(predictions[index,:,:,1], predictions[index,:,:,0]) * 1.0

plt.subplot(3,1,1)
plt.imshow(input_data)
plt.subplot(3,1,2)
plt.imshow(truth_data, cmap='gray')
plt.subplot(3,1,3)
plt.imshow(segmentation, cmap='gray')
plt.show() 
