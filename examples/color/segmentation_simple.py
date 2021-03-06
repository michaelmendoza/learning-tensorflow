
'''
Tensorflow Code for a color segmentation network
'''

from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
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
x_train = data.x_train
x_test = data.x_test
y_train = data.y_train
y_test = data.y_test

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

# Setup Unet model
model = Model()
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

start = time.time()
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, shuffle=True)
evaluation = model.evaluate(x_test, y_test, verbose=0)
end = time.time()
print("Training Complete in " + "{0:.2f}".format(end - start) + " secs" )

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
