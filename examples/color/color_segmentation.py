
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
import matplotlib
import matplotlib.pyplot as plt

# Generate Dataset
from ..mockdata.color_square_data import DataGenerator
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

# Unet Model Architecture
from ..models import unet

# Setup Unet model
model = unet.unet(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)
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

def plot(history):
    # Plot Accuracy / Loss 
    fig, axs = plt.subplots(2)
    fig.suptitle('Accuracy / Loss')

    axs[0].plot(history['acc'])
    axs[0].plot(history['val_acc'])
    axs[0].set_ylabel('acc')
    axs[0].legend(["Train", "Test"], loc="lower right")

    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set_ylabel('loss')
    axs[1].legend(["Train", "Test"], loc="upper right")

    plt.show()