
'''
Ref: # https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Training Parameters
epochs = 30

# Network Parameters
WIDTH = 32; HEIGHT = 32; CHANNELS = 3 

# Import Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()

valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(5000).shuffle(10000)
valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
valid_dataset = valid_dataset.repeat()

def resnet():

    def res_net_block(input_data, filters, conv_size):
      x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
      x = layers.BatchNormalization()(x)
      x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
      x = layers.BatchNormalization()(x)
      x = layers.Add()([x, input_data])
      x = layers.Activation('relu')(x)
      return x

    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)

    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

model = resnet()
model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# Train and Evaluate model
model.fit(train_dataset, epochs=epochs, steps_per_epoch=200,
          validation_data=valid_dataset,
          validation_steps=3)