import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def resnet(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):

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
    outputs = layers.Dense(NUM_OUTPUTS, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model