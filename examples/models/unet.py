import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate

# Unet Model Architecture
def unet(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS):

    def down_block(x, filters):
        x = Conv2D(filters, (3, 3), padding="same", activation=tf.nn.relu, kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), padding="same", activation=tf.nn.relu, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)(x)
        #x = Dropout(rate=0.0)(x)
        return x

    def max_pool(x):
        return MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2))(x)

    def up_block(x, filters, skip_connect):
        x = Conv2DTranspose(filters, (3, 3), strides=2, padding="same", activation=tf.nn.relu)(x)
        x = concatenate([x, skip_connect], axis=3)
        x = Conv2D(filters, (3, 3), padding="same", activation=tf.nn.relu, kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), padding="same", activation=tf.nn.relu, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)(x)
        #x = Dropout(rate=0.0)(x)
        return x 

    def unet():
        fn = [32, 64, 128, 256, 512]
        fdepth = len(fn)

        x_stack = []
        xin = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), name='img')

        x = xin
        for idx in range(fdepth):
            x = down_block(x, fn[idx])

            if(idx < fdepth - 1):
                x_stack.append(x)
                x = max_pool(x)

        for idx in range(fdepth - 1):
            idx = fdepth - idx - 2
            x = up_block(x, fn[idx], x_stack.pop())

        xout = Conv2D(NUM_OUTPUTS, (1, 1), padding="same", activation=tf.nn.softmax)(x)
        return tf.keras.Model(inputs=xin, outputs=xout)

    return unet()