
'''
Trains CIFAR-10 model with a ResNet with data augmentation. 
Data augmentation is done by maing the preprocessing layers part of the model

References: 
ResNet:  https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/
DataAug: https://www.tensorflow.org/tutorials/images/data_augmentation
         https://www.kaggle.com/hiramcho/hubmap-keras-augmentation-layers

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from ..models.resnet import resnet

# Reproducibility
seed = 42
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

''' Model Network and Training Parameters '''
# Training Parameters
epochs = 500
batch_size = 256

# Network Parameters
WIDTH = 32; HEIGHT = 32; CHANNELS = 3; NUM_OUTPUTS = 10

def generate_dataset():
    # Import Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(10000)
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    train_dataset = train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(5000).shuffle(10000)
    valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    valid_dataset = valid_dataset.repeat()

    return train_dataset, valid_dataset

def resnet_with_data_augmentation():
    resnet_model = resnet(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)

    '''
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomContrast(0.1),
        layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
        layers.experimental.preprocessing.RandomZoom(0.1, 0.1)
    ])
    '''

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomRotation(factor=0.15),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.experimental.preprocessing.RandomFlip(),
        layers.experimental.preprocessing.RandomContrast(factor=0.1),
    ])

    model = keras.Sequential([
        data_augmentation,
        resnet_model
    ])

    return model

def train_without_data_augmentation():
    train_dataset, valid_dataset = generate_dataset()
    model = resnet(HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)
    model.compile(optimizer=keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])

    # Train and Evaluate model
    history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=200,
            validation_data=valid_dataset,
            validation_steps=3)
    return history

def train_with_data_augmentation():
    train_dataset, valid_dataset = generate_dataset()
    model = resnet_with_data_augmentation()
    model.compile(optimizer=keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])

    # Train and Evaluate model
    history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=200,
            validation_data=valid_dataset,
            validation_steps=3)
    return history

def plot(data):
    # Plot Accuracy / Loss 
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Accuracy / Loss')

    axs[0,0].plot(data[0].history['acc'])
    axs[0,0].plot(data[0].history['val_acc'])
    axs[0,0].set_title('Without Data Aug')
    axs[0,0].set_ylabel('acc')
    axs[0,0].legend(["Train", "Test"], loc="lower right")

    axs[1,0].plot(data[0].history['loss'])
    axs[1,0].plot(data[0].history['val_loss'])
    axs[1,0].set_ylabel('loss')
    axs[1,0].legend(["Train", "Test"], loc="upper right")
    
    axs[0,1].plot(data[1].history['acc'])
    axs[0,1].plot(data[1].history['val_acc'])
    axs[0,1].set_title('With Data Aug')
    axs[0,1].legend(["Train", "Test"], loc="lower right")

    axs[1,1].plot(data[1].history['loss'])
    axs[1,1].plot(data[1].history['val_loss'])
    axs[1,1].legend(["Train", "Test"], loc="upper right")

    plt.show()

def train():
    history_without_data_aug = train_without_data_augmentation()
    history_with_data_aug = train_with_data_augmentation()
    plot([history_without_data_aug, history_with_data_aug])

train()