
'''
Keras Code for ResNet50 with transfer learning

https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b'''

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
epochs = 10

# Network Parameters
WIDTH = 32; HEIGHT = 32; CHANNELS = 3 
NUM_OUTPUTS = 10

def preprocess_cifar_data(X,Y):
    xp = keras.applications.resnet50.preprocess_input(X)
    yp = keras.utils.to_categorical(Y, NUM_OUTPUTS)
    return xp, yp

# Import Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print((x_train.shape, y_train.shape))
x_train, y_train = preprocess_cifar_data(x_train, y_train)
x_test, y_test = preprocess_cifar_data(x_test, y_test)
print((x_train.shape, y_train.shape))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()

valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(5000).shuffle(10000)
valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
valid_dataset = valid_dataset.repeat()

def resnet():
    model_resnet = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

    for layer in model_resnet.layers[:143]: # Freeze all layers except for the last block of ResNet50
        layer.trainable = False
    
    for i, layer in enumerate(model_resnet.layers):
        print(i, layer.name, "-", layer.trainable)

    #model = keras.models.Sequential()
    #model.add(model_resnet)
    #model.add(keras.layers.Flatten())
    #model.add(K.layers.Dense(NUM_OUTPUTS, activation='softmax'))
    to_res = (224, 224)
    model = keras.models.Sequential()
    model.add(keras.layers.Lambda(lambda image: tf.image.resize(image, to_res))) 
    model.add(model_resnet)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


model = resnet()
model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])

check_point = keras.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                              monitor="val_acc",
                                              mode="max",
                                              save_best_only=True,
                                              )

# Train and Evaluate model
history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=200,
            verbose=1,
            validation_data=valid_dataset,
            validation_steps=3,
            callbacks=[check_point])

model.summary()


