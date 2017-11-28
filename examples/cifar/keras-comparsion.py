
'''
Keras Code for the comparstion of a variety of deep learning models for CIFAR-10 Image Classification
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
 
# Import Dataset
from data_loader import DataLoader
cifar = DataLoader()

# Training Parameters
batch_size = 32
epochs = 50

# Network Parameters
_WIDTH = 32; _HEIGHT = 32; _CHANNELS = 3 
NUM_INPUTS = _WIDTH * _HEIGHT * _CHANNELS 
NUM_OUTPUTS = 10

def model_basic():
    # Simple Perception
    model = Sequential()
    model.add(Dense(NUM_OUTPUTS, activation='softmax', input_dim=NUM_INPUTS))
    return model

def model_mlp():
    # Multi-Layer Perception
    NUM_H1 = 512
    NUM_H2 = 256

    model = Sequential()
    model.add(Dense(NUM_H1, activation='relu', input_dim=NUM_INPUTS))
    model.add(Dense(NUM_H2, activation='relu'))
    model.add(Dense(NUM_OUTPUTS, activation='softmax'))
    return model

def model_conv():
    # Convolutional Network Architecture
    NUM_C1 = 32
    NUM_H1 = 512
    NUM_H2 = 256

    model = Sequential()
    model.add(Conv2D(NUM_C1, (3, 3), padding="same", activation="relu", input_shape=(_WIDTH, _HEIGHT, _CHANNELS)))
    model.add(Conv2D(NUM_C1, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(NUM_H1, activation='relu'))
    model.add(Dense(NUM_H2, activation='relu'))
    model.add(Dense(NUM_OUTPUTS, activation='softmax'))
    return model

def model_conv2():
    # Convolutional Network Architecture
    NUM_C1 = 32
    NUM_C2 = 64
    NUM_C3 = 128
    NUM_H1 = 1024
    NUM_H2 = 1024

    model = Sequential()
    model.add(Conv2D(NUM_C1, (3, 3), padding="same", activation="relu", input_shape=(_WIDTH, _HEIGHT, _CHANNELS)))
    model.add(Conv2D(NUM_C1, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25)) 

    model.add(Conv2D(NUM_C2, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(NUM_C2, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(NUM_C3, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(NUM_C3, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(NUM_H1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_H2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_OUTPUTS, activation='softmax'))
    return model 

def model_vgg():
    # VGG Network Architecture 

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(_WIDTH, _HEIGHT, _CHANNELS)))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
     
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(NUM_OUTPUTS, activation='softmax'))
    return model

def train_and_evaluate(model, useReshape):

    # Define Loss and Optimizier
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    # Reshape data
    if(useReshape):
        cifar.x_train = cifar.x_train.reshape(-1, _WIDTH, _HEIGHT, _CHANNELS)
        cifar.x_test = cifar.x_test.reshape(-1, _WIDTH, _HEIGHT, _CHANNELS)

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit(cifar.x_train, cifar.y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(cifar.x_test, cifar.y_test))

    # Evaluate
    loss_and_metrics = model.evaluate(cifar.x_test, cifar.y_test, verbose=0)
    print('\n Evaluate: Loss over the test dataset: %.2f, Accuracy: %.2f' % (loss_and_metrics[0], loss_and_metrics[1]))
    return history, loss_and_metrics

def plot_accuracy(name, history):
    # Plot Training History
    plt.plot(history.history['categorical_accuracy'], label="train accuracy")
    plt.plot(history.history['val_categorical_accuracy'], label="test accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy")
    plt.legend(['train','test'], loc='upper left')
    #plt.show()
    plt.savefig(name + '-acc.png')

if __name__== "__main__":
    
    models = { "basic": model_basic(), "mlp": model_mlp(), "conv":model_conv(), "conv2":model_conv2(), "vgg":model_vgg() }

    for key in models:
        useReshape = not(key == "basic" or key == "mlp")
        history, metrics = train_and_evaluate(models[key], useReshape)
        plot_accuracy(key, history)
