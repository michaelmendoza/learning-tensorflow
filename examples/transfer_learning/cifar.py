
'''
Trains CIFAR-10 model with a ResNet with transfer learning. 

References: 
ResNet:  https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/
Transfer Learning: https://medium.com/swlh/resnet-with-tensorflow-transfer-learning-13ff0773cf0c
                    https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b
                    https://medium.com/swlh/hands-on-the-cifar-10-dataset-with-transfer-learning-2e768fd6c318
                    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
EarlyStopping: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
                https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from examples.models.transfer_learning import model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
seed = 42
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

''' Model Network and Training Parameters '''
# Training Parameters
epochs = 50
batch_size = 128

# Network Parameters
CIFAR_WIDTH = 32; CIFAR_HEIGHT = 32
WIDTH = 160; HEIGHT = 160; CHANNELS = 3; NUM_OUTPUTS = 10

'''
Generates a dataset using a tensorflow dataset. Includes pre-processing. 
'''
def generate_dataset():
    # Import Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(10000)
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.resize_with_pad(x, HEIGHT, WIDTH), y))  # upscale to prevent overfitting
    train_dataset = train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(500).shuffle(10000)
    valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    valid_dataset = valid_dataset.map(lambda x, y: (tf.image.resize_with_pad(x, HEIGHT, WIDTH), y))
    valid_dataset = valid_dataset.repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    test_dataset = test_dataset.map(lambda x, y: (tf.image.resize_with_pad(x, HEIGHT, WIDTH), y))

    return train_dataset, valid_dataset, test_dataset

'''
Trains a model for a given model_name 
'''
def trainOne(model_name):
    train_dataset, valid_dataset, test_dataset = generate_dataset()

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1, patience=5)

    from ..models.transfer_learning import model 
    model = model(model_name, HEIGHT, WIDTH, CHANNELS, NUM_OUTPUTS)
    model.compile(optimizer=keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])
    #model.summary()
    
    # Train and Evaluate model
    history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=200,
            validation_data=valid_dataset,
            validation_steps=20,
            callbacks=[earlyStopping])
    loss_and_metrics = model.evaluate(test_dataset)

    return model, history, loss_and_metrics

'''
Plots metric history data for training a model 
'''
def plot(data, model_name, summary):
    # Plot Accuracy / Loss 
    fig, axs = plt.subplots(2)
    fig.suptitle(model_name + ': ' + summary)

    axs[0].plot(data[0].history['acc'])
    axs[0].plot(data[0].history['val_acc'])
    axs[0].set_ylabel('acc')
    axs[0].legend(["Train", "Test"], loc="lower right")

    axs[1].plot(data[0].history['loss'])
    axs[1].plot(data[0].history['val_loss'])
    axs[1].set_ylabel('loss')
    axs[1].legend(["Train", "Test"], loc="upper right")
    
    #plt.show()
    plt.savefig('cifar_' + model_name +'.png')

'''
Saves training and test metrics to metrics.npy
'''
def load_and_save(model_name, history, metrics):
    fname = 'metrics.npy' 
    fileExists = os.path.isfile(fname)
    if(fileExists):
        data = np.load(fname, allow_pickle=True)[()]
        data[model_name] = { 'history': history.history, 'metrics': metrics }
    else:
        data = { model_name: { 'history': history.history, 'metrics': metrics } }
    
    np.save(fname, data)

'''
Runs a single training session
'''
def run():
    model, history = trainOne("DenseNet121")
    plot([history], "DenseNet121", "")


'''
Runs and trains all models in transfer_learning models
'''
def runAll():
    from ..models.transfer_learning import getModelNames
    for model_name in getModelNames():
        print("Running ... " + model_name)

        # Train model 
        start = time.time()
        model, history, metrics = trainOne(model_name)
        end = time.time()

        # Plot and Output results
        summary = 'Loss: %.4f, Accuracy: %.4f, Time: %.2fs' % (metrics[0], metrics[1], (end - start))
        print(model_name + ': ' + summary)
        plot([history], model_name, summary)

        # Save metric history 
        metrics = { 'loss': metrics[0], 'acc': metrics[1], 'time': (end - start) }
        load_and_save(model_name, history, metrics)

# Program Entry 
runAll()
