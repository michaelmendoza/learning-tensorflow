
'''
Basic Keras Code for a convolutional neural network.
Uses simple style of creating networks with Sequential API.
'''

# Training Parameters
epochs = 10
batch_size = 16
validation_split = 0.2
shuffle = True

# Network Parameters
WIDTH = 28; HEIGHT = 28; CHANNELS = 1;
NUM_OUTPUTS = 10 

import time
import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, HEIGHT, WIDTH, CHANNELS)
x_test = x_test.reshape (-1,  HEIGHT, WIDTH, CHANNELS)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(HEIGHT, WIDTH, CHANNELS)),
  tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(padding="same", strides=(2, 2), pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(NUM_OUTPUTS, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
model.summary()

start = time.time();
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=shuffle)
evaluation = model.evaluate(x_test, y_test, verbose=1)
end = time.time()
print("Training Complete in " + "{0:.2f}".format(end - start) + " secs" )

# Plot Accuracy 
plt.plot(history.history["sparse_categorical_accuracy"])
plt.plot(history.history["val_sparse_categorical_accuracy"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Train Accuracy", "Test Accuracy"], loc="upper left")
plt.show();
