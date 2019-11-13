'''
Using tensorflow for regression with a simple neural network
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Create dataset
N = 400
data = lambda: None
data.x = np.linspace(-1, 1, N)
data.y = (10 * np.exp(data.x) + 2 * np.random.rand(N)) / (10 * math.exp(1))
plt.scatter(data.x, data.y)

# Training Parameters
num_epochs = 100

# Define Network
model = tf.keras.models.Sequential([
    Dense(256, activation=tf.nn.relu, input_shape=[1]),
    Dense(256, activation=tf.nn.relu),
    Dense(1)
])
model.compile(optimizer='adam',
              loss='mean_squared_error')
model.summary()

history = model.fit(data.x, data.y, epochs=num_epochs, validation_split=0.2, shuffle=True)
predictions = model.predict(data.x)

# Plot Results
plt.plot(data.x, predictions, 'r')
plt.show()