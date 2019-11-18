
'''
Tensorflow Code for a fourier transform network using a linear network 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from data_loader import DataLoader
data = DataLoader(64, 64)
x_test, y_test = data.get_flatten_data()
print(x_test.shape)

x_train, y_train = data.generate(10000, 64, 64)
print(x_train.shape)

# Training Parameters
learning_rate = 0.0001
num_epochs = 100

# Network Parameters 
WIDTH = data.WIDTH; HEIGHT = data.HEIGHT; CHANNELS = data.CHANNELS
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS =  WIDTH * HEIGHT * CHANNELS

x_train = x_test.reshape((-1, NUM_INPUTS))
y_train = y_test.reshape((-1, NUM_INPUTS))

# Total params: 67,108,864
model0 = Sequential([Dense(NUM_OUTPUTS, input_dim=NUM_INPUTS, use_bias=False)])

# Total params: 4,268,544
model1 = Sequential([
    Dense(256, activation=tf.nn.relu, input_dim=NUM_INPUTS),
    Dense(256, activation=tf.nn.relu, input_dim=NUM_INPUTS),
    Dense(NUM_OUTPUTS)
    ])

model = model1

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(x_train, y_train, epochs=num_epochs, batch_size=1)

data.show()

img = model(x_test)
img = np.reshape(img, (data.WIDTH, data.HEIGHT, data.CHANNELS))
plt.imshow(img[:,:,0], cmap="gray")
plt.show()

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from .data_loader import DataLoader
data = DataLoader()
print('Data Loaded')

# Training Parameters
learning_rate = 0.0001
num_steps = 2000
batch_size = 32
display_step = 100

# Network Parameters 
WIDTH = data.WIDTH; HEIGHT = data.HEIGHT; CHANNELS = data.CHANNELS
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS =  WIDTH * HEIGHT * CHANNELS

# Network Varibles and placeholders
X = tf.placeholder(tf.float64, [None, NUM_INPUTS])  # Input
Y = tf.placeholder(tf.float64, [None, NUM_OUTPUTS]) # Truth Data - Output

# Network Architecture
def simple_net(x): 
    he_init = tf.contrib.layers.variance_scaling_initializer()
    fc1 = tf.layers.dense(x, 128, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1')   
    fc2 = tf.layers.dense(fc1, NUM_OUTPUTS, activation=None, kernel_initializer=he_init, name='fc2') 
    return fc2

# Define loss and optimizer
prediction = simple_net(X) #unet(X)
loss = tf.reduce_mean(tf.square(prediction - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Initalize varibles, and run network 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print ('Start Training: BatchSize:', batch_size,' LearningRate:', learning_rate, 'Total Steps:', num_steps)

for step in range(num_steps):
    x, y = data.get()
    sess.run(trainer, feed_dict={X:x, Y:y})

    if(step % display_step == 0):
        _loss = sess.run(loss, feed_dict={ X:x, Y:y })    
        print("Step: " + str(step) + " Loss: " + str(_loss)) 

x, y = data.get()
img = sess.run(prediction, feed_dict={X:x})
img = np.reshape(img, (data.WIDTH, data.HEIGHT, data.CHANNELS))
plt.imshow(img[:,:,0], cmap="gray")
plt.show()
'''