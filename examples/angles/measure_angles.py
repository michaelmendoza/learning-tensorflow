
'''
Tensorflow Convolution Angle Estimator
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from angle_data_generator import DataGenerator
data = DataGenerator()

# Training Parameters
learning_rate = 0.001
num_steps = 100
batch_size = 128
display_step = 10

# Network Parameters
WIDTH = 128; HEIGHT = 128; CHANNELS = 1
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS = 1

# Network Varibles and placeholders
X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS])  # Input
Y = tf.placeholder(tf.float32, [None, NUM_OUTPUTS]) # Truth Data - Output

def simple_net(x): 
    # Reshape to match picture format [BatchSize, Height x Width x Channel] => [Batch Size, Height, Width, Channel]
    x = tf.reshape(X, shape=[-1, HEIGHT, WIDTH, CHANNELS])
    
    # Reshape to fit to fully connected layer input
    flatten = tf.contrib.layers.flatten(x)

    # Fully-connected layers 
    he_init = tf.contrib.layers.variance_scaling_initializer()
    fc1 = tf.layers.dense(flatten, 128, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1')   # First hidden layer with relu
    out = tf.layers.dense(fc1, NUM_OUTPUTS,  activation=None, name='logits')  # this tf.layers.dense is same as tf.matmul(x, W) + b
    return out

# Network Architecture
def network(x):
    # Reshape to match picture format [BatchSize, Height x Width x Channel] => [Batch Size, Height, Width, Channel]
    x = tf.reshape(X, shape=[-1, HEIGHT, WIDTH, CHANNELS])

    # Convolutional layers and max pool
    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     16, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h1')
    conv2 = tf.layers.conv2d(conv1, 16, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h2')
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # Reshape to fit to fully connected layer input
    flatten = tf.contrib.layers.flatten(pool1)

    # Fully-connected layers 
    fc1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1')   # First hidden layer with relu
    fc2 = tf.layers.dense(fc1, 256, activation=tf.nn.relu, kernel_initializer=he_init, name='fc2') # Second hidden layer with relu
    out = tf.layers.dense(fc2, NUM_OUTPUTS,  activation=None, name='logits')  # this tf.layers.dense is same as tf.matmul(x, W) + b
    return out

# Define loss and optimizer 
prediction = simple_net(X)
loss = tf.reduce_mean(tf.square(prediction - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Initalize varibles, and run network
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print ('Start Training: BatchSize:', batch_size,' LearningRate:', learning_rate, 'Total Steps:', num_steps)

# Train network
_step = []
_loss = []
for step in range(num_steps):
    batch_xs, _, batch_ys = data.next_batch(batch_size)
    sess.run( trainer, feed_dict={X: batch_xs, Y: batch_ys} )

    if(step % display_step == 0):
      loss_value = sess.run(loss, feed_dict={X: data.x_test, Y:data.angles_test})
      _step.append(step)
      _loss.append(loss_value)

      print("Step: " + str(step) + " Test Loss: " + str(loss_value)) 

# Plot Accuracy 
plt.plot(_step, np.log(_loss), label="test accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for Angle Classification")
plt.show()

pred = sess.run(prediction, feed_dict={X: data.x_test})

plt.plot(pred[0:100])
plt.plot(data.angles_test[0:100])
plt.show()
