
'''
Basic Code for a multi-layer neural network with two hidden layers
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 5000
batch_size = 128
display_step = 100

# Network Parameters
NUM_INPUTS = 784
NUM_OUTPUTS = 10
NUM_H1 = 512
NUM_H2 = 256

# Network Varibles and placeholders
X = tf.placeholder(tf.float32, [None, NUM_INPUTS])  # Input
Y = tf.placeholder(tf.float32, [None, NUM_OUTPUTS]) # Truth Data - Output

# Network Architecture
he_init = tf.contrib.layers.variance_scaling_initializer() # Sets init wieghts, used with relu activation 
fc1 = tf.layers.dense(X, NUM_H1, activation=tf.nn.relu, kernel_initializer=he_init, name='fc1')   # First hidden layer with relu
fc2 = tf.layers.dense(fc1, NUM_H2, activation=tf.nn.relu, kernel_initializer=he_init, name='fc2') # Second hidden layer with relu
logits = tf.layers.dense(fc2, NUM_OUTPUTS, name='logits')  # this tf.layers.dense is same as tf.matmul(x, W) + b
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initalize varibles, and run network
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train network
_step = []
_acc = []
for step in range(num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run( trainer, feed_dict={X: batch_xs, Y: batch_ys} )

    if(step % display_step == 0):
      acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y:mnist.test.labels})
      _step.append(step)
      _acc.append(acc)

      print("Step: " + str(step) + " Test Accuracy: " + str(acc)) 

# Plot Accuracy
plt.plot(_step, _acc, label="test accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for MINST Classification")
plt.show()
