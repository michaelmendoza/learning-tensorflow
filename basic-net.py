
'''
Basic Code for a single layer neural network
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Training Parameters
learning_rate = 0.1
num_steps = 5000
batch_size = 128
display_step = 10

# Network Parameters
NUM_INPUTS = 784
NUM_OUTPUTS = 10

# Network Varibles and placeholders
x = tf.placeholder(tf.float32, [None, NUM_INPUTS])   # Input
y_ = tf.placeholder(tf.float32, [None, NUM_OUTPUTS]) # Truth Data - Output
W = tf.Variable(tf.zeros([NUM_INPUTS, NUM_OUTPUTS])) # Weights for layer
b = tf.Variable(tf.zeros([NUM_OUTPUTS]))						 # Bias for layer

# Network Architecture
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Calculate Loss and Accuracy 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Setup Optimizer for trainning
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Initalize varibles, and run network
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train network
_step = []
_acc = []
for step in range(num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run( train_step, feed_dict={x: batch_xs, y_: batch_ys} )

    if(step % display_step == 0):
      acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})
      _step.append(step)
      _acc.append(acc)
      print("Step: " + str(step) + " Test Accuracy: " + str(acc)) 

# Plot Accuracy
plt.plot(_step, _acc, label="test accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for MINST Classification")
plt.show()






