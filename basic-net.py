
'''
Basic Code for a single layer neural network
'''

import tensorflow as tf

# Import Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Training Parameters
learning_rate = 0.5
num_steps = 5000
batch_size = 100
display_step = 100

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
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run( train_step, feed_dict={x: batch_xs, y_: batch_ys} )
    
    if(i % display_step == 0)
      acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})
    	print("Step: " + step + " Test Accuracy: " + acc); 
