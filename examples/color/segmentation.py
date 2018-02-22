
'''
Tensorflow Code for a color segmentation network
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from data_loader import DataGenerator
data = DataGenerator()
data.print()

# Training Parameters
learning_rate = 0.0001
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
WIDTH = 32; HEIGHT = 32; CHANNELS = 3
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS = 2
NUM_C1 = 32
NUM_C2 = 32

# Network Varibles and placeholders
X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS])  # Input
Y = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_OUTPUTS]) # Truth Data - Output

# Network Architecture
def network(x):

    # Convolutional layers and max pool
    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     NUM_C1, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h1')
    conv2 = tf.layers.conv2d(conv1, NUM_C2, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h2')
    logits = tf.layers.conv2d(conv2, NUM_OUTPUTS, [1, 1], padding="SAME", activation=None, kernel_initializer=he_init, name='output')
    prediction = tf.nn.softmax(logits)
    return logits, prediction

# Define loss and optimizer
logits, prediction = network(X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Evaluate model
prediction_state = tf.argmax(prediction, 3)
correct_pred = tf.equal(tf.argmax(prediction, 3), tf.argmax(Y, 3))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initalize varibles, and run network 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print ('Start Training: BatchSize:', batch_size,' LearningRate:', learning_rate)

# Train network
_step = []
_acc = []
for step in range(num_steps):
    batch_xs, batch_ys = data.next_batch(batch_size)
    sess.run( trainer, feed_dict={ X: batch_xs, Y: batch_ys } )

    if(step % display_step == 0):
      acc = sess.run(accuracy, feed_dict={ X: data.x_test, Y: data.y_test })
      _step.append(step)
      _acc.append(acc)

      print("Step: " + str(step) + " Test Accuracy: " + str(acc)) 

# Plot Accuracy
plt.plot(_step, _acc, label="test accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy for Color Segmentation")
plt.show()


# Show results
predict = sess.run(correct_pred, feed_dict={ X: data.x_test, Y: data.y_test })
print(predict.shape)
index = 0;
matplotlib.image.imsave('results/real-img.png', data.x_test[index], cmap='gray') 
matplotlib.image.imsave('results/real-test.png', data.y_test[index][:,:,0], cmap='gray') 
matplotlib.image.imsave('results/real-results.png', predict[index], cmap='gray') 