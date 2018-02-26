
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Dataset
from data_loader import DataGenerator
data = DataGenerator()
data.print()

# Training Parameters
learning_rate = 0.0001
num_steps = 10000
batch_size = 32
display_step = 100

# Network Parameters 
WIDTH = 128; HEIGHT = 128; CHANNELS = 3
NUM_INPUTS = WIDTH * HEIGHT * CHANNELS
NUM_OUTPUTS = 2
NUM_C1 = 32
NUM_C2 = 32

# Network Varibles and placeholders
X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS])  # Input
Y = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_OUTPUTS]) # Truth Data - Output

# Network Architecture
def network(x):

    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     NUM_C1, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h1')
    conv2 = tf.layers.conv2d(conv1, NUM_C2, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='h2')
    logits = tf.layers.conv2d(conv2, NUM_OUTPUTS, [1, 1], padding="SAME", activation=None, kernel_initializer=he_init, name='output')
    prediction = tf.nn.softmax(logits)
    return logits, prediction

def unet_simple(x):

    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1')
    conv2 = tf.layers.conv2d(conv1, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2')
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    conv3 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3')
    conv4 = tf.layers.conv2d(conv3, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4')
    up1 = tf.layers.conv2d_transpose(conv4, 64, [3, 3], strides=2, padding="SAME", name='Up1')

    conv5 = tf.layers.conv2d(up1,   32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5')
    conv6 = tf.layers.conv2d(conv5, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6')
    logits = tf.layers.conv2d(conv6, NUM_OUTPUTS, [1, 1], padding="SAME", activation=None,       kernel_initializer=he_init, name='Output')
    prediction = tf.nn.softmax(logits)
    return logits, prediction

def unet(x):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.conv2d(x,     32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1')
    conv1 = tf.layers.conv2d(conv1, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv1-2')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    conv2 = tf.layers.conv2d(pool1, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2')
    conv2 = tf.layers.conv2d(conv2, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv2-2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(pool2, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3')
    conv3 = tf.layers.conv2d(conv3, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv3-3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)    

    conv4 = tf.layers.conv2d(pool3, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4')
    conv4 = tf.layers.conv2d(conv4, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv4-2')
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)    

    conv5 = tf.layers.conv2d(pool4, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5')
    conv5 = tf.layers.conv2d(conv5, 512, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv5-2')
 
    up6 = tf.layers.conv2d_transpose(conv5, 256, [3, 3], strides=2, padding="SAME", name='Up6')
    up6 = tf.concat([up6, conv4], 3)
    conv6 = tf.layers.conv2d(up6, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6')
    conv6 = tf.layers.conv2d(conv6, 256, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv6-2')
    
    up7 = tf.layers.conv2d_transpose(conv6, 128, [3, 3], strides=2, padding="SAME", name='Up7')
    up7 = tf.concat([up7, conv3], 3)
    conv7 = tf.layers.conv2d(up7, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv7')
    conv7 = tf.layers.conv2d(conv7, 128, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv7-2')

    up8 = tf.layers.conv2d_transpose(conv7, 64, [3, 3], strides=2, padding="SAME", name='Up8')
    up8 = tf.concat([up8, conv2], 3)
    conv8 = tf.layers.conv2d(up8, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv8')
    conv8 = tf.layers.conv2d(conv8, 64, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv8-2')
 
    up9 = tf.layers.conv2d_transpose(conv8, 32, [3, 3], strides=2, padding="SAME", name='Up9')
    up9 = tf.concat([up9, conv1], 3)
    conv9 = tf.layers.conv2d(up9, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv9')
    conv9 = tf.layers.conv2d(conv9, 32, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=he_init, name='Conv9-2')
    
    logits = tf.layers.conv2d(conv9, NUM_OUTPUTS, [1, 1], padding="SAME", activation=None, kernel_initializer=he_init, name='Output')
    prediction = tf.nn.softmax(logits)
    return logits, prediction

# Define loss and optimizer
logits, prediction = unet(X) #unet_simple(X) # network(X) 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
trainer = optimizer.minimize(loss)

# Evaluate model
segmentation = tf.argmax(prediction, 3)
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
#plt.show()
plt.savefig('results/segmentation-accuracy.png')

# Show results
segmentation = sess.run(segmentation, feed_dict={ X: data.x_test, Y: data.y_test })
print(segmentation.shape)
index = 0;
matplotlib.image.imsave('results/real-img.png', data.x_test[index], cmap='gray') 
matplotlib.image.imsave('results/real-test.png', data.y_test[index][:,:,1], cmap='gray') 
matplotlib.image.imsave('results/real-results.png', segmentation[index], cmap='gray') 