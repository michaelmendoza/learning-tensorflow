import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

N = 100

def gen():
    # Create dataset
    data = lambda: None
    data.x = np.linspace(0, 3, N)
    data.y = np.exp(data.x) + 2 * np.random.rand(N)
    data.x = np.reshape(data.x, (-1, 1))
    data.y = np.reshape(data.y, (-1, 1))
    return data

# Training Parameters
learning_rate = 0.01
num_steps = 1000

# Setup Network
X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

# Define Network
fc1 = tf.layers.dense(X, 100, activation=tf.nn.relu, name='fc1')
fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='fc2')
Y_pred = tf.layers.dense(fc2, 1, name='out')

# Define loss and optimizer
loss = tf.square(Y_pred - Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initalize varibles, and run network 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train 
for step in range(num_steps):
    data = gen()
    sess.run(optimizer, feed_dict={X:data.x, Y:data.y})

_x = np.reshape(np.linspace(0, 3, 10 * N), (-1,1))
_y = sess.run(Y_pred, feed_dict={X:_x})

# Reshape
x = np.reshape(data.x, (-1)); y = np.reshape(data.y, (-1))
_y = np.reshape(_y, (-1)); _x = np.reshape(_x, (-1))

# Plot Results
plt.scatter(x, y)
plt.plot(_x, _y, 'r')
plt.show()
