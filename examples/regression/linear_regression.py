import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Create dataset
N = 400
data = lambda: None
data.x = np.linspace(0, 1, N)
data.y = np.exp(data.x) + 0.5 * np.random.rand(N)
plt.scatter(data.x, data.y)

# Training Parameters
learning_rate = 0.001
num_steps = 100

# Setup Network
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Define Network
w = tf.Variable(0.0, name='w')
b = tf.Variable(0.0, name='b')
Y_pred = X * w + b

# Define loss and optimizer
loss = tf.square(Y_pred - Y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Initalize varibles, and run network 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train 
for step in range(num_steps):
    sess.run(optimizer, feed_dict={X:data.x, Y:data.y})

w, b = sess.run([w, b])
print(w, b)

# Plot Results
plt.plot(data.x, w * data.x + b, 'r')
plt.show()
