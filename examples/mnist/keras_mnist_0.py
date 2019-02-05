
'''
Basic Keras Code for a single-layer neural network
'''
 
# Training Parameters
epochs = 10

# Network Parameters
NUM_INPUTS = 784
NUM_OUTPUTS = 10 
 
import time
import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
 
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(-1, NUM_INPUTS)
x_test = x_test.reshape (-1, NUM_INPUTS)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(NUM_OUTPUTS, activation='softmax', input_dim=NUM_INPUTS))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

start = time.time()
history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, shuffle=True)
evaluation = model.evaluate(x_test, y_test, verbose=1)
end = time.time()

print('Summary: Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)) )

# Plot Accuracy 
plt.plot(history.history["categorical_accuracy"])
plt.plot(history.history["val_categorical_accuracy"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Train Accuracy", "Test Accuracy"], loc="upper left")
plt.show();
