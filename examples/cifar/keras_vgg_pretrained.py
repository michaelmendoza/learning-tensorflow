
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot

vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')