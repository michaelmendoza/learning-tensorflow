
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math
import cairo

class DataGenerator:

    def __init__(self, size = 1000):

        self.WIDTH = 128
        self.HEIGHT = 128
        self.CHANNELS = 1

        self.size = size
        self.minAngle = 45;
        self.maxAngle = 135;

        print("Generating angle image data ....", end="", flush=True)
        self.generate()
        print("Generation Complete")
        print("Data size: Input Data", self.x_train.shape, " Truth Data:", self.y_train.shape, "Angle Data", self.angles_train.shape);

    def generate(self):
        self.x_train, self.angles_train = self.generate_image_set()
        self.x_test, self.angles_test = self.generate_image_set() 

        # Threshold for Acute Angle i.e. angle less than 90
        self.y_train = np.less(self.angles_train, 90) * 1 
        self.y_test = np.less(self.angles_test, 90) * 1

        self.y_train = self.one_hot(self.y_train)
        self.y_test = self.one_hot(self.y_test)

        # Reshape angle data
        self.angles_train = np.reshape(self.angles_train, (-1, 1))
        self.angles_test = np.reshape(self.angles_test, (-1, 1))

    def one_hot(self, data):
        # Create one-hot vectors for truth data
        data = np.reshape(data, (-1, 1))
        return np.concatenate( (1 - data, data ), axis=1) 

    def randAngles(self):
        return self.minAngle + np.random.rand(self.size) * (self.maxAngle - self.minAngle);

    def generate_image_set(self):
        angles = self.randAngles()
        img = self.generate_image(angles[0])[None,:]
        for _ in range(1, self.size):
            img = np.concatenate( (img, self.generate_image(angles[_])[None,:]), axis=0)
        return (img, angles)

    def generate_image(self, angle):
        data = np.zeros( (self.HEIGHT, self.WIDTH, 4), dtype=np.uint8 ) 
        surface = cairo.ImageSurface.create_for_data( data, cairo.FORMAT_ARGB32, self.WIDTH, self.HEIGHT )
        ctx = cairo.Context( surface )

        ctx.scale (self.WIDTH, self.HEIGHT) # Normalizing the canvas
        ctx.set_source_rgb(0, 0, 0)
        ctx.rectangle (0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1) 
        ctx.fill()

        # Create a randomly placed red box
        x0 = 0.5
        y0 = 0.4

        ctx.set_source_rgb(1, 1, 1)
        ctx.move_to(1.0, 1.0 - y0)
        ctx.line_to(x0, 1.0 - y0)
        x = 1.0 / math.tan(angle * math.pi / 180)
        ctx.line_to(x+x0, 0)
        ctx.set_line_width(0.1)
        ctx.stroke()

        img = data[:,:,0]
        img = np.reshape(img, (self.WIDTH, self.HEIGHT, self.CHANNELS))
        return img

    def next_batch(self, batch_size):
        ''' Retrieves the next batch for a given batch size '''
        length = self.x_train.shape[0]
        indices = np.random.randint(0, length, batch_size) # Grab batch_size values randomly
        return [self.x_train[indices], self.y_train[indices], self.angles_train[indices]]

if __name__ == '__main__':
    data = DataGenerator()
