
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math
import cairo

WIDTH = 128
HEIGHT = 128
CHANNELS = 3

class DataGenerator:

    def __init__(self):
        self.size = 1000
        self.ratio = 0.8
        self.threshold = [128, 0, 0]

        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CHANNELS = CHANNELS

        self.generate();

    def generate_image(self):
        ''' Randomly generates an image with random boxes '''    

        data = np.zeros( (HEIGHT,WIDTH, 4), dtype=np.uint8 ) 
        surface = cairo.ImageSurface.create_for_data( data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT )
        ctx = cairo.Context( surface )

        ctx.scale (WIDTH, HEIGHT) # Normalizing the canvas
        ctx.set_source_rgb(0, 0, 0)
        ctx.rectangle (0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1) 
        ctx.fill()

        # Create random colored boxes
        for _ in range(50):
            rc = np.random.rand(3)
            ctx.set_source_rgb(rc[0], rc[1], rc[2])

            r = np.random.rand(2)
            ctx.translate (r[0], r[1])      # Changing the current transformation matrix
            ctx.rectangle (0, 0, 0.1, 0.1)  # Rectangle(x0, y0, x1, y1)
            ctx.fill()
            ctx.translate (-r[0], -r[1])    # Changing the current transformation matrix

        # Create a randomly placed red box
        ctx.set_source_rgb(0, 0, 1)
        r = np.random.rand(2)
        ctx.translate (r[0], r[1])      # Changing the current transformation matrix
        ctx.rectangle (0, 0, 0.1, 0.1)  # Rectangle(x0, y0, x1, y1)
        ctx.fill()
        ctx.translate (-r[0], -r[1])

        img = data[:,:,0:3]
        return img;

    def whiten_data(self, features):
        """ whiten dataset - zero mean and unit standard deviation """
        features = np.reshape(features, (self.size, WIDTH * HEIGHT * CHANNELS))
        features = (np.swapaxes(features,0,1) - np.mean(features, axis=1)) / np.std(features, axis=1)
        features = np.swapaxes(features,0,1)
        features = np.reshape(features, (self.size, WIDTH, HEIGHT, CHANNELS))
        #features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        return features

    def unwhiten_img(self, img): 
        """ remove whitening for a single image """ 
        img = np.reshape(img, (WIDTH * HEIGHT * CHANNELS))
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) 
        img = np.reshape(img, (WIDTH, HEIGHT, CHANNELS))
        return img

    def generate(self):
        ''' Generates a randomly generated dataset '''
        img = self.generate_image()
        self.data = np.stack( (img, self.generate_image()))
        for _ in range(self.size - 2):
            img = self.generate_image()
            self.data = np.concatenate( (self.data, img[None,:]), axis=0)
        
        # Generate truth data
        self.label = np.all(np.greater_equal(self.data, self.threshold), axis=3) * 1.0;
        self.label = np.reshape(self.label, (self.size, WIDTH, HEIGHT, 1))
        self.label = np.concatenate( (1 - self.label, self.label), axis=3) # Index 0: Incorrect, Index 1: Correct

        # Setup data 
        self.data = self.whiten_data(self.data)

        # Split data into test/training sets
        index = int(self.ratio * len(self.data)) # Split index
        self.x_train = self.data[0:index, :].astype(np.float32)
        self.y_train = self.label[0:index].astype(np.float32)
        self.x_test = self.data[index:,:].astype(np.float32)
        self.y_test = self.label[index:].astype(np.float32)

    def show(self, index):
        ''' Show a data slice at index'''
        img = self.unwhiten_img( self.data[index] )
        plt.imshow(img)
        plt.show()

    def show_label(self, index):
        ''' Show a truth data slice at index'''
        img = self.label[index]
        plt.imshow(img[:,:,0], cmap='gray')
        plt.show()  
        plt.imshow(img[:,:,1], cmap='gray')
        plt.show()  

    def print(self):
        print("Data Split: ", self.ratio)
        print("Train => x:", self.x_train.shape, " y:", self.y_train.shape)
        print("Test  => x:", self.x_test.shape, " y:", self.y_test.shape)

    def next_batch(self, batch_size):
        ''' Retrieves the next batch for a given batch size '''
        length = self.x_train.shape[0]
        indices = np.random.randint(0, length, batch_size) # Grab batch_size values randomly
        return [self.x_train[indices], self.y_train[indices]]

if __name__ == '__main__':
    data = DataGenerator()
    data.print()
    data.show(2)
    data.show_label(2)
