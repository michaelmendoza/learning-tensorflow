
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, transform

class DataLoader:

    def __init__(self, height = 128, width = 128):
        self.load(height, width)

    def get(self):
        return self.input[None,...], self.output[None,...]

    def get_flatten_data(self):
        # Convert to array shape = (1, WIDTH * HEIGHT * 2)
        input = np.reshape(self.input, (-1, self.WIDTH * self.HEIGHT * 2))
        output = np.reshape(self.output, (-1, self.WIDTH * self.HEIGHT * 2))
        return input, output

    def generate(self, samples, height, width):
        img = np.random.randn(samples, height, width) + 1j* np.random.randn(samples, height, width) 
        
        kSpace = np.fft.ifftshift(np.fft.fft2(img))
        inverse = np.fft.ifft2(kSpace)
        
        inputs = np.stack((np.abs(kSpace), np.angle(kSpace)), axis = 3)
        outputs = np.stack((np.abs(inverse), np.angle(inverse)), axis = 3)
        return inputs, outputs

    def load(self, height, width):

        filepath = './examples/fft/'
        filename = 'shepp256.png'
        img = mpimg.imread(os.path.join(filepath, filename))
        img = np.array(img)
        img = transform.resize(img, (height, width), mode='constant')

        self.WIDTH = img.shape[0]
        self.HEIGHT = img.shape[1]
        self.CHANNELS = 2

        kSpace = np.fft.ifftshift(np.fft.fft2(img))
        inverse = np.fft.ifft2(kSpace)
        
        self.input = np.dstack((np.abs(kSpace), np.angle(kSpace)))
        self.output = np.dstack((np.abs(inverse), np.angle(inverse)))

    def show(self):

        input = self.input[:,:,0] * np.exp(1j*self.input[:,:,1])
        output = self.output[:,:,0] * np.exp(1j*self.output[:,:,1])

        plt.subplot(2, 1, 1)
        plt.imshow(np.abs(input), cmap='gray')

        plt.subplot(2, 1, 2)
        plt.imshow(np.abs(output), cmap='gray')
        plt.show()

    def info(self):
        print(self.input.dtype)

if __name__ == '__main__':
    data = DataLoader()
    data.info()
    data.show()
    x, y = data.get()
    print(x.shape, y.shape)
