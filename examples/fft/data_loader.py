
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DataLoader:

    def __init__(self):
        self.load()

    def get(self):
        input = np.reshape(self.input, (-1, self.WIDTH * self.HEIGHT * 2))
        output = np.reshape(self.output, (-1, self.WIDTH * self.HEIGHT * 2))
        return input, output

    def load(self):

        filepath = '.'
        filename = 'shepp256.png'
        img = mpimg.imread(os.path.join(filepath, filename))
        img = np.array(img)

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
