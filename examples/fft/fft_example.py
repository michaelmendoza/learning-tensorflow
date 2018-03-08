
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


filepath = '.'
filename = 'shepp256.png'
img = mpimg.imread(os.path.join(filepath, filename))
img = np.array(img)

kSpace = np.fft.ifftshift(np.fft.fft2(img))
inverse = np.fft.ifft2(kSpace)

plt.subplot(2, 1, 1)
plt.imshow(np.abs(kSpace), cmap='gray')

plt.subplot(2, 1, 2)
plt.imshow(np.abs(inverse), cmap='gray')
plt.show()

