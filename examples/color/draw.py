
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math
import cairo

WIDTH = 64
HEIGHT = 64
CHANNELS = 3

# Draw a random Grey Image
img = 255 * np.random.rand(WIDTH, HEIGHT)
print(img)
plt.imshow(img, cmap='gray')
plt.show()

# Draw a random Color Image
img = 255 * np.random.rand(WIDTH, HEIGHT, CHANNELS)
print(img)
plt.imshow(img)
plt.show()

# Draw random Red boxes
data = np.zeros( (HEIGHT,WIDTH, 4), dtype=np.uint8 ) 
surface = cairo.ImageSurface.create_for_data( data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT )
ctx = cairo.Context( surface )

ctx.scale (WIDTH, HEIGHT) # Normalizing the canvas
ctx.set_source_rgb(0, 0, 0)
ctx.rectangle (0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1) 
ctx.fill()

ctx.set_source_rgb(0, 0, 1)
for _ in range(50):
    r = np.random.rand(2)
    ctx.translate (r[0], r[1])      # Changing the current transformation matrix
    ctx.rectangle (0, 0, 0.1, 0.1)  # Rectangle(x0, y0, x1, y1)
    ctx.fill()
    ctx.translate (-r[0], -r[1])    # Changing the current transformation matrix

img = data[:,:,0:3]
plt.imshow(img)
plt.show()

