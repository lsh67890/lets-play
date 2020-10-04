import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
img = train_x[0]
print(img.shape) # always check the shape and nr of channel

# target dimension: [batch_size, height, width, channel]
plt.imshow(img, 'gray')
img = img[tf.newaxis, ..., tf.newaxis]
print(img.shape)

# convolution layers
conv_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='SAME', activation='relu')
print(conv_layer)
conv_layer = tf.keras.layers.Conv2D(3, 3, 1, 'SAME') # this also works
print(conv_layer)

# visualisation
img = tf.cast(img, dtype=tf.float32) # make sure it has float32 as datatype
img_layer = tf.keras.layers.Conv2D(3, 3, 1, padding='SAME')
output = img_layer(img)
print(output)

# compare the original image with the convoluted image
print(np.min(img), np.max(img))
print(np.min(output), np.max(output))
plt.subplot(1, 2, 1)
plt.imshow(img[0, :, :, 0], 'gray')
plt.subplot(1, 2, 2)
plt.imshow(output[0, :, :, 0], 'gray')
plt.show()
plt.close()

