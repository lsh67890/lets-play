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

# check weights and visualise
weight = img_layer.get_weights()
print(weight[0].shape, weight[1].shape)
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.hist(output.numpy().ravel(), range=[-2,2])
plt.ylim(0, 500)
plt.subplot(132)
plt.title(weight[0].shape)
plt.imshow(weight[0][:,:,0,0], 'gray')
plt.subplot(133)
plt.title(output.shape)
plt.imshow(output[0,:,:,0], 'gray')
plt.colorbar()
plt.show()
plt.close()

# activation function
print(tf.keras.layers.ReLU())
act_layer = tf.keras.layers.ReLU()
act_output = act_layer(output)
print(act_output)
print(output.shape)
print(np.min(act_output), np.max(act_output)) # compare this with output min and max
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.hist(act_output.numpy().ravel(), range=[-2,2])
plt.ylim(0, 100)
plt.subplot(122)
plt.title(act_output.shape)
plt.imshow(act_output[0,:,:,0], 'gray')
plt.show()
plt.close()

# pooling
pool_layer = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')
pool_output = pool_layer(act_output)
print(act_output.shape)
print(pool_output.shape)
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.hist(pool_output.numpy().ravel(), range=[-2,2])
plt.ylim(0,100)
plt.subplot(122)
plt.title(pool_output.shape)
plt.imshow(pool_output[0,:,:,0], 'gray')
plt.colorbar()
plt.show()
plt.close()

# flatten
layer = tf.keras.layers.Flatten()
flatten = layer(output)
print(flatten.shape) # 1 indicates the batch size
plt.figure(figsize=(10,5))
plt.subplot(211)
plt.hist(flatten.numpy().ravel())
plt.subplot(212)
plt.imshow(flatten[:,:100])
plt.show()
plt.close()

# dense
layer = tf.keras.layers.Dense(32, activation='relu')
dense = layer(flatten)
print(dense.shape) # shape will be reduced to 32

# dropout
layer = tf.keras.layers.Dropout(0.7) # the ratio to temporarily disconnect
dropout = layer(output)
print(dropout.shape)