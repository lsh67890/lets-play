import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# call MNIST dataset
from tensorflow.keras import datasets
mnist = datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_x.shape)

# look at the image dataset
img = train_x[0]
plt.imshow(img, 'gray')
plt.show()
plt.close()

# increase data dimension
train_x = np.expand_dims(train_x, -1) # using np
print(train_x.shape)
new_train_x = tf.expand_dims(train_x, -1) # using tf
new_train_x = train_x[..., tf.newaxis] # using tf.newaxis
reshaped = train_x.reshape([60000, 28, 28, 1])

# view the image
disp = new_train_x[1, :, :, 0] # match the new dimension
disp = np.squeeze(new_train_x[0])
plt.imshow(disp, 'gray')
plt.show()
plt.close()

# look at the label dataset
plt.title(train_y[0])
plt.imshow(train_x[0], 'gray')
plt.show()
plt.close()

# OneHot encoding
# convert the label so the computer understands, e.g. [0, 0, 1, 0, 0]
# meaning it's classified as 2
from tensorflow.keras.utils import to_categorical
print(to_categorical(1, 5))
label = train_y[0]
label_onehot = to_categorical(label, num_classes=10)
print(label, label_onehot)
plt.title(label_onehot)
plt.imshow(train_x[0], 'gray')
plt.show()
plt.close()
