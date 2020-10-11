import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets

# prepare MNIST dataset
mnist = datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

input_shape = (28, 28, 1)
num_classes = 10

# feature extraction
inputs = layers.Input(shape=input_shape)
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D(pool_size=(2,2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D(pool_size=(2,2))(net)
net = layers.Dropout(0.25)(net)

# fully connected
net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(10)(net) # num_classes
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')

# LOSS FUNCTION
# categorical vs binary
bloss = 'binary_crossentropy'
closs = 'categorical_crossentropy'
loss_fun = tf.keras.losses.sparse_categorical_crossentropy
print(tf.keras.losses.categorical_crossentropy)
print(tf.keras.losses.binary_crossentropy)

# metrics - this is how to evaluate the model
# there are three ways of metrics
metrics = tf.keras.metrics.Accuracy() # metrics should be a list
print(tf.keras.metrics.Precision())
print(tf.keras.metrics.Recall())
# metrics = ['accuracy']

# COMPILE
# apply optimiser - 'sgd', 'rmsprop', 'adam'
optm = tf.keras.optimizers.Adam()
model.compile(optimizer=optm, loss=loss_fun, metrics=metrics)

# PREPARE DATASET
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
# add additional dimension
np.expand_dims(train_x, -1)
tf.expand_dims(train_x, -1)
train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]
print(train_x.shape, test_x.shape)
print(np.min(train_x), np.max(train_x)) # we will rescale this
train_x = train_x / 255.
test_x = test_x / 255.
print(np.min(train_x), np.max(train_x))

# TRAINING
# set the hyperparameter for training
num_epochs = 1
batch_size = 32
model.fit(train_x, train_y, batch_size=batch_size, shuffle=True, epochs=num_epochs)
