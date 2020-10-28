import os
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers, datasets
import numpy as np
import matplotlib.pyplot as plt

data_paths = glob('dataset/mnist_png/training/0/*.png')
# tf.io.matching_files(PATH) does the same as above
path = data_paths[0]

gfile = tf.io.read_file(path)
img = tf.io.decode_image(gfile)
plt.imshow(img[:, :, 0], 'gray')
plt.show()
plt.close()

# set data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator( # handles loading image and preprocessing
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
inputs = img[tf.newaxis, ...] # check shape
img = next(iter(datagen.flow(inputs)))
plt.subplot(121) # compare with the input image
plt.title('Original Image')
plt.imshow(np.squeeze(inputs), 'gray')
plt.subplot(122)
plt.title('Transformed Image')
plt.imshow(np.squeeze(img), 'gray')
plt.show()
plt.close()

# transformation
datagen = ImageDataGenerator(
    width_shift_range=0.3, # how it will randomly move width-wise
    zoom_range=0.7, # it zooms both width and height
    # preprocessing_function= uses lambda function
)
train_datagen = ImageDataGenerator(
    zoom_range=0.7,
    rescale=1./255. # always rescale in BOTH train and test datagen!
)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_dir = 'dataset/mnist_png/training'
test_dir = 'dataset/mnist_png/testing'

# hyperparameter tuning
num_epochs = 10
batch_size = 32
learning_rate = 0.001
dropout_rate = 0.7
input_shape = (28, 28, 1)
num_classes = 10

# preprocess
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255.
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2], # (28, 28)
    batch_size=batch_size,
    color_mode='grayscale', # default is rgb
    class_mode='categorical', # or binary
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2], # (28, 28)
    batch_size=batch_size,
    color_mode='grayscale', # default is rgb
    class_mode='categorical', # or binary
)

# build model
inputs = layers.Input(input_shape)
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(dropout_rate)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(dropout_rate)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# training
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)