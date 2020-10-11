import matplotlib.pyplot as plt
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

# add chanel dimension
train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]
# data normalisation
train_x, test_x = train_x / 255.0, test_x / 255.0

# using tf.data
train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.shuffle(1000) # 1000 is the buffer size. the higher, the longer to load
train_ds = train_ds.batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds = test_ds.batch(32)

# visualise data
for img, label in train_ds.take(2):
    plt.title(str(label[0]))
    plt.imshow(img[0, :, :, 0], 'gray')
    plt.show()
    plt.close()

# training (keras) - we can input train_ds as it is since it's a generator
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_ds, epochs=10000)

# optimisation
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optm = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# training
@tf.function
def train_step(imgs, labels):
    with tf.GradientTape() as tape:
        predictions = model(imgs)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optm.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(imgs, labels):
    predictions = model(imgs)
    t_loss = loss_obj(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

for epoch in range(2):
    for imgs, labels in train_ds:
        train_step(imgs, labels)
    for test_imgs, test_labels in test_ds:
        test_step(test_imgs, test_labels)
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(), train_accuracy.result() * 100,
                          test_loss.result(), test_accuracy.result() * 100)
          )