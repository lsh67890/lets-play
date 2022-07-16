import tensorflow as tf
import numpy as np

epochs = 10

# define model
# similar to fully connected layer
class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        maxpool = tf.keras.layers.MaxPool2D
        self.sequence = list() # sequence was useful for back propagation
        # (3, 3) is the size of kernel. padding = same to keep the original size
        # 28 x 28 x 16
        self.sequence.append(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
        self.sequence.append(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
        self.sequence.append(maxpool((2, 2))) # 14 x 14 x 16
        self.sequence.append(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.sequence.append(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.sequence.append(maxpool((2, 2))) # 7 x 7 x 32
        self.sequence.append(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.sequence.append(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.sequence.append(tf.keras.layers.Flatten()) # 1568
        self.sequence.append(tf.keras.layers.Dense(2048, activation='relu'))
        self.sequence.append(tf.keras.layers.Dense(10, activation='softmax'))

    def call(self, x, training=False, mask=None):
        for layer in self.sequence:
            x = layer(x)
        return x

# dataset preparation
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

# create model
model = ConvNet()

# define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# training loop
for epoch in range(epochs):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'epoch: {}, loss: {}, accuracy: {}, test loss: {}, test accuracy: {}'
    print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100,
                          test_loss.result(), test_accuracy.result()*100))