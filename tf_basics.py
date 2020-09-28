import numpy as np
import tensorflow as tf

# create an array
# both tuple or list and np.array() or array works
arr = np.array([1, 2, 3])
print(arr.shape)
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr1.shape)

# create a tensor
tf1 = tf.constant([1, 2, 3]) # or tf.constant(arr)
print(tf1)
tf2 = tf.constant(((1, 2, 3), (4, 5, 6))) # or tf.constant(arr1)
print(tf2)

# assign datatype
arr3 = tf.constant([2, 3, 4], dtype=tf.float32)
print(arr3.shape)
arr4 = np.array([1, 2, 3], dtype=np.float32)
arr4.astype(np.uint8) # to change datatype
tf.cast(arr3, dtype=tf.uint8) # like astype in np, in tf we use cast

# call numpy from tensor
arr3.numpy()
print(type(arr3.numpy()))

# create a random array
np_norm = np.random.randn(9)
print(np_norm)
tf_norm = tf.random.normal([3, 3]) # for normal dist. define shape
print(tf.norm)
tf_uni = tf.random.uniform([3, 3]) # for uniform dist.
print(tf_uni)