# Numpy basics and handy functions part 2!

import numpy as np

# identifying datatype
arr = np.array([[1.0, 2, 3], [1, 2, 3]])
print(arr.dtype)

# changing datatype
arr1 = arr.astype('int32')

# assigning datatype
arr2 = np.array([[1.0, 2, 3], [1, 2, 3]], dtype=np.uint8)

# length of the array
print(len(arr2.shape))
print(arr2.shape)

# reshape the array
arr3 = arr.reshape([1, 6])
print(arr3.shape)

# creating a random array
arr5 = np.random.randn(8, 8)
arr6 = arr5.reshape(32, 2) # reshaping

# expanding the array to 3D
arr7 = arr5.reshape(-1, 2, 1, 1, 1, 1) # from above array arr5
print(arr7.shape)

# Ravel - changing the dimension to 1
arr8 = arr7.ravel()
# this is used to flatten the layer, e.g. when reducing dimension from 4D to 2D
print(arr8.shape)

# when want to increase the dimension but hold the values
arr9 = np.expand_dims(arr8, 0)
print(arr9.shape) # (1, 64)
arr10 = np.expand_dims(arr8, -1)
print(arr10.shape) # (64, 1)