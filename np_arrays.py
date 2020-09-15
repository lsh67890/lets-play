# How to

import numpy as np

# 0-d
arr0 = np.array(5)
print(arr0.shape)
print(arr0.ndim)

# 1-d
arr1 = np.array([1, 2])
print(arr1.shape)
print(arr1.ndim)

# 2-d
arr2 = np.array([[1, 2, 3], [1, 2, 3]])
print(arr2.shape)
print(arr2.ndim)

# 4-d
arr4 = np.array([[[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
print(arr4.shape)
print(arr4.ndim)