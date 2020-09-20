# Some numpy basic (and handy!) functions

import numpy as np

# initialise an array with 3x3 matrix zeros
zeros = np.zeros([3, 3])

# initialise an array with 10x2 ones
ones = np.ones([10, 2])
ones1 = ones * 2

# array with consecutive numbers 0-4
ara1 = np.arange(5)

# array with consecutive numbers 4-9
ara2 = np.arange(4, 9)

# 3x3 reshaped array with consecutive numbers 0-8
ara3 = np.arange(9).reshape(3, 3)

# use index to get 5 from ara3
num1 = ara3[1][2]
num2 = ara3[1, 2]

# use slice to get two arrays with the numbers 3-8 from ara3
sli1 = ara3[1:]

# use slice to get two arrays with 4,5 and 7,8 from ara3
sli2 = ara3[1:, 1:]

# boolean index
bool1 = np.random.randn(3, 3) # generates random numbers for 3x3
bool2 = bool1 <= 0 # to check which numbers are less/equal to 0
bool1[bool1 <= 0] = 1 # set the numbers less/equal to 0 to 1

# broadcast - reshaping during calculation according to the shape
bro = np.arange(9).reshape(3, 3)
bro1 = bro + 3 # adds 3 to all values
bro2 = bro * 3 # multiplies 3 to all values
bro3 = bro + np.array([1, 1, 1]) # adds the array [1, 1, 1]

# math functions
mat = np.arange(9).reshape(3, 3)
mat1 = mat + mat # adds two arrays
mat2 = np.multiply(mat, 2) # multiplies mat1 with 2
mat3 = np.max(mat + mat) # gets the max value of mat2
mat4 = np.min(mat + mat) # gets the min value of mat2
mat5 = np.sum(mat, -1) # sums up mat1 on axis/dimension -1
mat6 = np.mean(mat) # gets the mean value of mat1

# get the index of a value using argmax/argmin
argm = np.array([1, 6, 3, 7, 3, 2, 9, 0])
argm1 = np.argmax(argm) # gets the index of max value, 9
argm2 = np.argmin(argm) # gets the index of min value, 0

# unique values in the array
uni = np.array([1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5])
uni1 = np.unique(uni)

# print() the array you want to see :)