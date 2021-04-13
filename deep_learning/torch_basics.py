import torch
import numpy as np

nums = torch.arange(9) # creates 1x9 tensor
print(nums)
print(nums.reshape(3, 3))
randoms = torch.rand((3, 3))
print(randoms)
zeros = torch.zeros((3, 3))
print(zeros)
ones = torch.ones((3, 3))
print(ones)
print(torch.zeros_like(ones))

# operations
result = torch.add(nums, 10)
print(result)

# view
range_nums = torch.arange(9).reshape(3, 3)
print(range_nums)
print(range_nums.view(1, 9))

# slice and index
print(nums[1])
print(nums[1:])

# compile
arr = np.array([1, 1, 1])
arr_torch = torch.from_numpy(arr)
print(arr_torch.float())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device, arr_torch.to(device))

# AutoGrad
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)
back = out.backward()
print(x.grad)
print(x.requires_grad, (x ** 2).requires_grad)
with torch.no_grad(): # this is faster
    print((x ** 2).requires_grad)
