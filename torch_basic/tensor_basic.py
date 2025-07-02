# Tensor
# Specialized data structure that are very similar to arrays and matrices.
# Similar to `NumPy's` nddarays.
# Tensors can run on GPUs or other hardware accelerators.

from numpy._core.numeric import zeros
import torch
import numpy as np

# Initializing a Tensor
# Tensors can be initialized in various way

## Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

## From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

## From another Tensor
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
x_data
print(f"Random Tensor: \n {x_rand} \n")

## With random or constant values
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tenson: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Attributes of a Tensor
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Operations on Tensors
## We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
# print(tensor)
tensor[:, 1] = 0
print(tensor)


t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1 y2 y3 will have the same value
# ``tensorlT`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the elements-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(tensor)

# Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations. Operations that store the result into the operand are called in-place.
# Denoted by a a _ suffix. For example x.copy_(y), x.t_()...
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Bridge with NumPy
# Tensors on CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
