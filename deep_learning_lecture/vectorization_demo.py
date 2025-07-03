import numpy as np

a = np.array([1, 2, 3, 4])
print(a)

import time

# Vectorized version

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
print("Vectorized version: " + str(1000 * (toc - tic)) + "ms")

# Non Vectorized

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()

print(c)
print("Non Vectorized version: " + str(1000 * (toc - tic)) + "ms")
