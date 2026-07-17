import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[2, 3, 4], [5, 6, 7]])
c = np.array([[1, 2], [3, 4], [5, 6]])

print(a.shape) # (2, 3)
print(c.shape) # (3, 2)
print(a + 1)   # [[ 2  3  4] [ 5  6  7]]
print(a * 2)   # [[ 2  4  6] [ 8 10 12]]
print(a + b)   # [[ 3  5  7] [ 9 11 13]]
print(a - b)   # [[-1 -1 -1] [-1 -1 -1]]
print(a * b)   # [[ 2  6 12] [20 30 42]]
print(b * a)   # [[ 2  6 12] [20 30 42]]
print(a @ c)   # [[22 28] [49 64]]
print(c @ a)   # [[ 9 12 15] [19 26 33] [29 40 51]]
print(a.T)     # [[1 4] [2 5] [3 6]]
