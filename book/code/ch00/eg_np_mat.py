import numpy as np

mat_a = np.array([[1, 2, 3], [4, 5, 6]])
mat_b = np.array([[2, 3, 4], [5, 6, 7]])
mat_c = np.array([[1, 2], [3, 4], [5, 6]])

print(mat_a.shape)   # (2, 3)
print(mat_c.shape)   # (3, 2)
print(mat_a + 1)     # [[ 2  3  4] [ 5  6  7]]
print(mat_a * 2)     # [[ 2  4  6] [ 8 10 12]]
print(mat_a + mat_b) # [[ 3  5  7] [ 9 11 13]]
print(mat_a - mat_b) # [[-1 -1 -1] [-1 -1 -1]]
print(mat_a * mat_b) # [[ 2  6 12] [20 30 42]]
print(mat_b * mat_a) # [[ 2  6 12] [20 30 42]]
print(mat_a @ mat_c) # [[22 28] [49 64]]
print(mat_c @ mat_a) # [[ 9 12 15] [19 26 33] [29 40 51]]
print(mat_a.T)       # [[1 4] [2 5] [3 6]]
