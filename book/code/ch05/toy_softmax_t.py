import numpy as np
from math import e

# 手写实现，用于逐步演示带温度的 Softmax 的计算过程。
# NumPy 没有现成的 Softmax，但 np.exp 可以一次算完整个向量，
# 于是整个函数可以简写成一行：np.exp(vec_z/t) / np.exp(vec_z/t).sum()
def softmax(vec_z: np.ndarray, t: float) -> np.ndarray:
    vec_exp = [e ** (v/t) for v in vec_z]
    total = sum(vec_exp)
    return np.array([v / total for v in vec_exp])

# 示例
vec_data = np.array([3, 2, 1])
print(softmax(vec_data, 0.1).round(3))
print(softmax(vec_data, 0.5).round(3))
print(softmax(vec_data, 1.0).round(3))
print(softmax(vec_data, 1.5).round(3))
print(softmax(vec_data, 999).round(3))

# [1. 0. 0.]
# [0.867 0.117 0.016]
# [0.665 0.245 0.09 ]
# [0.563 0.289 0.148]
# [0.334 0.333 0.333]

vec_data = np.array([3, 1, 6, 2, 5, 4])
print((vec_data / 1.5).round(3))
# [2.    0.667 4.    1.333 3.333 2.667]

print(softmax(vec_data, 1.5).round(3))
# [0.067 0.018 0.496 0.034 0.254 0.131]
