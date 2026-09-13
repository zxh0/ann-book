import numpy as np
from math import e

# 手写实现，用于逐步演示计算过程。
# NumPy 没有现成的 Softmax，但 np.exp 可以一次算完整个向量，
# 于是整个函数可以简写成一行：np.exp(vec_z) / np.exp(vec_z).sum()
def softmax(vec_z: np.ndarray) -> np.ndarray:
    vec_exp = [e ** v for v in vec_z]
    total = sum(vec_exp)
    return np.array([v / total for v in vec_exp])

# 示例
vec_data = np.array([2.0, 1.0, 0.1])
print(softmax(vec_data))
# [0.65900114 0.24243297 0.09856589]
