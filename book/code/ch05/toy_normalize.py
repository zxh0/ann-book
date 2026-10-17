import numpy as np

# 手写实现，用于逐步演示归一化的计算过程。
# NumPy 有现成实现：vec_x / vec_x.sum()
def normalize(vec_x: np.ndarray) -> np.ndarray:
    total = sum(vec_x)
    return np.array([v / total for v in vec_x])

print(normalize(np.array([0.496, 0.254, 0.131, 0.067, 0.0, 0.0, 0.0])).round(3))
print(normalize(np.array([0.496, 0.254, 0.131, 0.0, 0.0, 0.0])).round(3))
print(normalize(np.array([0.496, 0.254, 0.0,  0.0, 0.0, 0.0])).round(3))
print(normalize(np.array([0.496, 0.0, 0.0,  0.0, 0.0, 0.0])).round(3))

# [0.523 0.268 0.138 0.071 0.    0.    0.   ]
# [0.563 0.288 0.149 0.    0.    0.   ]
# [0.661 0.339 0.    0.    0.    0.   ]
# [1. 0. 0. 0. 0. 0.]
