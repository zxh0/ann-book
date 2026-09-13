import numpy as np

# 手写实现，用于逐步演示加权求和的计算过程。
# NumPy 有现成实现：np.sum(mat_x * mat_w)
def weighted_sum(mat_x: np.ndarray,
                 mat_w: np.ndarray,
                 k: int) -> float:
    total = 0.0
    for i in range(k):
        for j in range(k):
            total += mat_x[i][j] * mat_w[i][j]
    return total


mat_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mat_w = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
k = 3
print(weighted_sum(mat_x, mat_w, k)) # 5.0
