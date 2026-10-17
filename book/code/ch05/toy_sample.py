import numpy as np
import random

# 手写实现，用于逐步演示按概率采样的过程。
# NumPy 有现成实现：np.random.choice(len(vec_probs), p=vec_probs)
def sample_by_prob(vec_probs: np.ndarray) -> int:
    # 生成 0~1 之间的随机数
    rand = random.random()

    # 累计概率，判断落在哪个区间
    cumulative = 0.0
    for idx, prob in enumerate(vec_probs):
        cumulative += prob
        if rand < cumulative:
            return idx

    # 兜底（防止浮点误差）
    return len(vec_probs) - 1


vec_probs = np.array([0.4, 0.3, 0.2, 0.1])
print(sample_by_prob(vec_probs))
