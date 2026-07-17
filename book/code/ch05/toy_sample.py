import random

def sample_by_prob(probs: list[float]) -> int:
    # 生成 0~1 之间的随机数
    rand = random.random()
    
    # 累计概率，判断落在哪个区间
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if rand < cumulative:
            return idx
    
    # 兜底（防止浮点误差）
    return len(probs) - 1


probs = [0.4, 0.3, 0.2, 0.1]
print(sample_by_prob(probs))
