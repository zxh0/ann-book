from math import e

def softmax(z: list[float]) -> list[float]:
    exp_values = [e ** v for v in z]
    total = sum(exp_values)
    return [v / total for v in exp_values]

# 示例
data = [2.0, 1.0, 0.1]
print(softmax(data))
