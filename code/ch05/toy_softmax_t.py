from math import e

def softmax(z: list[float], t: float) -> list[float]:
    exp_values = [e ** (v/t) for v in z]
    total = sum(exp_values)
    return [v / total for v in exp_values]

# 示例
data = [3, 2, 1]
print([round(x, 3) for x in softmax(data, 0.1)])
print([round(x, 3) for x in softmax(data, 0.5)])
print([round(x, 3) for x in softmax(data, 1.0)])
print([round(x, 3) for x in softmax(data, 1.5)])
print([round(x, 3) for x in softmax(data, 999)])

# [1.0, 0.0, 0.0]
# [0.867, 0.117, 0.016]
# [0.665, 0.245, 0.09]
# [0.563, 0.289, 0.148]
# [0.334, 0.333, 0.333]

data = [3, 1, 6, 2, 5, 4]
print([round(x/1.5, 3) for x in data])
# [2.0, 0.667, 4.0, 1.333, 3.333, 2.667]

print([round(x, 3) for x in softmax(data, 1.5)])
# [0.067, 0.018, 0.496, 0.034, 0.254, 0.131]


# [0.496, 0.254, 0.131, 0.067, 0.034, 0.018]