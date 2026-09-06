import numpy as np
from math import e

def sigmoid(x: float) -> float:
    return 1 / (1 + e ** (-x))

def tanh(x: float) -> float:
    a, b = e ** x, e ** (-x)
    return (a - b) / (a + b)

def relu(x: float) -> float:
    return x if x >= 0 else 0

# 手写实现，用于逐步演示点积的计算过程。
# NumPy 有现成实现：vec_a @ vec_b（也可以写成 np.dot(vec_a, vec_b)）
def dot_product(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return sum(x * y for x, y in zip(vec_a, vec_b))

def new_neuron(vec_w: np.ndarray, b: float, af):
    return lambda vec_x: af(dot_product(vec_w, vec_x) + b)

neuron = new_neuron(vec_w=np.array([1, 1, -1]), b=0, af=relu)
print(neuron(np.array([1, 2, 3])) > 0) # False
print(neuron(np.array([4, 5, 6])) > 0) # True
