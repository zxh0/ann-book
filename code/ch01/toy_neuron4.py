from math import e

def sigmoid(x: float) -> float:
    return 1 / (1 + e ** (-x))

def tanh(x: float) -> float:
    a, b = e ** x, e ** (-x)
    return (a - b) / (a + b)

def relu(x: float) -> float:
    return x if x >= 0 else 0

def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def new_neuron(w: list[float], b: float, af):
    return lambda x: af(dot_product(w, x) + b)

neuron = new_neuron(w=[1, 1, -1], b=0, af=relu)
print(neuron([1, 2, 3]) > 0) # False
print(neuron([4, 5, 6]) > 0) # True
