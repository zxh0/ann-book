from math import e

def sigmoid(x: float) -> float:
	return 1 / (1 + e ** (-x))

def tanh(x: float) -> float:
	a = e ** x
	b = e ** (-x)
	return (a - b) / (a + b)

def relu(x: float) -> float:
	return x if x >= 0 else 0

def new_neuron(w: float, b: float, af):
	return lambda x: af(w * x + b)

neuron1 = new_neuron(w=1, b=2, af=sigmoid)
neuron2 = new_neuron(w=3, b=4, af=tanh)
neuron3 = new_neuron(w=5, b=6, af=relu)
print(neuron1(-1.2)) # 打印出0.9608342772032357
print(neuron2(-3.4)) # 打印出0.9999999999990731
print(neuron3(-5.6)) # 打印出34.0
