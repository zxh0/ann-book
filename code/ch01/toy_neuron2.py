def new_neuron(w: float, b: float):
	return lambda x: w * x + b

neuron = new_neuron(3, 4) # w=3, b=4
print(neuron(2.5)) # 打印出11.5
