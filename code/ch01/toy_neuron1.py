def new_neuron(w: float):
	return lambda x: w * x

neuron = new_neuron(2) # w=2
print(neuron(1.2)) # 打印出2.4
