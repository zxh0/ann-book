import numpy as np

def new_fc_layer(w, b, af):
    return lambda x: af(w @ x + b)

w = np.array([[0.11, 0.12, 0.13],
              [0.21, 0.22, 0.23]])
b = np.array([0.31, 0.32])

layer = new_fc_layer(w, b, np.tanh) 
x = np.array([0.4, 0.5, 0.6])
print(layer(x)) # [0.45580236 0.57301481]

x = np.array([0.1, 0.2, 0.3])
print(layer(x)) # [0.36617619 0.42518145]
