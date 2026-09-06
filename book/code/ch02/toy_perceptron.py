import numpy as np

def new_fc_layer(mat_w, vec_b, af):
    return lambda vec_x: af(mat_w @ vec_x + vec_b)

mat_w = np.array([[0.11, 0.12, 0.13],
                  [0.21, 0.22, 0.23]])
vec_b = np.array([0.31, 0.32])

layer = new_fc_layer(mat_w, vec_b, np.tanh)
vec_x = np.array([0.4, 0.5, 0.6])
print(layer(vec_x)) # [0.45580236 0.57301481]

vec_x = np.array([0.1, 0.2, 0.3])
print(layer(vec_x)) # [0.36617619 0.42518145]
