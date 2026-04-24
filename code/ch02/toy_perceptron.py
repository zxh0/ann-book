import numpy as np

def new_fc_layer(w, b, af):
    return lambda x: af(w @ x + b)

w = np.array([[0.11, 0.12, 0.13],
              [0.21, 0.22, 0.23]])
b = np.array([0.31, 0.32])

layer = new_fc_layer(w, b, np.tanh) 
x = np.array([0.4, 0.5, 0.6])
print(layer(x)) # [0.45580236 0.57301481]



# 如果你想让它一次输入多个样本，只需要把 x 变成 (3, N)：
# 你的代码天然支持批量处理！
x = np.array([[0.4, 0.1],
              [0.5, 0.2],
              [0.6, 0.3]])
print(layer(x))
