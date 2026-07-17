import numpy as np

def new_rnn_layer(w_xh, w_hh, w_hy, b_h, b_y, h, af):
    def rnn_layer(x):
        nonlocal h  # 声明使用外部的h
        h = af(w_xh @ x + w_hh @ h + b_h)
        y = af(w_hy @ h + b_y)
        return h, y
    return rnn_layer


# 3 -> 4 -> 2
_x, _h, _y = 3, 4, 2

w_xh = np.random.rand(_h, _x) # 4×3
w_hh = np.random.rand(_h, _h) # 4×4
w_hy = np.random.rand(_y, _h) # 2x4
b_h = np.random.rand(_h)      # 4×1
b_y = np.random.rand(_y)      # 2×1
h = np.zeros(_h)              # 4×1
x = np.random.rand(_x)        # 3×1
layer = new_rnn_layer(w_xh, w_hh, w_hy, b_h, b_y, h, np.tanh)
h2, y = layer(x)
print(h2, y)

print('w_xh:', w_xh.shape, w_xh)
print('w_hh:', w_hh.shape, w_hh)
print('w_hy:', w_hy.shape, w_hy)
print('b_h:', b_h.shape, b_h)
print('b_y:', b_y.shape, b_y)
print('h:', h.shape, h)
print('x:', x.shape, x)
print(layer)
print('y:', y.shape, y)

for i in range(10):
    x = np.random.rand(_x)
    h, y = layer(x)
    # print('#', i, ', x:', x, ', y:', y, ', h:', h)
    print(f't{i}: x={x.round(3)}, y={y.round(3)}, h={h.round(3)}')



def new_rnn(rnn_layers: list, fc_layer):
    def rnn(x):
        current = x                     # 保存输入，逐层向前传播
        for rnn_layer in rnn_layers:    # 遍历每一个RNN层
            _h, _y = rnn_layer(current) # 一层一层计算
            current = _y                # 忽略隐藏状态
        return fc_layer(current)        # 全连接层计算
    return rnn