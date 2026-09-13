import numpy as np

def new_rnn_layer(mat_w_xh, mat_w_hh, mat_w_hy, vec_b_h, vec_b_y, vec_h, af):
    def rnn_layer(vec_x):
        nonlocal vec_h  # 声明使用外部的隐藏状态
        vec_h = af(mat_w_xh @ vec_x + mat_w_hh @ vec_h + vec_b_h)
        vec_y = af(mat_w_hy @ vec_h + vec_b_y)
        return vec_h, vec_y
    return rnn_layer


# 3 -> 4 -> 2
_x, _h, _y = 3, 4, 2

mat_w_xh = np.random.rand(_h, _x) # 4×3
mat_w_hh = np.random.rand(_h, _h) # 4×4
mat_w_hy = np.random.rand(_y, _h) # 2x4
vec_b_h = np.random.rand(_h)      # 4×1
vec_b_y = np.random.rand(_y)      # 2×1
vec_h = np.zeros(_h)              # 4×1
vec_x = np.random.rand(_x)        # 3×1
layer = new_rnn_layer(mat_w_xh, mat_w_hh, mat_w_hy, vec_b_h, vec_b_y, vec_h, np.tanh)
vec_h2, vec_y = layer(vec_x)
print(vec_h2, vec_y)

print('mat_w_xh:', mat_w_xh.shape, mat_w_xh)
print('mat_w_hh:', mat_w_hh.shape, mat_w_hh)
print('mat_w_hy:', mat_w_hy.shape, mat_w_hy)
print('vec_b_h:', vec_b_h.shape, vec_b_h)
print('vec_b_y:', vec_b_y.shape, vec_b_y)
print('vec_h:', vec_h.shape, vec_h)
print('vec_x:', vec_x.shape, vec_x)
print(layer)
print('vec_y:', vec_y.shape, vec_y)

for i in range(10):
    vec_x = np.random.rand(_x)
    vec_h, vec_y = layer(vec_x)
    print(f't{i}: x={vec_x.round(3)}, y={vec_y.round(3)}, h={vec_h.round(3)}')



def new_rnn(rnn_layers: list, fc_layer):
    def rnn(vec_x):
        vec_current = vec_x                  # 保存输入，逐层向前传播
        for rnn_layer in rnn_layers:         # 遍历每一个RNN层
            vec_h, vec_y = rnn_layer(vec_current) # 一层一层计算
            vec_current = vec_y              # 忽略隐藏状态
        return fc_layer(vec_current)         # 全连接层计算
    return rnn
