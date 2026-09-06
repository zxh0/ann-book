import numpy as np

def new_fc_layer(mat_w, vec_b, af=np.tanh):
    return lambda vec_x: af(mat_w @ vec_x + vec_b)

def new_mlp(layers: list):
    def mlp(vec_x):
        vec_current = vec_x # 保存输入，逐层向前传播
        for layer in layers:
            vec_current = layer(vec_current)  # 一层一层计算
        return vec_current
    return mlp


layer1 = new_fc_layer(np.random.randn(300, 784), np.random.randn(300))
layer2 = new_fc_layer(np.random.randn(100, 300), np.random.randn(100))
layer3 = new_fc_layer(np.random.randn(10, 100), np.random.randn(10))
mlp = new_mlp([layer1, layer2, layer3])
print(mlp(np.random.randn(28 * 28)))
