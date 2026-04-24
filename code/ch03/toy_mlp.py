import numpy as np

def new_fc_layer(w, b, af=np.tanh):
    return lambda x: af(w @ x + b)

def new_mlp(layers: list):
    def mlp(x):
        current = x # 保存输入，逐层向前传播
        for layer in layers:
            current = layer(current)  # 一层一层计算
        return current
    return mlp


layer1 = new_fc_layer(np.random.randn(300, 784), np.random.randn(300))
layer2 = new_fc_layer(np.random.randn(100, 300), np.random.randn(100))
layer3 = new_fc_layer(np.random.randn(10, 100), np.random.randn(10))
mlp = new_mlp([layer1, layer2, layer3])
print(mlp(np.random.rand(28, 28).flatten()))
