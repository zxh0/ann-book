import numpy as np

def zero_pad(input_mat, padding):
    if padding > 0:
        return np.pad(
            input_mat,
            pad_width=((padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )
    return input_mat


def apply_sliding_window(input_mat, k_h, k_w, stride, func):
    h, w = input_mat.shape                # 获取输入矩阵的高、宽
    out_h = (h - k_h) // stride + 1       # 计算输出矩阵的高、宽
    out_w = (w - k_w) // stride + 1       #
    output_mat = np.zeros((out_h, out_w)) # 初始化输出矩阵

    # 开始滑动窗口
    for i in range(out_h):
        for j in range(out_w):
            # 截取当前窗口
            start_i = i * stride
            start_j = j * stride
            window = input_mat[start_i:start_i + k_h, start_j:start_j + k_w]

            # 进行计算
            output_mat[i, j] = func(window)

    return output_mat


def conv2d(input_mat, kernel_mat, bias, padding=0, stride=1):
    conv_func = lambda window: np.sum(window * kernel_mat) + bias # 卷积核计算
    k_h, k_w = kernel_mat.shape # 获取输入卷积核的高、宽
    input_mat = zero_pad(input_mat, padding) # 填充
    output_mat = apply_sliding_window(input_mat, k_h, k_w, stride, conv_func)
    return output_mat


input_mat = np.random.rand(4, 4)
kernel_mat = np.random.rand(2, 2)
print(input_mat)
print(zero_pad(input_mat, 2))
print(conv2d(input_mat, kernel_mat, 2.3))

#####

def new_conv_layer(kernel_list, padding=0, stride=1):
    def conv_layer(input_mat):
        output_mat_list = []
        for kernel_mat, bias in kernel_list:
            output_mat = conv2d(input_mat, kernel_mat, bias, padding, stride)
            output_mat_list.append(output_mat)
        return output_mat_list
    return conv_layer

kernel_list = [
    (np.random.rand(3, 3), np.random.rand()),
    (np.random.rand(3, 3), np.random.rand()),
    (np.random.rand(3, 3), np.random.rand()),
]
# print('kernel_list:', kernel_list)
conv_layer = new_conv_layer(kernel_list)

input_mat = np.random.rand(16, 16)
output_mat_list = conv_layer(input_mat)
print('output_mat_list:', output_mat_list)


#####

def new_conv_layer2(kernel_list, padding=0, stride=1):
    def conv_layer(input_mat_list):
        output_mat_list = []
        
        # 遍历每一个多通道卷积核 (kernel_mat_list, bias)
        for kernel_mat_list, bias in kernel_list:
            output_mat_list_tmp = []
            
            # 遍历每个输入通道（对应每个卷积核通道）
            for i in range(len(input_mat_list)):
                input_mat = input_mat_list[i]
                kernel_mat = kernel_mat_list[i]
                output_mat = conv2d(input_mat, kernel_mat, 0, padding, stride)
                output_mat_list_tmp.append(output_mat)
            
            # 所有通道结果逐元素相加 → 再加偏置
            fused_mat = np.sum(output_mat_list_tmp, axis=0)
            output_mat = fused_mat + bias
            output_mat_list.append(output_mat)
        
        return output_mat_list
    return conv_layer


kernel_list2 = [
    ([np.random.rand(3, 3), np.random.rand(3, 3)], np.random.rand()),
    ([np.random.rand(3, 3), np.random.rand(3, 3)], np.random.rand()),
    ([np.random.rand(3, 3), np.random.rand(3, 3)], np.random.rand()),
]
# print('kernel_list:', kernel_list)
conv_layer2 = new_conv_layer2(kernel_list2)

input_mat2 = [np.random.rand(16, 16), np.random.rand(16, 16)]
output_mat_list2 = conv_layer2(input_mat2)
print('output_mat_list:', output_mat_list2)


###

def max_pool2d(input_mat, win_size, stride=1):
    return apply_sliding_window(input_mat, win_size, win_size, stride, np.max)


### 

def flatten(input_mat_list):
    flattened = []                # 创建一个空列表，用于存放所有元素
    for mat in input_mat_list:    # 遍历每一个通道的特征图
        for row in mat:           # 遍历矩阵中的每一行元素
            flattened.extend(row) # 将一行元素全部加入列表
    return flattened              # 返回展平后的一维数组