import numpy as np

def zero_pad(mat_input, padding):
    if padding > 0:
        return np.pad(
            mat_input,
            pad_width=((padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )
    return mat_input


def apply_sliding_window(mat_input, k_h, k_w, stride, func):
    h, w = mat_input.shape                   # 获取输入矩阵的高、宽
    out_h = (h - k_h) // stride + 1          # 计算输出矩阵的高、宽
    out_w = (w - k_w) // stride + 1          #
    mat_output = np.zeros((out_h, out_w))    # 初始化输出矩阵

    # 开始滑动窗口
    for i in range(out_h):
        for j in range(out_w):
            # 截取当前窗口
            start_i = i * stride
            start_j = j * stride
            mat_window = mat_input[start_i:start_i + k_h, start_j:start_j + k_w]

            # 进行计算
            mat_output[i, j] = func(mat_window)

    return mat_output


def conv2d(mat_input, mat_kernel, bias, padding=0, stride=1):
    conv_func = lambda mat_window: np.sum(mat_window * mat_kernel) + bias # 卷积核计算
    k_h, k_w = mat_kernel.shape # 获取输入卷积核的高、宽
    mat_input = zero_pad(mat_input, padding) # 填充
    mat_output = apply_sliding_window(mat_input, k_h, k_w, stride, conv_func)
    return mat_output


mat_input = np.random.rand(4, 4)
mat_kernel = np.random.rand(2, 2)
print(mat_input)
print(zero_pad(mat_input, 2))
print(conv2d(mat_input, mat_kernel, 2.3))

#####

def new_conv_layer(kernel_list, padding=0, stride=1):
    def conv_layer(mat_input):
        mat_output_list = []
        for mat_kernel, bias in kernel_list:
            mat_output = conv2d(mat_input, mat_kernel, bias, padding, stride)
            mat_output_list.append(mat_output)
        return mat_output_list
    return conv_layer

kernel_list = [
    (np.random.rand(3, 3), np.random.rand()),
    (np.random.rand(3, 3), np.random.rand()),
    (np.random.rand(3, 3), np.random.rand()),
]
# print('kernel_list:', kernel_list)
conv_layer = new_conv_layer(kernel_list)

mat_input = np.random.rand(16, 16)
mat_output_list = conv_layer(mat_input)
print('mat_output_list:', mat_output_list)


#####

def new_conv_layer2(kernel_list, padding=0, stride=1):
    def conv_layer(mat_input_list):
        mat_output_list = []

        # 遍历每一个多通道卷积核 (mat_kernel_list, bias)
        for mat_kernel_list, bias in kernel_list:
            mat_output_list_tmp = []

            # 遍历每个输入通道（对应每个卷积核通道）
            for i in range(len(mat_input_list)):
                mat_input = mat_input_list[i]
                mat_kernel = mat_kernel_list[i]
                mat_output = conv2d(mat_input, mat_kernel, 0, padding, stride)
                mat_output_list_tmp.append(mat_output)

            # 所有通道结果逐元素相加 → 再加偏置
            mat_fused = np.sum(mat_output_list_tmp, axis=0)
            mat_output = mat_fused + bias
            mat_output_list.append(mat_output)

        return mat_output_list
    return conv_layer


kernel_list2 = [
    ([np.random.rand(3, 3), np.random.rand(3, 3)], np.random.rand()),
    ([np.random.rand(3, 3), np.random.rand(3, 3)], np.random.rand()),
    ([np.random.rand(3, 3), np.random.rand(3, 3)], np.random.rand()),
]
# print('kernel_list:', kernel_list)
conv_layer2 = new_conv_layer2(kernel_list2)

mat_input_list2 = [np.random.rand(16, 16), np.random.rand(16, 16)]
mat_output_list2 = conv_layer2(mat_input_list2)
print('mat_output_list:', mat_output_list2)


###

def max_pool2d(mat_input, win_size, stride=1):
    return apply_sliding_window(mat_input, win_size, win_size, stride, np.max)


###

def flatten(mat_input_list):
    vec_flattened = []                # 创建一个空列表，用于存放所有元素
    for mat in mat_input_list:        # 遍历每一个通道的特征图
        for vec_row in mat:           # 遍历矩阵中的每一行元素
            vec_flattened.extend(vec_row) # 将一行元素全部加入列表
    return vec_flattened              # 返回展平后的一维数组
