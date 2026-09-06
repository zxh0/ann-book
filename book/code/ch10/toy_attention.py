import numpy as np

def calc_qkv(mat_w_q, mat_w_k, mat_w_v, mat_x):
    mat_q = mat_x @ mat_w_q
    mat_k = mat_x @ mat_w_k
    mat_v = mat_x @ mat_w_v
    return mat_q, mat_k, mat_v


def calc_score(mat_q, mat_k, mat_v, sqrt_d_k):
    mat_score = mat_q @ mat_k.T / sqrt_d_k
    return softmax(mat_score)


def calc_self_attention(mat_w_q, mat_w_k, mat_w_v, sqrt_d_k, mat_x):
    mat_q, mat_k, mat_v = calc_qkv(mat_w_q, mat_w_k, mat_w_v, mat_x)
    mat_score = calc_score(mat_q, mat_k, mat_v, sqrt_d_k)
    return mat_score @ mat_v


def new_head(mat_w_q, mat_w_k, mat_w_v, sqrt_d_k):
    return lambda mat_x: calc_self_attention(mat_w_q, mat_w_k, mat_w_v, sqrt_d_k, mat_x)


def new_multi_head(heads, mat_w_o):
    def multi_head(mat_x):
        mat_z_list = [head(mat_x) for head in heads]
        mat_z = np.concatenate(mat_z_list, axis=1)
        return mat_z @ mat_w_o
    return multi_head
