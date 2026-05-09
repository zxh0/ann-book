import numpy as np

def calc_qkv(w_q, w_k, w_v, x):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    return q, k, v


def calc_score(q, k, v, sqrt_d_k):
    score = q @ k.T / sqrt_d_k
    return softmax(score)


def calc_self_attention(w_q, w_k, w_v, sqrt_d_k, x):
    q, k, v = calc_qkv(w_q, w_k, w_v, x)
    score = calc_score(q, k, v, sqrt_d_k)
    return score @ v


def new_head(w_q, w_k, w_v, sqrt_d_k):
    return lambda x: calc_self_attention(w_q, w_k, w_v, sqrt_d_k, x)


def new_multi_head(heads, w_o):
    def multi_head(x):
        z_list = [head(x) for head in heads]
        z = np.concatenate(z_list, axis=1)
        return z @ w_o
    return multi_head
