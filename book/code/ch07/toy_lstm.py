# vec_x => vec_y
def new_fc_layer(mat_w, vec_b, af):
    return lambda vec_x: af(mat_w @ vec_x + vec_b)

# vec_x, vec_h => vec_h'
def new_gate(mat_w_xh, mat_w_hh, vec_b_h, af):
    return lambda vec_x, vec_h: af(mat_w_xh @ vec_x + mat_w_hh @ vec_h + vec_b_h)


def new_lstm_layer(forget_gate, 
                   input_gate, 
                   output_gate, 
                   candidate_gate,
                   fc_layer):
    def layer(vec_x, vec_h, vec_c):
        vec_f = forget_gate(vec_x, vec_h)
        vec_i = input_gate(vec_x, vec_h)
        vec_o = output_gate(vec_x, vec_h)
        vec_c_hat = candidate_gate(vec_x, vec_h)
        vec_c_new = vec_f * vec_c + vec_i * vec_c_hat
        vec_h_new = vec_o * tanh(vec_c_new)
        vec_y = fc_layer(vec_h_new)
        return vec_y
    return layer


def new_gru_layer(update_gate, reset_gate, tanh_gate):
    def layer(vec_x, vec_h):
        vec_z = update_gate(vec_x, vec_h)
        vec_r = reset_gate(vec_x, vec_h)
        vec_h_hat = tanh_gate(vec_x, vec_r * vec_h)
        vec_h_new = (1 - vec_z) * vec_h + vec_z * vec_h_hat
        vec_y = fc_layer(vec_h_new)
        return vec_y
    return layer
