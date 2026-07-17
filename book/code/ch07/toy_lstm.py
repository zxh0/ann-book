# x => y
def new_fc_layer(w, b, af):
    return lambda x: af(w @ x + b)

# x, h => h'
def new_gate(w_xh, w_hh, b_h, af):
    return lambda x, h: af(w_xh @ x + w_hh @ h + b_h)


def new_lstm_layer(forget_gate, 
                   input_gate, 
                   output_gate, 
                   candidate_gate,
                   fc_layer):
    def layer(x, h, c):
        _f = forget_gate(x, h)
        _i = input_gate(x, h)
        _o = output_gate(x, h)
        _c = candidate_gate(x, h)
        new_c = _f * c + _i * _c
        new_t = _o * tanh(new_c)
        y = fc_layer(new_t)
        return y
    return layer


def new_gru_layer(update_gate, reset_gate, tanh_gate):
    def layer(x, h):
        _z = update_gate(x, h)
        _r = reset_gate(x, h)
        _h = tanh_gate(x, _r * h)
        new_h = (1 - _z) * h + _z * _h
        y = fc_layer(new_t)
        return y
    return layer
