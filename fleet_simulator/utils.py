import numpy as np

def generate_name(exp_conf):
    return exp_conf['net'].__class__.__name__ + "_GAMMA_" + str(exp_conf['gamma']) + "_LR_" + str(exp_conf['lr'])

def get_state_repr_from_int(state_idx):

    city = np.zeros((6,6))

    row = int(state_idx / 6)
    col = state_idx % 6
    city[row,col] = 1

    return get_state_repr(city)

def get_state_from_int(state_idx):

    city = np.zeros((6,6))

    row = int(state_idx / 6)
    col = state_idx % 6
    city[row,col] = 1

    return city


def get_state_as_int(state):

    rows = state.shape[0]
    cols = state.shape[1]
    r = 0
    c = 0
    for i in range(rows):
        for j in range(cols):
            if state[i,j] == 1:
                return state.shape[0] * i + j

def get_state_repr(state_repr):
    state = state_repr.flatten()
    return state

def get_state_as_pair(state):
    rows = state.shape[0]
    cols = state.shape[1]
    r = 0
    c = 0
    for i in range(rows):
        for j in range(cols):
            if state[i,j] == 1:
                c = j
                r = i
    return "(" + str(r) + "," + str(c) + ")"