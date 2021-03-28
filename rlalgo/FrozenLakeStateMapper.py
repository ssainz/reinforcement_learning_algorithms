import numpy as np
class FrozenLakeStateMapper:
    def generate_state(self, obs):
        st = np.zeros(16)
        st[obs] = 1
        return st
    def adjust_output_action(self, action):
        return action.item()