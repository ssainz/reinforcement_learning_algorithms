import numpy as np
import torch
from torch.distributions import Categorical

from rlalgo.NetConf import NetConf
from .Robot import Robot
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)
class RobotRebalance(Robot):
    def __init__(self, config):
        super().__init__(config)
        self.cum_reward = 0
        self.state_mapper = config["state_mapper"]
        self.config = config
        self._output_shape = self.config["net_config"]["output_shape"]
        self.output_shape = (1, self._output_shape[1], self._output_shape[2])
    def generate_state(self, obs):
        return self.state_mapper.generate_state(obs)
    def adjust_output_action(self, action):
        return self.state_mapper.adjust_output_action(action)
    def add_observation_reward(self, prev_obs, action, obs, reward, done):
        self.cum_reward += reward
    def learn_at_end_of_step(self):
        return
    def learn_at_end_of_episode(self):
        return
    def decide(self, obs):
        state = self.generate_state(obs)
        state_np = np.asarray(state)
        sav_density = state_np[0:21]
        rider_density = state_np[21:42]
        imbalance = (sav_density-rider_density)/rider_density
        index_greater_10_pct = np.nonzero(imbalance >= 0.1)
        index_negative_imbalance = np.nonzero(imbalance < 0)
        target = imbalance.copy()
        target[target >= 0] = 0 # no positive left in the target array.
        target = target * -1 # now the most negatives are the most positive.
        matt = np.identity(21) # 21 x 21 identity matrix
        matt[index_greater_10_pct] = target # only those areas with high SAV density will send to areas with low density
        matt = matt.copy()
        park_action_col = np.zeros((self.output_shape[1],1)) # additional column
        mat2 = np.hstack((park_action_col, matt)) # mat2 is (21,22) , park is first col.
        action_prob = torch.tensor(mat2)
        action_prob = action_prob.unsqueeze(0) # action_prob is (1,21,22)
        m = Categorical(action_prob)
        self.action = m.sample() # self.action is (1,21)
        return self.adjust_output_action(self.action)
    def get_cum_reward(self):
        return self.cum_reward
    def set_cum_reward(self, val):
        self.cum_reward = val