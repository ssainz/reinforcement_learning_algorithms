import numpy as np
import torch
from torch.distributions import Categorical

from NetConf import NetConf
from Robot import Robot
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)
class RobotReinforce(Robot):
    def __init__(self, config):
        super().__init__(config)
        self.GAMMA = config["gamma"]
        self.net = NetConf(config["net_config"])
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        #print(self.net)
        self.net.inits()
        self.optimizer = optim.Adam(self.net.parameters(), lr=config["lr"])
        self.trajectory = []
        self.optimizer.zero_grad()  # zero gradient buffers
        self.wins = 0
    def generate_state(self, obs):
        st = np.zeros(16)
        st[obs] = 1
        return st
    def add_observation_reward(self, prev_obs, obs, reward):
        prev_state = self.generate_state(prev_obs)
        state = self.generate_state(obs)
        self.trajectory.append((prev_state, self.action, self.log_prob, state, reward))
        if reward > 0:
            self.wins += 1
    def learn_at_end_of_step(self):
        return
    def learn_at_end_of_episode(self):
        index = 0
        GAMMA_outer = self.GAMMA
        losses = []
        reward_greater_zero = False
        for (prev_state, action_tensor, log_prob_tensor, state, reward) in self.trajectory:
            # G:
            i = index
            G = 0
            GAMMA_inner = self.GAMMA
            while i < len(self.trajectory):
                G += GAMMA_inner * self.trajectory[i][4] # reward
                GAMMA_inner *= GAMMA_inner
                i += 1
            loss = -log_prob_tensor * G * GAMMA_outer
            if G > 0.0:
                reward_greater_zero=True
            losses.append(loss)
            GAMMA_outer *= GAMMA_outer
            index += 1
        if reward_greater_zero:
            # gradient descent
            losses_sum = torch.cat(losses).sum()
            losses_sum.backward(retain_graph=True)
            self.optimizer.step()
            # gradient descent
        self.optimizer.zero_grad()  # zero gradient buffers
        self.trajectory.clear()
        return
    def decide(self, obs):
        state = self.generate_state(obs)
        tensor = torch.from_numpy(state)
        if torch.cuda.is_available():
            tensor = tensor.type(torch.FloatTensor).cuda()
        action_prob = self.net(tensor)
        m = Categorical(action_prob)
        self.action = m.sample()
        self.log_prob = m.log_prob(self.action)
        return self.action.item()