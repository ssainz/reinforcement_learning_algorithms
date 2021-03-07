import random
import math
import numpy as np
import torch
from torch.distributions import Categorical

from Memory import ReplayMemory, Transition
from NetConf import NetConf
from Robot import Robot
import torch.optim as optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 500000
BATCH_SIZE = 500
TARGET_UPDATE = 50
EPS_START = 0.95
EPS_END = 0.00
EPS_DECAY = 1000000

class RobotDQN(Robot):
    def __init__(self, config):
        super().__init__(config)
        self.GAMMA = config["gamma"]
        self.online_net = NetConf(config["net_config"])
        self.target_net = NetConf(config["net_config"])
        self.memory = ReplayMemory(10000)
        if torch.cuda.is_available():
            self.online_net = self.online_net.cuda()
            self.target_net = self.target_net.cuda()
        self.online_net.inits()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config["lr"])
        self.optimizer.zero_grad()  # zero gradient buffers
        self.cum_reward = 0
        self.steps_done = 0
        self.iterations = 0
    def generate_state(self, obs):
        st = np.zeros(16)
        st[obs] = 1
        return st
    def add_observation_reward(self, prev_obs, action, obs, reward, done):
        prev_state = self.generate_state(prev_obs)
        state = self.generate_state(obs)
        self.memory.push(prev_state, self.action, state, reward, done)
        if reward > 0:
            self.cum_reward += reward
    def learn_at_end_of_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda d: d is False,
                                                batch.done)), device=device, dtype=torch.bool).unsqueeze(1)
        non_final_next_states = torch.cat([torch.FloatTensor([s]) for s, d in zip(batch.next_state, batch.done)
                                           if d is False])

        state_batch = torch.FloatTensor(batch.state)
        state_batch = state_batch.view(BATCH_SIZE, 16)
        action_batch = torch.LongTensor(batch.action).view(BATCH_SIZE, 1)
        reward_batch = torch.FloatTensor(batch.reward).view(BATCH_SIZE, 1)
        if torch.cuda.is_available():
            state_batch = state_batch.to("cuda")
            action_batch = action_batch.to("cuda")
            reward_batch = reward_batch.to("cuda")
            non_final_next_states = non_final_next_states.to("cuda")

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.online_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device).view(BATCH_SIZE, 1)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss (this is like MSE , but less sensitive to outliers )
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return
    def learn_at_end_of_episode(self):
        self.iterations += 1
        if self.iterations % TARGET_UPDATE == 0:
            #print("Overwrite target net")
            self.target_net.load_state_dict(self.online_net.state_dict())
    def decide(self, observation):
        state = self.generate_state(observation)
        observation_tensor = torch.from_numpy(state)
        if torch.cuda.is_available():
            observation_tensor = observation_tensor.type(torch.FloatTensor).cuda()
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        q_values = self.online_net(observation_tensor)
        if sample >= eps_threshold:
            self.action = q_values.max(1)[1]  # First 1 is the dimension, second 1 is the index (this is argmax)
        elif torch.cuda.is_available():
            self.action = torch.FloatTensor([[random.randrange(4)]]).to('cuda')
        else:
            self.action = torch.FloatTensor([[random.randrange(4)]])
        return self.action.item()