import random
import math
import numpy as np
import torch
from torch.distributions import Categorical

from rlalgo.Memory import ReplayMemory, Transition
from rlalgo.NetConf import NetConf
from .Robot import Robot
import torch.optim as optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 500000
BATCH_SIZE = 50
TARGET_UPDATE = 50
EPS_START = 0.95
EPS_END = 0.00
EPS_DECAY = 5000

class RobotDQN(Robot):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.in_size = self.config["net_config"]["layers"][0]
        self.out_size = self.config["net_config"]["layers"][-1]
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
        self.last_dimension = len(self.config["net_config"]["output_shape"])-1
        self.state_mapper = config["state_mapper"]
    def generate_state(self, obs):
        return self.state_mapper.generate_state(obs)
    def adjust_output_action(self, action):
        return self.state_mapper.adjust_output_action(action)
    def add_observation_reward(self, prev_obs, action, obs, reward, done):
        prev_state = self.generate_state(prev_obs)
        state = self.generate_state(obs)
        self.memory.push(prev_state, self.action, state, reward, done)
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
        if len(self.get_one_decision_action_shape()) >= 3:
            # For SAV we need to reduce the dimensions from [50,1] to [50]
            non_final_mask = non_final_mask.squeeze(len(non_final_mask.size())-1)
            # For SAV , non_final_mask is now [50]

        non_final_next_states = torch.cat([torch.FloatTensor([s]) for s, d in zip(batch.next_state, batch.done)
                                           if d is False])

        state_batch = torch.FloatTensor(batch.state)
        state_batch = state_batch.view(BATCH_SIZE, self.in_size)

        if len(self.get_one_decision_action_shape()) >= 3:
            action_batch = torch.cat(batch.action).long() # For SAV action_batch is [50,21]
            action_batch = action_batch.unsqueeze(len(action_batch.size())) # For SAV action_batch is [50,21,1]
        else:
            action_batch = torch.LongTensor(batch.action) # example FrozenLake has action_batch such as (1,)
            action_batch = action_batch.view(BATCH_SIZE, 1) # For FrozenLake action_batch is 50, 1

        reward_batch = torch.FloatTensor(batch.reward).view(BATCH_SIZE, 1)
        if torch.cuda.is_available():
            state_batch = state_batch.to("cuda")
            action_batch = action_batch.to("cuda")
            reward_batch = reward_batch.to("cuda")
            non_final_next_states = non_final_next_states.to("cuda")

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        net_output = self.online_net(state_batch)
        if len(self.get_one_decision_action_shape()) >= 3:
            # gather last dimension
            state_action_values = net_output.gather(len(self.get_one_decision_action_shape())-1, action_batch)
            # for SAV state_action_values is 50, 21, 1
        else:
            state_action_values = net_output.gather(1, action_batch)
            # for FrozenLake, state_action_values is 50,1

        # Compute V(s_{t+1}) for all next states.
        net_output_non_final_next_states_value = self.target_net(non_final_next_states).max(self.last_dimension)[0].detach()
        if len(self.get_one_decision_action_shape()) >= 3:
            shape_of_output = self.get_one_decision_action_shape(BATCH_SIZE)
            shape_of_output = (shape_of_output[0], shape_of_output[1], 1) # for SAV, from (50,21,22) to (50,21,1)
            next_state_values = torch.zeros(shape_of_output, device=device)
            net_output_non_final_next_states_value = net_output_non_final_next_states_value.unsqueeze(len(net_output_non_final_next_states_value.size()))
            # for SAV net_output_non_final_next_states_value is [50,21,1]
            next_state_values[non_final_mask] = net_output_non_final_next_states_value
            # for SAV next_state_values is now [50,21,1]
        else:
            next_state_values = torch.zeros(BATCH_SIZE, device=device).view(BATCH_SIZE, 1)
            # for FrozenLake net_output_non_final_next_states_value is [50,1]
            next_state_values[non_final_mask] = net_output_non_final_next_states_value

        # Compute the expected Q values
        if len(self.get_one_decision_action_shape()) >= 3:
            # For SAV , we need to duplicate the reward:
            reward_batch = reward_batch # reward_batch is [50,1]
            reward_batch = reward_batch.repeat(1,21) # reward_batch is [50,21]
            reward_batch = reward_batch.unsqueeze(len(reward_batch.size()))
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
        observation_tensor = torch.from_numpy(state).type(torch.FloatTensor)
        if torch.cuda.is_available():
            observation_tensor = observation_tensor.type(torch.FloatTensor).cuda()
        sample = random.random()
        #eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        eps_threshold = max((1.0 - (self.steps_done / EPS_DECAY)), 0.0)
        self.steps_done += 1
        if sample >= eps_threshold:
            q_values = self.online_net(observation_tensor)
            self.action = q_values.max(len(q_values.size()) - 1)[1]  # First 1 is the dimension, second 1 is the index (this is argmax)
        elif torch.cuda.is_available():
            shp = self.get_one_decision_action_shape()
            self.action = torch.rand(shp).float().to('cuda').max(len(shp)-1)[1]
        else:
            shp = self.get_one_decision_action_shape()
            self.action = torch.rand(shp).float().max(len(shp)-1)[1]
        return self.adjust_output_action(self.action)
    def get_cum_reward(self):
        return self.cum_reward
    def set_cum_reward(self, val):
        self.cum_reward = val
    def get_one_decision_action_shape(self, BATCH_SIZE=1):
        if len(self.config["net_config"]["output_shape"]) == 3:
            #example [-1, 21, 21*22]
            return (BATCH_SIZE, self.config["net_config"]["output_shape"][1], self.config["net_config"]["output_shape"][2])
        else:
            # len of shape is 2 we hope!
            #example [-1, 4]
            return (BATCH_SIZE, self.config["net_config"]["output_shape"][1])