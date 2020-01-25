from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))

def print_q_table():
    for i in range(16):

        q_vals = online_net(FloatTensor(get_state_repr(i)))
        outp = " state (" + str(i) + ") "
        n = 0
        for tensr in q_vals:
            for cell in tensr:
                outp = outp + " A[" + str(n) + "]:(" + str(cell.item()) + ")"
                n += 1
        print(outp)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


import torch
import torchvision
import gym

import random
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import gym
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

# env = gym.make('FrozenLake8x8-v0')
# env = gym.make('FrozenLake-v0')
env = gym.make('FrozenLakeNotSlippery-v0')
env.render()

use_cuda = torch.cuda.is_available
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class q_net(nn.Module):
    def __init__(self):
        super(q_net, self).__init__()
        self.linear1 = nn.Linear(16, 20)
        self.linear2 = nn.Linear(20, 40)
        self.linear3 = nn.Linear(40, 4)

    def forward(self, x):

        x = self.linear1(x)

        x_avg = torch.sum(x) / 20
        x_minus_x_avg = x - x_avg
        x_std = torch.sum(torch.pow(x_minus_x_avg, 2)) / 20
        epsilon = 0.0000001
        x_norm = (x_minus_x_avg) / (torch.sqrt(x_std) + epsilon)
        x = torch.tanh(x_norm)

        x = self.linear2(x)

        # x_avg = torch.sum(x) / 40
        # x_minus_x_avg = x - x_avg
        # x_std = torch.sum(torch.pow(x_minus_x_avg, 2)) / 20
        # epsilon = 0.0000001
        # x_norm = (x_minus_x_avg) / (torch.sqrt(x_std) + epsilon)
        x = torch.tanh(x)

        x = self.linear3(x)
        #x = torch.tanh(x)
        x = x.view(-1, 4)
        return x


import gym
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
import random
import math


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        # m.weight.data.uniform_(0.0, 0.02)
        m.weight.data.normal_(0.0, 0.02)
        if not m.bias is None:
            m.bias.data.normal_(0.0, 0.02)

def get_state_repr(state_idx):
    state = np.zeros(16)
    state[state_idx] = 1
    return state

def get_index_repr(state):
    return np.argwhere(state==1).item()


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss()
NUM_EPISODES = 500000
BATCH_SIZE = 500
GAMMA = 0.99
TARGET_UPDATE = 10
EPS_START = 0.95
EPS_END = 0.00
EPS_DECAY = 100000
online_net = q_net().to(device)
online_net.apply(weights_init)
target_net = q_net().to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

memory = ReplayMemory(1000)
#optimizer = optim.RMSprop(online_net.parameters(), lr=0.001)
optimizer = optim.Adam(online_net.parameters(), lr=0.000001)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    #non_final_mask = torch.tensor(tuple(map(lambda d: d is False,
    #                                        batch.done)), device=device, dtype=torch.bool).unsqueeze(1)
    non_final_mask = torch.tensor(tuple(map(lambda d: not get_index_repr(d) in [5, 7, 11, 12], batch.next_state)),device=device, dtype=torch.bool).unsqueeze(1)

    # non_final_next_states = torch.cat([FloatTensor([s]) for s, d in zip(batch.next_state, batch.done)
    #                                    if d is False])
    non_final_next_states = torch.cat([FloatTensor([s]) for s, d in zip(batch.next_state, batch.done)
                                       if not get_index_repr(s) in [5, 7, 11, 12]])

    # state_batch = torch.cat([torch.FloatTensor([s]) for s in batch.state])
    state_batch = FloatTensor(batch.state)
    state_batch = state_batch.view(BATCH_SIZE, 16)
    # action_batch = torch.cat([torch.LongTensor([[a.item()]]) for a in batch.action])
    action_batch = LongTensor(batch.action).view(BATCH_SIZE, 1)
    # reward_batch = torch.cat([torch.tensor([r]) for r in batch.reward])
    reward_batch = Tensor(batch.reward).view(BATCH_SIZE, 1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    #     print("state_batch "+"-" * 10)
    #     print(state_batch.shape)
    #     print("action_batch "+"-" * 10)
    #     print(action_batch.shape)
    state_action_values = online_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device).view(BATCH_SIZE, 1)
    #     print("non_final_mask")
    #     print(non_final_mask.shape)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    #     print("next_state_values")
    #     print(next_state_values.shape)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    #     print("expected_state_action_values")
    #     print(expected_state_action_values.shape)

    # Compute Huber loss (this is like MSE , but less sensitive to outliers )
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    loss = mse(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in online_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()

score = []
times_trained = 0
times_reach_goal = 0
q_loss_avg = [1.0]

steps_done = 0
for k in range(NUM_EPISODES):
    done = False
    observation = env.reset()
    # observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    episode_series = []
    reward = 0
    steps_done_in_episode = 0

    while not done:
        # Get action from pi
        # action = env.action_space.sample()
        #np_observation = np.array(observation)
        #np_observation = np.expand_dims(np_observation, axis=0)
        observation_tensor = FloatTensor(get_state_repr(observation))
        # print(observation_tensor)
        # net.eval()
        # print("before eval")
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        steps_done_in_episode += 1
        q_values = online_net(observation_tensor)
        if sample >= eps_threshold:
            # print "observation_tensor"
            # print observation_tensor.type()
            action = q_values.max(1)[1]  # First 1 is the dimension, second 1 is the index (this is argmax)
        else:
            action = torch.FloatTensor([[random.randrange(4)]]).to('cuda')

        # break
        # Execute action in environment.
        old_state = observation
        observation, reward, done, info = env.step(action.item())
        new_state = observation

        # Store the transition in memory
        memory.push(get_state_repr(old_state), action, get_state_repr(new_state), reward, done)


        # Perform one step of the optimization (on the target network)
        if k > BATCH_SIZE:
            q_loss_val = optimize_model()
            times_trained = times_trained + 1
            q_loss_avg.append(q_loss_val)

        # if k > 100 and done and new_state in [5, 7, 11, 12]:
        #     # print("old_state != new_state")
        #     # print(old_state != new_state)
        #     # print("oldstate " + str(old_state) + " newstate " + str(new_state))
        #     print("On state=" + str(old_state) + ", selected action=" + str(action.item()))
        #     print("new state=" + str(new_state) + ", done=" + str(done) + \
        #           ". Reward: " + str(reward))
        #     exit()

        # env.render()
    if k % TARGET_UPDATE == 0 and k > BATCH_SIZE:
        target_net.load_state_dict(online_net.state_dict())

    if len(score) < 100:
        score.append(reward)
    else:
        score[k % 100] = reward

    if k > BATCH_SIZE and k % 1000 == 0:
        print_q_table()
        print(
            "Episode {} finished after {} timesteps with r={}. Running score: {}. Times trained: {}. Times reached goal: {}.Epsilon: {}, QLoss: {}".format(
                k, steps_done_in_episode, reward, np.mean(score), times_trained, times_reach_goal, eps_threshold, np.mean(q_loss_avg)))
        times_trained = 0
        times_reach_goal = 0
        # print("Game finished. " + "-" * 5)
        # print(len(episode_series))
    #         for param in net.parameters():
    #             print(param.data)

    if reward > 0.0:
        times_reach_goal = times_reach_goal + 1



