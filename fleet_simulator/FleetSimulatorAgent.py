import torch
import torchvision
import gym
import random
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from FleetSimulator import FleetEnv
import numpy as np
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False},
#     max_episode_steps=100,
#     reward_threshold=0.78, # optimum = .8196
# )

#env = gym.make('FrozenLake8x8-v0')
env = FleetEnv()
#env = gym.make('FrozenLakeNotSlippery-v0')

class pi_net(nn.Module):
    def __init__(self):
        super(pi_net, self).__init__()
        bias_on = True
        self.linear1 = nn.Linear(36, 64, bias=bias_on)
        self.linear2 = nn.Linear(64, 64, bias=bias_on)
        self.linear3 = nn.Linear(64, 36, bias=bias_on)
        #torch.nn.init.xavier_uniform_(self.linear1)
        #torch.nn.init.xavier_uniform_(self.linear2)

    def forward(self, x):


        # --- 0000 ---- 0000 >>>  z-score normalization
        x = self.linear1(x)
        # print("AFTER linear1 = = = = = = = = = =")
        # print(x)
        # print("AFTER linear1 = = = = = = = = = =")

        x_avg = torch.sum(x)  / 20
        # print("AVG " + str(x_avg) )
        # print("x - x_avg ~~~~~~~~~~~~~~")
        x_minus_x_avg = x - x_avg
        # print(x_minus_x_avg)
        # print("x - x_avg ~~~~~~~~~~~~~~")
        x_std = torch.sum(torch.pow(x_minus_x_avg, 2)) / 20
        # print("VAR " + str(x_std))
        epsilon = 0.0000001
        # print("STD " + str(torch.sqrt(x_std)))
        x_norm = (x_minus_x_avg) / (torch.sqrt(x_std) + epsilon)
        # print("BEFORE sigmoid = = = = = = = = = =")
        # print(x_norm)
        # print("BEFORE sigmoid = = = = = = = = = =")
        #x = F.sigmoid(x_norm)
        x = F.tanh(x_norm)

        x = self.linear2(x)

        x_avg = torch.sum(x) / 40
        x_minus_x_avg = x - x_avg
        x_std = torch.sum(torch.pow(x_minus_x_avg, 2)) / 40
        x_norm = (x_minus_x_avg) / (torch.sqrt(x_std) + epsilon)
        x = F.tanh(x_norm)

        # print("AFTER sigmoid = = = = = = = = = =")
        # print(x)
        # print("AFTER sigmoid = = = = = = = = = =")
        x = self.linear3(x)
        return x.view(-1, 36)


        # --- 0000 ---- 0000 >>>  feature scaling
        # x = self.linear1(x)
        # print("AFTER linear1 = = = = = = = = = =")
        # print(x)
        # print("AFTER linear1 = = = = = = = = = =")

        # x_max = torch.max(x)
        # x_min = torch.min(x)
        # epsilon = 0.00001
        # x_norm = ((x - x_min) / (x_max - x_min + epsilon))

        # print("BEFORE sigmoid = = = = = = = = = =")
        # print(x_norm)
        # print("BEFORE sigmoid = = = = = = = = = =")
        # x = F.sigmoid(x_norm)
        # print("AFTER sigmoid = = = = = = = = = =")
        # print(x)
        # print("AFTER sigmoid = = = = = = = = = =")
        # x = self.linear2(x)
        # return x.view(-1, 4)


# custom weights initialization
def weights_init_1st(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.15)
        #m.weight.data.uniform_(-0.15, 0.15)
        #m.weight.data.fill_(0.5)

def weights_init_2nd(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(-0.3, 0.3)
        #m.weight.data.uniform_(0.01, 0.02)
        #m.weight.data.fill_(0.5)


def print_net(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.numpy())


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

import torch.optim as optim
from torch.distributions import Categorical

NUM_EPISODES = 50000
GAMMA = 0.99
net = pi_net()
net.apply(weights_init_1st)
optimizer = optim.RMSprop(net.parameters(), lr=0.000001)



FloatTensor =  torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

def print_table():
    for i in range(36):
        st = np.array(get_state_repr_from_int(i))
        st = np.expand_dims(st, axis=0)
        net.eval()
        action_probs = net(FloatTensor(st))
        action_probs = F.softmax(action_probs, dim=1)
        outp = " state (" + str(i) + ") "
        n = 0
        for tensr in action_probs:
            for cell in tensr:
                outp = outp + " A[" + str(n) + "]:(" + str(cell.item()) + ")"
                n += 1
        print(outp)

    print("--------------")


score = []
times_trained = 0
times_reach_goal = 0

reward_chart = []
for k in range(NUM_EPISODES):
    done = False
    observation = env.reset()
    # observation, reward, done, info = env.step(env.action_space.sample()) # take a random action

    episode_series = []
    reward_acum = []
    time_of_day = 0
    while not done:
        # Get action from pi
        # action = env.action_space.sample()
        #np_observation = np.array(get_state_repr(observation))
        np_observation = get_state_repr(observation)
        # np_observation = np.expand_dims(np_observation, axis=0)
        np_observation = np.expand_dims(np_observation, axis=0)
        observation_tensor = torch.FloatTensor(np_observation)

        action_probs = net(observation_tensor)
        action_probs_orig = action_probs

        # FOR EXPLORATION:
        action_probs = F.dropout(action_probs, p=0.3, training=True)

        action_probs = F.softmax(action_probs, dim=1)

        m = Categorical(action_probs)
        action = m.sample()

        log_prob = m.log_prob(action)


        # break
        # Execute action in environment.

        if k % 1000 == 0:
            #print("action_probs_orig ")
            #print(action_probs_orig)
            print("Time of day=" + str(time_of_day) + ", on state=" + str(get_state_as_pair(observation)) +
                  ", selected action=" + str(get_state_as_pair(get_state_from_int(action.item()))) + " ,")

        time_of_day += 1

        observation, reward, done, info = env.step(action.item())

        if k % 1000 == 0:
            print("new state=" + str(get_state_as_pair(observation)) + ", rewards=" + str(reward) + ", done=" + str(done))

        # if done and reward != 1.0:
        # 	if observation == 5 or observation == 7 or observation == 11 or observation == 12:
        # 		reward = -1.0

        step_data = [get_state_repr(observation), action, log_prob, reward, done, info]
        episode_series.append(step_data)
        last_reward = reward
        reward_acum.append(reward)


    # END WHILE SIMULATION

    reward_chart.append(np.sum(reward_acum))

    if len(score) < 100:
        score.append(np.sum(reward_acum))
    else:
        score[k % 100] = np.sum(reward_acum)

    if k % 1000 == 0:
        print(
        "Episode {} finished after {} timesteps with r={}. Running score: {}. Times trained: {}. Times reached goal: {}.".format(
            k, len(episode_series), np.sum(reward_acum), np.mean(score), times_trained, times_reach_goal))
        times_trained = 0
        times_reach_goal = 0
        #print_table()
    # print("Game finished. " + "-" * 5)
    # print(len(episode_series))
    #         for param in net.parameters():
    #             print(param.data)

    # break
    # Training:
    # episode_series.reverse()
    policy_loss = []
    rewards_list = []
    for i in range(len(episode_series)):
        j = i
        G = 0
        #alpha = 1 / len(episode_series)

        # get the log_prob of the last state:
        gamma_cum = 1



        while j < len(episode_series):
            [observation, action, log_prob, reward, done, info] = episode_series[j]
            G = G + reward * gamma_cum

            gamma_cum = gamma_cum * GAMMA
            j = j + 1

        [observation, action, log_prob, reward, done, info] = episode_series[i]

        policy_loss.append(G * -log_prob)

        rewards_list.append(G)


    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    policy_loss = []

    # if reward > 0.0:
    #     print_table()
    #     print_net(net)


    # policy_loss = torch.cat(policy_loss).sum()
    # policy_loss.backward()
    # optimizer.step()

    times_trained = times_trained + 1

    #if G != 0.0:  # Optimize only if rewards are non zero.
        # print "Reward list"
        # print rewards_list
    #	optimizer.zero_grad()
    #	policy_loss = torch.cat(policy_loss).sum()
    #	policy_loss.backward()
    #	optimizer.step()
    #	times_trained = times_trained + 1
    # if reward > 0.0:
    #     print("========= Reward " + str(reward) + " ============")
        # print_net(net)
        # print_table()
        # if times_trained > 0:
        #     exit()

    if reward > 0.0:
        times_reach_goal = times_reach_goal + 1


import matplotlib.pyplot as plt
import matplotlib

chart = plt.plot(reward_chart)
#chart.title("plot")

plt.show()