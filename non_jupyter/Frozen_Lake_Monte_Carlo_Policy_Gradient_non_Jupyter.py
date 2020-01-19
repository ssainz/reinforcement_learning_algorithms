import torch
import torchvision
import gym
import random
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import gym
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#env = gym.make('FrozenLake8x8-v0')
#env = gym.make('FrozenLake-v0')
env = gym.make('FrozenLakeNotSlippery-v0')
env.render()
class pi_net(nn.Module):
    def __init__(self):
        super(pi_net, self).__init__()
        bias_on = True
        self.linear1 = nn.Linear(16, 20, bias=bias_on)
        self.linear2 = nn.Linear(20, 40, bias=bias_on)
        self.linear3 = nn.Linear(40, 4, bias=bias_on)
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
        return x.view(-1, 4)


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


def get_state_repr_idx(state_idx):
    sign = 1
    if state_idx % 2 == 0:
        sign = -1
    return (state_idx + 1 ) * 13 * sign

def get_state_repr(state_idx):
    state = np.zeros(16)
    state[state_idx] = 1
    return state


import numpy as np
import torch.optim as optim
from torch.distributions import Categorical

NUM_EPISODES = 5000
GAMMA = 0.95
net = pi_net().to(device)
net.apply(weights_init_1st)
optimizer = optim.RMSprop(net.parameters(), lr=0.0001)



use_cuda = torch.cuda.is_available
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

def print_table():
    for i in range(16):
        st = np.array(get_state_repr(i))
        st = np.expand_dims(st, axis=0)
        net.eval()
        action_probs = net(FloatTensor(st))
        # action_probs = F.softmax(action_probs, dim=1)
        outp = " state (" + str(i) + ") "
        n = 0
        for tensr in action_probs:
            for cell in tensr:
                outp = outp + " A[" + str(n) + "]:(" + str(cell.item()) + ")"
                n += 1
        print(outp)



score = []
res = {}
times_trained = 0
times_reach_goal = 0
for k in range(NUM_EPISODES):
    done = False
    observation = env.reset()
    # observation, reward, done, info = env.step(env.action_space.sample()) # take a random action

    episode_series = []
    while not done:
        # Get action from pi
        # action = env.action_space.sample()
        #np_observation = np.array(get_state_repr(observation))
        np_observation = get_state_repr(observation)
        # np_observation = np.expand_dims(np_observation, axis=0)
        np_observation = np.expand_dims(np_observation, axis=0)
        observation_tensor = FloatTensor(np_observation)

        action_probs = net(observation_tensor)
        action_probs_orig = action_probs

        # FOR EXPLORATION:
        # action_probs = F.dropout(action_probs, p=0.3, training=True)

        action_probs = F.softmax(action_probs, dim=1)

        m = Categorical(action_probs)
        action = m.sample()

        log_prob = m.log_prob(action)


        # break
        # Execute action in environment.

        if k % 1000 == 0:
            print("action_probs_orig ")
            print(action_probs_orig)
            print("On state=" + str(observation) + ", selected action=" + str(action.item()) + " , ")

        observation, reward, done, info = env.step(action.item())

        if k % 1000 == 0:
            print("new state=" + str(observation) + ", done=" + str(done))

        # if done and reward != 1.0:
        # 	if observation == 5 or observation == 7 or observation == 11 or observation == 12:
        # 		reward = -1.0

        step_data = [get_state_repr(observation), action, log_prob, reward, done, info]
        episode_series.append(step_data)
        last_reward = reward

    # END WHILE SIMULATION

    if len(score) < 100:
        score.append(reward)
    else:
        score[k % 100] = reward

    if k % 100 == 0:
        res[k] = np.mean(score)

    if k % 1000 == 0:
        print(
        "Episode {} finished after {} timesteps with r={}. Running score: {}. Times trained: {}. Times reached goal: {}.".format(
            k, len(episode_series), reward, np.mean(score), times_trained, times_reach_goal))
        times_trained = 0
        times_reach_goal = 0
        print_table()
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

print("RESULTS")

kk = sorted(res)

for key in kk:
    print(str(key) + " " + str(res[key]))
