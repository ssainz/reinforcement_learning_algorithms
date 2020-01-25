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
import random
import heapq

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

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class value_net(nn.Module):
    def __init__(self):
        super(value_net, self).__init__()
        bias_on = True
        self.linear1 = nn.Linear(16, 20, bias=bias_on)
        self.linear2 = nn.Linear(20, 40, bias=bias_on)
        self.linear3 = nn.Linear(40, 1, bias=bias_on)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # --- 0000 ---- 0000 >>>  z-score normalization
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
        # x_std = torch.sum(torch.pow(x_minus_x_avg, 2)) / 40
        # x_norm = (x_minus_x_avg) / (torch.sqrt(x_std) + epsilon)
        x = torch.tanh(x)
        x = self.linear3(x)


        return x.view(-1, 1)


class policy_net(nn.Module):
    def __init__(self):
        super(policy_net, self).__init__()
        bias_on = True
        self.linear1 = nn.Linear(16, 20, bias=bias_on)
        self.linear2 = nn.Linear(20, 40, bias=bias_on)
        self.linear3 = nn.Linear(40, 4, bias=bias_on)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # --- 0000 ---- 0000 >>>  z-score normalization
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
        # x_std = torch.sum(torch.pow(x_minus_x_avg, 2)) / 40
        # x_norm = (x_minus_x_avg) / (torch.sqrt(x_std) + epsilon)
        x = torch.tanh(x)
        x = self.linear3(x)
        return x.view(-1, 4)


from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'log_prob', 'action_prob', 'log_action_prob', 'next_state', 'reward',
                         'entropy_impact', 'done'))


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


class ReplayMemoryNoReplacement(object):

    def __init__(self, capacity):
        self.h = []

    def push(self, *args):
        random_index = random.random()
        heapq.heappush(self.h, (random_index, Transition(*args)))

    def sample(self, batch_size):
        result = []
        for i in range(batch_size):
            result.append(heapq.heappop(self.h)[1])

        return result

    def __len__(self):
        return len(self.h)


class ReplayMemoryNew(object):
    def __init__(self, capacity):
        self.h = []
        self.capacity = capacity

    def push(self, *args):
        tran = Transition(*args)
        self.push_transition(tran)

    def push_transition(self, tran):
        if self.capacity <= len(self.h):
            heapq.heappop(self.h)
        random_index = random.random()
        heapq.heappush(self.h, (random_index, tran))

    def sample(self, batch_size):
        result = []
        for i in range(batch_size):
            el = heapq.heappop(self.h)[1]
            result.append(el)
            heapq.heappush(self.h, (random.random(), el))
        return result

    def __len__(self):
        return len(self.h)


def print_v_table():
    for i in range(16):
        # st = np.array(get_state_repr(i))
        # st = np.expand_dims(st, axis=0)
        st = get_state_repr(i)
        v_net.eval()
        action_probs = v_net(FloatTensor(st))
        # action_probs = F.softmax(action_probs, dim=1)
        outp = " state (" + str(i) + ") "
        n = 0
        for tensr in action_probs:
            for cell in tensr:
                outp = outp + " A[" + str(n) + "]:(" + str(cell.item()) + ")"
                n += 1
        print(outp)


def print_pi_table():
    for i in range(16):
        # st = np.array(get_state_repr(i))
        # st = np.expand_dims(st, axis=0)
        st = get_state_repr(i)
        pi_net.eval()
        action_probs = pi_net(FloatTensor(st))
        action_probs = F.softmax(action_probs, dim=1)
        outp = " state (" + str(i) + ") "
        n = 0
        for tensr in action_probs:
            for cell in tensr:
                outp = outp + " A[" + str(n) + "]:(" + str(cell.item()) + ")"
                n += 1
        print(outp)


# def get_state_repr(state_idx):
#     return state_idx * 13


import gym
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
import random

random.seed(1999)
import math
import torch
from torch.optim.lr_scheduler import StepLR


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    # print classname
    # print q_net
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if not m.bias is None:
            m.bias.data.normal_(0.0, 0.02)


def get_state_repr(state_idx):
    state = np.zeros(16)
    state[state_idx] = 1
    return state

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 300
GAMMA = 0.99
TARGET_UPDATE = 1000
PRINT_OUT_TIMES = 1000
ENTROPY_REDUCTION_STEPS = 100000.0
NUM_EPISODES = 10000000
# NUM_STEPS_VALUE_FUNCTION_LEARNS = NUM_EPISODES
#NUM_STEPS_VALUE_FUNCTION_LEARNS = (ENTROPY_REDUCTION_STEPS * 1)
NUM_STEPS_VALUE_FUNCTION_LEARNS = 1

v_net = value_net()
v_net.apply(weights_init)
v_net.to(device)
target_v_net = value_net()
target_v_net.load_state_dict(v_net.state_dict())
target_v_net.to(device)
pi_net = policy_net()
pi_net.apply(weights_init).to(device)

# prepare for optimizer, merge both networks parameters

# parameters = set()
# for net_ in [v_net, pi_net]:
#     parameters |= set(net_.parameters())

# optimizer = optim.RMSprop(online_net.parameters(), lr=0.001)

# optimizer = optim.Adam(parameters, lr=0.0001)


v_optimizer = optim.Adam(v_net.parameters(), lr=0.0001)
pi_optimizer = optim.Adam(pi_net.parameters(), lr=0.00001)

# scheduler = StepLR(v_optimizer, step_size=10000, gamma=0.5)


MEMORY_SIZE = 2000
# memory = ReplayMemoryNoReplacement(MEMORY_SIZE)
memory = ReplayMemoryNew(MEMORY_SIZE)
# memory = ReplayMemory(MEMORY_SIZE)

value_loss_cum = []


def optimize(k):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    final_mask = torch.tensor(tuple(map(lambda d: d is True,batch.done)), device=device, dtype=torch.bool).unsqueeze(1)
    final_mask_list = [d for d in batch.done if d is True]
    # Compute states that are final.
    #     next_state_final_mask = torch.tensor(tuple(map(lambda d: (d) in [5,7,11,12,15],
    #                                           batch.next_state)), device=device, dtype=torch.uint8).unsqueeze(1)
    #     next_state_finak_list = [d for d in batch.next_state if d in [5,7,11,12,15] ]

    # Unpack the parameters from the memory

    state_batch = FloatTensor(batch.state)
    state_batch = state_batch.view(BATCH_SIZE, 16)
    next_state_batch = FloatTensor(batch.next_state)
    next_state_batch = next_state_batch.view(BATCH_SIZE, 16)
    action_batch = LongTensor(batch.action).view(BATCH_SIZE, 1)
    reward_batch = Tensor(batch.reward).view(BATCH_SIZE, 1)
    entropy_impact_batch = FloatTensor(batch.entropy_impact).view(BATCH_SIZE, 1)
    # log_prob_batch = torch.cat(batch.log_prob).view(BATCH_SIZE, 1)
    # action_probs_batch = torch.cat(batch.action_prob).view(BATCH_SIZE,4)
    # log_action_probs_batch = torch.cat(batch.log_action_prob).view(BATCH_SIZE,4)

    # FIRST , calculate V(next_state)and backpropagate MSE on V

    target_v_net.eval()
    v_next = target_v_net(next_state_batch).detach()
    # v_next[next_state_final_mask] = torch.zeros(len(next_state_finak_list), device=device).view(len(next_state_finak_list))
    v_next[final_mask] = torch.zeros(len(final_mask_list), device=device).view(len(final_mask_list))

    ##HACK FIXING expected value
    #     v_current_fixed = [get_expected_value_fixed(_st) for _st in batch.state]
    #     v_current_fixed = FloatTensor(v_current_fixed).view(BATCH_SIZE,1)
    ##HACK FIXING expected value

    ##HACK FIXING current value
    #     v_next_fixed = [get_expected_value_fixed(_st) for _st in batch.next_state]
    #     v_next_fixed = FloatTensor(v_next_fixed).view(BATCH_SIZE,1)
    # v_next = v_next_fixed
    ##HACK FIXING current value

    expected_value = reward_batch + v_next * GAMMA

    ##HACK FIXING expected value
    # expected_value = expected_value_fixed
    ##HACK FIXING expected value

    # calculate V(current_state)
    #if k <= NUM_STEPS_VALUE_FUNCTION_LEARNS:
    #    v_net.train()
    #else:
    #    v_net.eval()
    v_net.train()

    v_current = v_net(state_batch)

    # backpropagate:
    value_loss = torch.sum((expected_value - v_current) ** 2)

    v_optimizer.zero_grad()
    value_loss.backward()  # keep graph for policy net optimizer
    v_optimizer.step()

    # if k <= NUM_STEPS_VALUE_FUNCTION_LEARNS:
    #     v_optimizer.zero_grad()
    #     # value_loss.backward(retain_graph=True) # keep graph for policy net optimizer
    #     value_loss.backward()  # keep graph for policy net optimizer
    #     v_optimizer.step()
        # scheduler.step()

    value_loss_cum.append(value_loss.item())

    v_current = v_current.detach()

    ##HACK FIXING expected value
    # v_current = v_current_fixed
    ##HACK FIXING expected value

    # SECOND, calculate gradient loss:
    # H(X) = P(X) log ( P(X) )

    # calculate the action probability
    actions_distr = pi_net(state_batch)
    actions_prob_batch = torch.softmax(actions_distr, dim=1)
    log_actions_prob_batch = torch.log_softmax(actions_distr, dim=1)

    action_batch = action_batch
    action_mask = FloatTensor(BATCH_SIZE, 4).zero_()
    action_mask.scatter_(1, action_batch, 1)  # This will have shape (BATCH_SIZE, 4), and its contents will be
    # like : [[0,0,1,0],[1,0,0,0],...]
    # log_prob_batch = log_actions_prob_batch.gather(1,action_batch)
    log_prob_batch = torch.sum(log_actions_prob_batch * action_mask, dim=1).view(BATCH_SIZE,
                                                                                 1)  # sum up across rows (ending tensor is shape (BATCH_SIZE, 1))

    entropy = entropy_impact_batch * torch.sum(actions_prob_batch * log_actions_prob_batch)

    #policy_loss = torch.sum(-log_prob_batch * (expected_value - v_current) + entropy)
    policy_loss = torch.sum(-log_prob_batch * (expected_value - v_current))

    pi_optimizer.zero_grad()
    policy_loss.backward()
    pi_optimizer.step()

    return policy_loss.item(), value_loss.item()


score = []
times_trained = 0
times_reach_goal = 0
steps_done = 0

policy_loss_avg = [1.0]
v_loss_avg = [1.0]

TARGET_UPDATE = 1000

for k in range(NUM_EPISODES):
    done = False
    observation = env.reset()
    # observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    reward = 0
    episode_step = 0
    # print("b")
    I = 1.0
    # entropy_impact = (ENTROPY_REDUCTION_STEPS - k) / ENTROPY_REDUCTION_STEPS
    if k == 0:
        entropy_impact = 1.0
    else:
        entropy_impact = min(1, (1 / (k * 0.005)))

    if k > ENTROPY_REDUCTION_STEPS:
        entropy_impact = 0.0

    # test entropy always 0
    # entropy_impact = 0.0

    # entropy_impact = 0.0
    # if entropy_impact < 0.0:
    #    entropy_impact = 0
    while not done:
        # print("c")
        steps_done += 1

        # Get action from pi
        # np_observation = np.array(get_state_repr(observation))
        # np_observation = np.expand_dims(np_observation, axis=0)
        np_observation = get_state_repr(observation)
        # print(np_observation)
        observation_tensor = FloatTensor(np_observation)

        # action distribution
        pi_net.eval()
        action_distr = pi_net(observation_tensor)
        action_probs = torch.softmax(action_distr, dim=1)
        log_action_probs = 0
        # log_action_probs = F.log_softmax(action_distr, dim=1)
        # Decide on an action based on the distribution
        m = Categorical(action_probs)
        action = m.sample()

        log_prob = m.log_prob(action).unsqueeze(1)

        # break
        # Execute action in environment.
        old_state = observation

        observation, reward, done, info = env.step(action.item())
        new_state = observation

        if k % 5000 == 0:
            # print("old_state != new_state")
            # print(old_state != new_state)
            # print("oldstate " + str(old_state) + " newstate " + str(new_state))
            print("action_dist ")
            print(action_probs)
            print("On state=" + str(old_state) + ", selected action=" + str(action.item()))
            print("new state=" + str(new_state) + ", done=" + str(done) + \
                  ". Reward: " + str(reward))

        # Perform one step of the optimization
        #         policy_loss, value_loss = optimize_model(I, \
        #                                                  old_state, \
        #                                                  log_prob, \
        #                                                  log_actions_probs, \
        #                                                  action_probs, \
        #                                                  reward, \
        #                                                  new_state, \
        #                                                  entropy_impact, \
        #                                                  done)

        #         I = I * GAMMA
        # if (not done) or (done and new_state in [5,7,11,12,15]):
        memory.push(get_state_repr(old_state), action.item(), log_prob, action_probs, log_action_probs,
                    get_state_repr(new_state), reward, entropy_impact, done)

        if len(memory) >= MEMORY_SIZE:
            policy_loss, value_loss = optimize(k)
            if len(policy_loss_avg) < PRINT_OUT_TIMES:
                policy_loss_avg.append(policy_loss)
                v_loss_avg.append(value_loss)
            else:
                policy_loss_avg[episode_step % PRINT_OUT_TIMES] = policy_loss
                v_loss_avg[episode_step % PRINT_OUT_TIMES] = value_loss

        times_trained = times_trained + 1

        episode_step += 1
        # env.render()

    if k % PRINT_OUT_TIMES == 0:
        print_pi_table()
        print_v_table()

    if len(score) < 100:
        score.append(reward)
    else:
        score[k % 100] = reward

    if k % TARGET_UPDATE == 0:
        target_v_net.load_state_dict(v_net.state_dict())

    if k % PRINT_OUT_TIMES == 0:
        print("Episode {} finished after {} . Running score: {}. Policy_loss: {}, Value_loss: {}. Times trained: \
              {}. Times reached goal: {}. \
              Steps done: {}.".format(k, episode_step, np.mean(score), np.mean(policy_loss_avg), np.mean(v_loss_avg), times_trained,times_reach_goal, steps_done))
        # print("policy_loss_avg")
        # print(policy_loss_avg)
        # print("value_loss_avg")
        # print(v_loss_avg)
        #         print("times_reach_goal")
        #         print(times_reach_goal)
        times_trained = 0
        times_reach_goal = 0
        # print("Game finished. " + "-" * 5)
        # print(len(episode_series))
    #         for param in net.parameters():
    #             print(param.data)

    if reward > 0.0:
        times_reach_goal = times_reach_goal + 1

