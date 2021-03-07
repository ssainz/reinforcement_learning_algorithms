import gym
import torch
import numpy as np
import torch.optim as optim
from torch.distributions.categorical import Categorical
from rlalgo.NetConf import Net_REINFORCE


# register(
#    id='FrozenLakeNotSlippery-v0',
#    entry_point='gym.envs.toy_text:FrozenLakeEnv',
#    kwargs={'map_name' : '4x4', 'is_slippery': False},
#    max_episode_steps=100,
#    reward_threshold=0.78, # optimum = .8196
# )


class agent:
    def __init__(self):
        self.GAMMA = 0.99
        self.net = Net_REINFORCE()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        #print(self.net)
        self.net.inits()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
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


torch.autograd.set_detect_anomaly(True)
env = gym.make('FrozenLake-v0')
robot = agent()
hits = 0
MAX_ITERATIONS = 2000000
FREQUENCY_CHECKS = 100
for iteration in range(MAX_ITERATIONS):
    observation = env.reset()
    done = False
    while not done:
        #env.render()
        (new_observation, reward, done, info) = env.step(robot.decide(observation)) # take a random action
        robot.add_observation_reward(observation, new_observation, reward)
        robot.learn_at_end_of_step()
        #print(new_observation)
        observation = new_observation
    robot.learn_at_end_of_episode()
    if (iteration+1) % FREQUENCY_CHECKS == 0:
        print("Wins per {}: {}".format(FREQUENCY_CHECKS, robot.wins))
        robot.wins = 0
env.close()

