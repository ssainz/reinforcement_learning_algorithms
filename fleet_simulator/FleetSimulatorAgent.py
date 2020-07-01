import torch
import torchvision
import gym
import random
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from FleetSimulator import FleetEnv
from utils import get_state_repr, get_state_as_pair, get_state_from_int, generate_name
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical


# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False},
#     max_episode_steps=100,
#     reward_threshold=0.78, # optimum = .8196
# )

#env = gym.make('FrozenLake8x8-v0')

#env = gym.make('FrozenLakeNotSlippery-v0')
FloatTensor =  torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

def start_experiment(exp_conf):
    NUM_EPISODES = exp_conf['iterations']
    DEBUG = exp_conf["DEBUG"]
    GAMMA = exp_conf['gamma']
    net = exp_conf['net']
    optimizer = optim.RMSprop(net.parameters(), lr=exp_conf['lr'])

    env = FleetEnv()

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

            if k % 1000 == 0 and DEBUG:
                #print("action_probs_orig ")
                #print(action_probs_orig)
                print("Time of day=" + str(time_of_day) + ", on state=" + str(get_state_as_pair(observation)) +
                      ", selected action=" + str(get_state_as_pair(get_state_from_int(action.item()))) + " ,")

            time_of_day += 1

            observation, reward, done, info = env.step(action.item())

            if k % 1000 == 0 and DEBUG:
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

        if k % 1 == 0 and DEBUG:
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


    return reward_chart, generate_name(exp_conf)