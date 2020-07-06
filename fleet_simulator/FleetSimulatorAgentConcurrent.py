from Models import pi_net, weights_init_1st
import multiprocessing
import torch.optim as optim
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import tanh
import numpy as np
from utils import get_state_repr_from_int, get_state_from_int, get_state_as_int, get_state_repr, get_state_as_pair



FloatTensor =  torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

def __repr__(self):
    return '<%s.%s object at %s>' % (
        self.__class__.__module__,
        self.__class__.__name__,
        hex(id(self))
    )

class Agent():

    def __init__(self, agent_id, sending_queue, response_queue, episodes, exp_conf, results):
        self.agent_id = agent_id
        self.action_queue = sending_queue
        self.continue_queue = response_queue

        self.net = pi_net()
        self.net.apply(weights_init_1st)
        self.lr = exp_conf['lr']
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr)

        self.episodes = episodes
        self.results = results
        self.net_class = exp_conf['net']
        self.DEBUG = exp_conf["DEBUG"]
        self.GAMMA = exp_conf['gamma']



    def reset(self):
        print("reset")

    def start(self):
        #print(multiprocessing.current_process(), __repr__(self.net), __repr__(self))
        reward_per_day = []
        score = []
        times_trained = 0
        times_reach_goal = 0
        for k in range(self.episodes):
            observation = np.zeros((6,6))
            observation[0,0] = 6

            episode_series = []
            reward_acum = []

            time_of_day = 0
            done = False
            while not done:
                np_observation = get_state_repr(observation)
                # np_observation = np.expand_dims(np_observation, axis=0)
                np_observation = np.expand_dims(np_observation, axis=0)
                observation_tensor = torch.FloatTensor(np_observation)

                action_probs = self.net(observation_tensor)
                action_probs_orig = action_probs

                # FOR EXPLORATION:
                action_probs = F.dropout(action_probs, p=0.3, training=True)

                action_probs = F.softmax(action_probs, dim=1)

                m = Categorical(action_probs)
                action = m.sample()

                log_prob = m.log_prob(action)

                # break
                # Execute action in environment.

                if k % 1000 == 0 and self.DEBUG:
                    # print("action_probs_orig ")
                    # print(action_probs_orig)
                    print("Time of day=" + str(time_of_day) + ", on state=" + str(get_state_as_pair(observation)) +
                          ", selected action=" + str(get_state_as_pair(get_state_from_int(action.item()))) + " ,")

                time_of_day += 1

                # sending to env:
                self.action_queue.put((self.agent_id, action.item()))

                # waiting for result:
                observation, reward, done, info = self.continue_queue.get()

                if k % 1000 == 0 and self.DEBUG:
                    print(
                    "new state=" + str(get_state_as_pair(observation)) + ", rewards=" + str(reward) + ", done=" + str(
                        done))

                # if done and reward != 1.0:
                # 	if observation == 5 or observation == 7 or observation == 11 or observation == 12:
                # 		reward = -1.0

                step_data = [get_state_repr(observation), action, log_prob, reward, done, info]
                episode_series.append(step_data)
                last_reward = reward
                reward_acum.append(reward)

            # FINISH EPISODE

            reward_per_day.append(np.sum(reward_acum))

            if len(score) < 100 :
                score.append(np.sum(reward_acum))
            else:
                score[k % 100] = np.sum(reward_acum)

            if k % 1000 == 0 and self.DEBUG:
                print(
                    "Episode {} finished after {} timesteps with r={}. Running score: {}. Times trained: {}. Times reached goal: {}.".format(
                        k, len(episode_series), np.sum(reward_acum), np.mean(score), times_trained, times_reach_goal))
                times_trained = 0
                times_reach_goal = 0


            policy_loss = []
            rewards_list = []
            for i in range(len(episode_series)):
                j = i
                G = 0
                # alpha = 1 / len(episode_series)

                # get the log_prob of the last state:
                gamma_cum = 1

                while j < len(episode_series):
                    [observation, action, log_prob, reward, done, info] = episode_series[j]
                    G = G + reward * gamma_cum

                    gamma_cum = gamma_cum * self.GAMMA
                    j = j + 1

                [observation, action, log_prob, reward, done, info] = episode_series[i]

                policy_loss.append(G * -log_prob)

                rewards_list.append(G)

            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
            policy_loss = []


            times_trained = times_trained + 1


            if reward > 0.0:
                times_reach_goal = times_reach_goal + 1

        self.results.put(reward_per_day) # MP.Queue()


        #print("reward_per_day")
        #print(reward_per_day)
