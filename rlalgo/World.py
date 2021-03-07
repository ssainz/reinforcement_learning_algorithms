import torch
from rlalgo import Env, Robot


class World:
    def __init__(self, robot: Robot, env: Env, episodes, frequency_checks, group_id):
        self.robot = robot
        self.env = env
        self.MAX_ITERATIONS = episodes
        self.FREQUENCY_CHECKS = frequency_checks
        self.results = []
        self.group_id = group_id
    def live(self):
        torch.autograd.set_detect_anomaly(True)
        env = self.env.env # actual gym or other env
        robot = self.robot
        hits = 0

        for iteration in range(self.MAX_ITERATIONS):
            observation = env.reset()
            done = False
            while not done:
                # env.render()
                action = robot.decide(observation)
                (new_observation, reward, done, info) = env.step(action)  # take a random action
                robot.add_observation_reward(observation, action, new_observation, reward, done)
                robot.learn_at_end_of_step()
                # print(new_observation)
                observation = new_observation
            robot.learn_at_end_of_episode()
            if (iteration + 1) % self.FREQUENCY_CHECKS == 0:
                self.results.append([robot.cum_reward])
                print("Avg. reward: {}, in last {} iterations".format( robot.cum_reward / self.FREQUENCY_CHECKS, self.FREQUENCY_CHECKS))
                robot.cum_reward = 0
        env.close()

