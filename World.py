import torch
import Robot
import Env


class World:
    def __init__(self, robot: Robot, env: Env, episodes, frequency_checks):
        self.robot = robot
        self.env = env
        self.MAX_ITERATIONS = episodes
        self.FREQUENCY_CHECKS = frequency_checks
        self.results = []
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
                (new_observation, reward, done, info) = env.step(robot.decide(observation))  # take a random action
                robot.add_observation_reward(observation, new_observation, reward)
                robot.learn_at_end_of_step()
                # print(new_observation)
                observation = new_observation
            robot.learn_at_end_of_episode()
            if (iteration + 1) % self.FREQUENCY_CHECKS == 0:
                self.results.append([robot.wins])
                print("Wins per {}: {}".format(self.FREQUENCY_CHECKS, robot.wins))
                robot.wins = 0
        env.close()

