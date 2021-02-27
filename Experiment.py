import gym
import torch

import Env
import Robot
from World import World
from RobotReinforce import RobotReinforce

class Experiment:
    def experiment(self, robot: Robot, env: Env):
        world = World(robot, env)
        world.live()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    env = gym.make('FrozenLake-v0')
    robot = RobotReinforce()
    exp = Experiment()
    exp.experiment(robot, env)
