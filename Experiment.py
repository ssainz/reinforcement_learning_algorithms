from DAO import dao
from Env import Env
from Robot import Robot
from World import World


class Experiment:
    def __init__(self, episodes, frequency_checks):
        self.episodes = episodes
        self.frequency_checks = frequency_checks
    def experiment(self, robot: Robot, env: Env):
        world = World(robot, env, self.episodes, self.frequency_checks)
        world.live()
        dao_object = dao() # store results
        dao_object.save_world(world)
        dao_object.close()