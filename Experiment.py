from DAO import dao
from Env import Env
from Env import EnvConstructor
from RobotConstructor import RobotConstructor
from Robot import Robot
from World import World


class Experiment:
    def __init__(self, episodes, frequency_checks, group_id, learn_config, env_config, repeats=3):
        self.episodes = episodes
        self.frequency_checks = frequency_checks
        self.group_id = group_id
        self.repeats = repeats
        self.learn_config = learn_config
        self.env_config = env_config
    def experiment(self):
        best_perf_world = None
        for repeat in range(self.repeats):
            robot = RobotConstructor(self.learn_config)
            env = EnvConstructor(self.env_config)
            world = World(robot, env, self.episodes, self.frequency_checks, group_id=self.group_id)
            world.live()
            if best_perf_world is None:
                best_perf_world = world
            elif world.results[-1][0] > best_perf_world.results[-1][0]:
                best_perf_world = world
        dao_object = dao() # store results
        dao_object.save_world(best_perf_world)
        dao_object.close()