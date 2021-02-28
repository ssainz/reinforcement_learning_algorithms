import Env
import Robot
from RobotConstructor import RobotConstructor
from Experiment import Experiment


if __name__ == "__main__":

    learn_configs = [
        {
            "robot_type": "RobotReinforce",
            "net_config": {
                "layers": [16, 120, 84, 4],
                "non_linear_function": "relu"
            },
            "gamma": 0.99,
            "lr": 0.0001
        }
    ]
    env_configs = [
        {
            "env_type": "gym",
            "name": "FrozenLake-v0"
        },
        {
            "env_type": "gym-registry",
            "name": "FrozenLakeNotSlippery-v0",
            "register": {
                "id": "FrozenLakeNotSlippery-v0",
                "entry_point": 'gym.envs.toy_text:FrozenLakeEnv',
                "kwargs": {'map_name': '4x4', 'is_slippery': False},
                "max_episode_steps": 100,
                "reward_threshold": 0.78 # optimum = .8196
            }
        }
    ]
    for learn_config in learn_configs:
        for env_config in env_configs:
            robot = RobotConstructor(learn_config)
            env = Env.EnvConstructor(env_config)
            episodes = 1000
            exp = Experiment(episodes=episodes, frequency_checks=episodes/10)
            exp.experiment(robot, env)
