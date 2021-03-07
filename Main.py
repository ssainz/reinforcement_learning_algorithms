import Env
import Robot
from DAO import dao
from RobotConstructor import RobotConstructor
from Experiment import Experiment


if __name__ == "__main__":

    learn_configs = [
        {
            "robot_type": "RobotDQN",
            "net_config": {
                "layers": [16, 20, 4],
                "non_linear_function": "relu"
            },
            "gamma": 0.99,
            "lr": 0.01
        },
        {
            "robot_type": "RobotDDQN",
            "net_config": {
                "layers": [16, 120, 84, 4],
                "non_linear_function": "relu"
            },
            "gamma": 0.99,
            "lr": 0.01
        },
        {
            "robot_type": "RobotReinforce",
            "net_config": {
                "layers": [16, 120, 84, 4],
                "non_linear_function": "relu"
            },
            "gamma": 0.99,
            "lr": 0.001
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
    dao_obj = dao()
    group_id = dao_obj.get_latest_group_id() + 1
    for learn_config in learn_configs:
        for env_config in env_configs:
            robot = RobotConstructor(learn_config)
            env = Env.EnvConstructor(env_config)
            episodes = 20000
            exp = Experiment(episodes=episodes, frequency_checks=episodes/10, group_id=group_id)
            exp.experiment(robot, env)
