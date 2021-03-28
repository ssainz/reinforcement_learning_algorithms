import gym
from gym import register
from gym import envs
class Env:
    def __init__(self, _env, env_config):
        self.env = _env
        self.env_config = env_config
    def reset(self):
        return
    def step(self, action):
        observation = ""
        reward = 1
        done = 1
        info = 1
        return observation, reward, done, info
    def close(self):
        return

def EnvConstructor(env_config):
    if env_config["env_type"] == "gym":
        _env = gym.make(env_config["name"])
        return Env(_env, env_config)
    elif env_config["env_type"] == "gym-registry":
        all_envs = envs.registry.all()
        env_ids = [env_spec.id for env_spec in all_envs]
        if env_config["register"]["id"] not in env_ids:
            register(
                id=env_config["register"]["id"],
                entry_point=env_config["register"]["entry_point"],
                kwargs=env_config["register"]["kwargs"],
                max_episode_steps=env_config["register"]["max_episode_steps"],
                reward_threshold=env_config["register"]["reward_threshold"]
            )
        _env = gym.make(env_config["register"]["id"])
        return Env(_env, env_config)
    elif env_config["env_type"] == "constructor_provided":
        env_constructor = env_config["env_constructor_instance"]
        _env = env_constructor.construct(env_config)
        return Env(_env, env_config)