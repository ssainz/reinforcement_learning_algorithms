class Env:
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