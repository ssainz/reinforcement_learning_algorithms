
import numpy as np


class FleetEnv():


    def __init__(self):

        self.len = 6

        self.city = np.zeros(self.len,self.len)
        self.rewards = np.zeros(self.len, self.len)

        self.high_mean = 4
        self.high_std = 1

        self.low_mean = 2
        self.low_std = 1

        self.time = 0
        self.distance = {}

    def step(self, action):

        action_row = int((action+1) / self.len)
        reminder = (action+1) % self.len
        action_col = reminder
        if reminder == 0:
            action_row = action_row - 1
            action_column = self.len - 1


        # calculate distance to previous:

        

        # generate rewards
        generate_rewards(self)

        # get rewards



    def generate_rewards(self):

        if self.time == 8:

            # High rewards in core


            center = round((self.len - 1) / 2)

            for i in range(center - 1, center + 2):
                for j in range(center - 1, center + 2):
                    self.rewards[i, j] = np.random.normal(self.low_mean, self.low_std)


            self.rewards[center, center] = np.random.normal(self.high_mean, self.high_std)


        if self.time == 20:

            # High rewards in border
            for i in range(0, self.len):
                for j in range(0, self.len):
                    self.rewards[i, j] = np.random.normal(self.low_mean, self.low_std)


            for i in range(1, self.len - 1):
                for j in range(1, self.len - 1):
                    self.rewards[i, j] = 0.0


