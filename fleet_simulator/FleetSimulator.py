
import numpy as np

# https://github.com/YingChen/cs229/blob/master/cs229_project/pacingenv3.py

class FleetEnv():


    def __init__(self):
        init(self)


    def reset(self):
        init(self)
        return self.city


    def init(self):
        self.len = 6

        self.city = np.zeros(self.len, self.len)
        self.rewards = np.zeros(self.len, self.len)

        self.high_mean = 4
        self.high_std = 1

        self.low_mean = 2
        self.low_std = 1

        self.time = 0
        self.distance = {}
        self.curr_location_col = 0
        self.curr_location_row = 0

    def step(self, action):

        action_row = int((action) / self.len)
        reminder = (action) % self.len
        action_column = reminder



        # calculate distance to previous:
        tot_distance = self.len * 2
        curr_distance = abs(self.curr_location_col - action_column) + abs(self.curr_location_row - action_row)


        # generate rewards
        generate_rewards(self)

        # get rewards
        reward = self.rewards[action_row, action_column] * ( 1 - 0.8 * (curr_distance / tot_distance))

        # Update state then next.
        self.curr_location_row = action_column
        self.curr_location_col = action_row

        for i in range(0, self.len):
            for j in range(0, self.len):
                self.city[i, j] = 0.0

        self.city[self.curr_location_row, self.curr_location_col] = 1.0

        self.time += 1

        done = False
        if self.time >= 24:
            done = True

        return self.city, reward, done


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


        else:
            for i in range(0, self.len ):
                for j in range(0, self.len ):
                    self.rewards[i, j] = 0.0