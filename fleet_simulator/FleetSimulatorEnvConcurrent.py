
import numpy as np

# https://github.com/YingChen/cs229/blob/master/cs229_project/pacingenv3.py

class FleetEnv():


    def init(self, receiving_queue, sending_queues, number_of_agents, episodes, DEBUG):
        self.len = 6
        self.tot_distance = self.len * 2

        self.city = np.zeros((self.len, self.len))
        self.city[0,0] = number_of_agents

        self.rewards = np.zeros((self.len, self.len))
        self.init_rewards()

        self.high_mean = 4
        self.high_std = 1

        self.low_mean = 2
        self.low_std = 1

        self.time = 0
        self.distance = {}
        self.curr_location_col = 0
        self.curr_location_row = 0
        self.receiving_queue = receiving_queue
        self.sending_queues = sending_queues
        self.number_of_agents = number_of_agents
        self.episodes = episodes
        self.DEBUG = DEBUG

        self.agents = {}
        for agent in range(number_of_agents):
            self.agents[agent] = (0,0)

    def __init__(self,  receiving_queue, sending_queues, number_of_agents, episodes, DEBUG):
        self.init(receiving_queue, sending_queues, number_of_agents, episodes, DEBUG)


    def reset(self):
        self.init()
        return self.city


    def start(self):

        if self.DEBUG:
            print("starting env.")


        for episode in range(self.episodes):

            for step in range(24):
                self.time = step
                if self.DEBUG:
                    print("Getting agent's actions")
                # Get all agent's actions
                new_agents_positions = {}
                new_agents_positions_matrix = np.zeros((self.len, self.len))
                for agent in range(self.number_of_agents):
                    agent_id, agent_action = self.receiving_queue.get()
                    agent_action_row, agent_action_col = self.get_action_rows_cols(agent_action)
                    new_agents_positions[agent_id] = (agent_action_row, agent_action_col)
                    new_agents_positions_matrix[agent_action_row, agent_action_col] += 1

                if self.DEBUG:
                    print("Got agent's actions")

                # generate rewards
                self.generate_rewards()



                # calculate agent's rewards
                agents_rewards = {}
                for agent_id in self.agents.keys():
                    agent_travelled_distance = abs(self.agents[agent_id][0] - new_agents_positions[agent_id][0]) + abs(self.agents[agent_id][1] - new_agents_positions[agent_id][1])
                    agents_rewards[agent_id] = self.rewards[new_agents_positions[agent_id][0], new_agents_positions[agent_id][1]] * ( 1 - 0.8 * (agent_travelled_distance / self.tot_distance))
                    #if self.rewards[new_agents_positions[agent_id][0], new_agents_positions[agent_id][1]] > 0:
                    #    self.rewards[new_agents_positions[agent_id][0], new_agents_positions[agent_id][1]] -= 1
                    self.rewards[new_agents_positions[agent_id][0], new_agents_positions[agent_id][1]] /=  new_agents_positions_matrix[new_agents_positions[agent_id][0], new_agents_positions[agent_id][1]]


                self.agents = new_agents_positions

                # update state:
                for i in range(0, self.len):
                    for j in range(0, self.len):
                        self.city[i, j] = 0.0

                for agent_id in self.agents.keys():
                    i = self.agents[agent_id][0]
                    j = self.agents[agent_id][1]
                    self.city[i, j] += 1


                done = False
                if step >= 23:
                    done = True

                if self.DEBUG:
                    print("Replying all agents")

                for agent_id in agents_rewards:
                    self.sending_queues[agent_id].put((self.city, agents_rewards[agent_id], done, {} ))

                if self.DEBUG:
                    print("Replied all agents")


    def get_action_rows_cols(self, action):
        action_row = int((action) / self.len)
        reminder = (action) % self.len
        action_column = reminder
        return action_row, action_column

    def generate_rewards(self):

        if self.time == 8 :

            # High rewards in core
            for i in range(0, self.len ):
                for j in range(0, self.len ):
                    self.rewards[i, j] = 0.0

            center = int(round((self.len - 1) / 2))

            for i in range(center - 1, center + 2):
                for j in range(center - 1, center + 2):
                    #self.rewards[i, j] = np.random.normal(self.low_mean, self.low_std)
                    #self.rewards[i, j] = 5
                    self.rewards[i, j] = 1


            #self.rewards[center, center] = 10
            self.rewards[center, center] = 1


        if self.time == 20:

            # High rewards in border
            for i in range(0, self.len):
                for j in range(0, self.len):
                    #self.rewards[i, j] = np.random.normal(self.low_mean, self.low_std)
                    #self.rewards[i, j] = 5
                    self.rewards[i, j] = 1


            for i in range(1, self.len - 1):
                for j in range(1, self.len - 1):
                    self.rewards[i, j] = 0.0

        #else:
        #    for i in range(0, self.len ):
        #        for j in range(0, self.len ):
        #            self.rewards[i, j] = 0.0

    def init_rewards(self):
        # High rewards in border
        for i in range(0, self.len):
            for j in range(0, self.len):
                # self.rewards[i, j] = np.random.normal(self.low_mean, self.low_std)
                #self.rewards[i, j] = 5
                self.rewards[i, j] = 1

        for i in range(1, self.len - 1):
            for j in range(1, self.len - 1):
                self.rewards[i, j] = 0.0