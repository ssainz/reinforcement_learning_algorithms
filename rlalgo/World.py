import torch
from rlalgo import Env, Robot
import traceback

class World:
    def __init__(self, robot: Robot, env: Env, episodes, frequency_checks, group_id):
        self.robot = robot
        self.env = env
        self.MAX_ITERATIONS = episodes
        self.FREQUENCY_CHECKS = frequency_checks
        self.results = []
        self.group_id = group_id
    def live(self):
        torch.autograd.set_detect_anomaly(True)
        env = self.env.env # actual gym or other env
        robot = self.robot
        hits = 0

        for iteration in range(self.MAX_ITERATIONS):
            print("World , robot %s, iteration %d" %(str(self.robot.config["robot_type"]), iteration))
            observation = env.reset() # SAV loads network
            done = False
            while not done:
                # env.render()
                action = robot.decide(observation)
                try:
                    (new_observation, reward, done, info) = env.step(action)  # take a random action
                    robot.add_observation_reward(observation, action, new_observation, reward, done)
                    robot.learn_at_end_of_step()
                    # print(new_observation)
                    observation = new_observation
                except Exception:
                    print(traceback.format_exc())
                    done = True # break while loop, continue next iteration using same envz
            robot.learn_at_end_of_episode()
            if (iteration) % self.FREQUENCY_CHECKS == 0:
                self.results.append([robot.get_cum_reward()])
                print("Avg. reward: {}, in last {} iterations".format( robot.get_cum_reward() / self.FREQUENCY_CHECKS, self.FREQUENCY_CHECKS))
                robot.set_cum_reward(0)
        env.close()

