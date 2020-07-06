from FleetSimulatorEnvConcurrent import FleetEnv
from FleetSimulatorAgentConcurrent import Agent
import multiprocessing as mp
from multiprocessing import Process, Queue
from utils import generate_name
import numpy as np

def __repr__(self):
    return '<%s.%s object at %s>' % (
        self.__class__.__module__,
        self.__class__.__name__,
        hex(id(self))
    )

# Start Agents
def start_agent(env, agent):
    #print(mp.current_process(), __repr__(agent))
    agent.start()


def start_experiment(exp_conf):

    number_of_agents =  exp_conf['num_of_agents']
    episodes = exp_conf['iterations']
    action_to_env = Queue(number_of_agents)
    agents = {}
    # Start Queues
    sending_queues = {}
    results = {}
    for agent_id in range(number_of_agents):
        agent_id_sending_queue = Queue(1)
        sending_queues[agent_id] = agent_id_sending_queue
        results[agent_id] = Queue()
        agents[agent_id] = Agent( agent_id, action_to_env, sending_queues[agent_id], episodes, exp_conf, results[agent_id])

    # Start Env
    env = FleetEnv(action_to_env, sending_queues, number_of_agents, episodes, exp_conf["DEBUG"])

    exp_name = generate_name(exp_conf)

    for agent_id in range(number_of_agents):
        p = Process(target=start_agent, args=(env, agents[agent_id]))
        p.start()

    env.start()

    # Aggregate results
    res = None
    for agent_id in range(number_of_agents):
        if res is None:
            res = np.array(results[agent_id].get())
        else:
            res += np.array(results[agent_id].get())
        #print("res")
        #print(res)

    return res , exp_name