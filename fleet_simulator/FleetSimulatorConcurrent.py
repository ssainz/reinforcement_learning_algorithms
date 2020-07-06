from FleetSimulatorEnvConcurrent import FleetEnv
from FleetSimulatorAgentConcurrent import Agent
import multiprocessing as mp
from multiprocessing import Process, Queue

# Start Agents
def start_agent(agent_id, sending_queue, response_queue, episodes):
    agent = Agent(agent_id, sending_queue, response_queue, episodes)
    agent.start()


def start_experiment(exp_conf):

    net = exp_conf['net']
    optimizer = optim.RMSprop(net.parameters(), lr=exp_conf['lr'])
    number_of_agents =  exp_conf['num_of_agents']
    episodes = exp_conf['iterations']
    action_to_env = Queue(number_of_agents)

    # Start Queues
    sending_queues = {}
    for agent_id in range(number_of_agents):
        agent_id_sending_queue = Queue(1)
        sending_queues[agent_id] = agent_id_sending_queue

    # Start Env
    env = FleetEnv(action_to_env, sending_queues, number_of_agents, episodes)

    for agent_id in range(number_of_agents):
        p = Process(target=start_agent, args=(env, agent_id,action_to_env, sending_queues[agent_id], episodes, exp_conf))
        p.start()

    # Aggregate results