from Models import get_pi_net
from FleetSimulatorAgent import start_experiment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import generate_name
import datetime

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

#matplotlib.use("TKAgg")
matplotlib.use("Qt5Agg")

experiments = []

experiment_1 = {
    'net': get_pi_net(),
    'iterations': 20,
    'gamma': 0.99,
    'lr': 0.003,
    'DEBUG': False
}

#Experiments SETUP
experiments.append(experiment_1)
experiments.append(experiment_1)
experiments.append(experiment_1)
experiments.append(experiment_1)

iterations = [20000]
net_generators = [get_pi_net]
lrs = [0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06]
gammas = [0.8, 0.85, 0.9, 0.95, 0.99, 1.0]

for it in iterations:
    for net_generator in net_generators:
        for lr in lrs:
            for gamma in gammas:
                cf = {
                    'net': net_generator(),
                    'iterations': it,
                    'gamma':gamma,
                    'lr': lr,
                    'DEBUG': False
                }
                #experiments.append(cf)

results = []
names = []
for exp in experiments:
    print(str(datetime.datetime.now()) + " test starts \t" + generate_name(exp))
    result, name = start_experiment(exp)
    print(str(datetime.datetime.now()) + " test ends \t" + generate_name(exp))
    results.append(result)
    names.append(name)

#Plotting
pref = str(datetime.datetime.now())
res = split(results, 2) #splits in sets of five
nam = split(names, 2)
i = 0
for results, names in zip(res, nam):
    i += 1
    maxx = max([len(result) for result in results])
    x = np.arange(maxx)
    plots = []
    figure = plt.figure()
    for result in results:
        y = np.array(result)
        #print("reward_chart=", reward_chart)
        chart, = plt.plot(x, y, label=name)
        plots.append(chart)

    plt.legend(names, loc='upper left')
    #plt.show()
    figure.savefig("results/" + pref + "_" + str(i) + ".pdf", bbox_inches='tight')