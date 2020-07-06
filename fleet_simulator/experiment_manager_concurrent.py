from Models import get_pi_net
from FleetSimulatorConcurrent import start_experiment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import generate_name
import datetime

matplotlib.use("Qt5Agg")

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


iterations = [10000]
net_generators = [get_pi_net]
lrs = [0.0003 ]
#lrs = [0.0003,  0.001, 0.006,  0.03, 0.06]
# 0.0003 seems the best option
#gammas = [0.95]
gammas = [0.9, 0.99]

experiments = []
for it in iterations:
    for net_generator in net_generators:
        for lr in lrs:
            for gamma in gammas:
                cf = {
                    'net': "pi_net",
                    'iterations': it,
                    'gamma':gamma,
                    'lr': lr,
                    'DEBUG': False,
                    'num_of_agents': 20
                }
                experiments.append(cf)

results = []
names = []
for exp in experiments:
    print(str(datetime.datetime.now()) + " test starts \t" + generate_name(exp))
    result, name = start_experiment(exp)
    print(str(datetime.datetime.now()) + " test ends \t" + generate_name(exp))
    results.append(result)
    names.append(name)

#print("results")
#print(results)

#Plotting
pref = str(datetime.datetime.now())
res = split(results, 5) #splits in sets of five
nam = split(names, 5)
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
    figure.savefig("results_concurrent/" + pref + "_" + str(i) + ".pdf", bbox_inches='tight')