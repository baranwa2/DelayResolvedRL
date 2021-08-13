import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({'font.size': 13})

"""Plotting"""
algorithms = ['Q']
# delays = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
delays = [2, 4, 6, 8, 10]
runs = 10
lambda_trace = 0.0
for algorithm in algorithms:
    rewards = {}
    rewards_err = {}
    episodes = 10000
    for delay in delays:
        reward_runs = np.zeros(runs)
        for run in range(runs):
            filename = 'Maze-Tabular\\Results-v3-cumulative\\maze_' + algorithm + '_lambda_' + str(lambda_trace) + '_' + str(delay) \
                       + '\\' + str(run) + '.npy'
            reward_current = np.load(filename)[-episodes-1:-1]
            reward_runs[run] = np.mean(reward_current)
        rewards[delay] = np.mean(reward_runs)
        rewards_err[delay] = np.std(reward_runs, axis=0)
    alg = 'DRQ'
    color = u'#1f77b4'
    # plt.plot(list(rewards.keys()), list(rewards.values()), marker='o', label=alg, color=color)
    plt.errorbar(list(rewards.keys()), list(rewards.values()), yerr=list(rewards_err.values()),
                 uplims=True, lolims=True,  label=alg, color=color)
    plt.title('W-Maze', fontsize=20)
    plt.xticks(list(rewards.keys()))

algorithms = ['Q', 'dQ']
for algorithm in algorithms:
    rewards = {}
    rewards_err = {}
    for delay in delays:
        reward_runs = np.zeros(runs)
        for run in range(runs):
            filename = 'Maze-Tabular\\Results-v3\\maze_' + algorithm + '_lambda_' + str(0.0) + '_' + str(delay) \
                       + '\\' + str(run) + '.npy'
            reward_current = np.load(filename)[-episodes-1:-1]
            reward_runs[run] = np.mean(reward_current)
        rewards[delay] = np.mean(reward_runs)
        rewards_err[delay] = np.std(reward_runs, axis=0)
    if algorithm == 'dQ':
        alg = 'delay-Q'
        color = 'red'
    else:
        alg = 'Q'
        color = u'#2ca02c'
    # plt.plot(list(rewards.keys()), list(rewards.values()), marker='o', label=alg, color=color)
    plt.errorbar(list(rewards.keys()), list(rewards.values()), yerr=list(rewards_err.values()),
                 uplims=True, lolims=True,  label=alg, color=color)

plt.legend()
plt.xlabel('Delays', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('Rewards', fontsize=16)
plt.yticks(fontsize=16)
save_dir = os.getcwd() + '/Maze-Tabular/Plots/'
try:
    plt.savefig(save_dir + '/rewards_comparison.pdf', bbox_inches="tight")
except FileNotFoundError:
    os.makedirs(os.getcwd() + '/Maze-Tabular/Plots/')
    plt.savefig(save_dir + '/rewards_comparison.pdf', bbox_inches="tight")
plt.tight_layout()
plt.show()
plt.show()
