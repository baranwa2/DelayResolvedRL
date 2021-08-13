import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({'font.size': 13})

# results = pd.read_csv('results_cartpole.csv')
# results = pd.read_csv('results_acrobot.csv')
results = pd.read_csv('results_mcar.csv')
# results = pd.read_csv('results_cartpole_reshaped.csv')

runs = 10
delays = np.unique(results['delay_value'])
delays_str = ['2','4','6','8','10','15','20']
algorithms = ['DRDQN', 'Delayed DQN', 'DQN']
colors = {'DRDQN':u'#1f77b4', 'Delayed DQN':u'#ff7f0e', 'DQN':u'#2ca02c'}
count = 0
for algorithm in algorithms:
    x = []
    x_std = []
    for delay in delays:
        avg = np.zeros(runs)
        for run in range(runs):
            avg[run] = results['final_score'][count]
            print(algorithm, delay, run, avg[run])
            if math.isnan(avg[run]):
                avg[run] = -200
            count += 1
        x.append(np.mean(avg))
        x_std.append(np.std(avg)/np.sqrt(runs))
    # plt.plot(delays_str, x, label=algorithm, marker='o', color=colors[algorithm])
    plt.errorbar(delays, x, yerr=x_std, label=algorithm, marker='o', color=colors[algorithm], uplims=True, lolims=True)
plt.xticks(fontsize=16) #[2,4,6,8,10,15,20]
plt.ylabel('Rewards', fontsize=16)
plt.xlabel('Delays', fontsize=16)
plt.yticks(fontsize=16)
# plt.ylim(0,220)
# plt.title('CartPole-v0', fontsize=20)
# plt.title('Acrobot-v1', fontsize=20)
plt.title('MountainCar-v0', fontsize=20)
# plt.title('CartPole-v0 (Reshaped Rewards)', fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()

