import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
import os

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({'font.size': 13})


def running_mean(x, n):
    cumulative_sum = np.cumsum(np.insert(x, 0, 0))
    return (cumulative_sum[n:] - cumulative_sum[:-n]) / float(n)


def get_file(index, ver):
    save_dir = os.getcwd() + '/Results/Results-{}/'.format(ver)  # Save Directory
    files_list = os.listdir(save_dir)
    if ver == '6.8':
        with open(save_dir + files_list[index] + '/log.txt', 'r') as f:
            env_name = f.readline().split(',')[0].split(':')[1]  # Gets the environment name
    else:
        with open(save_dir + files_list[index] + '/log_sd.txt', 'r') as f:
            env_name = f.readline().split(',')[0].split(':')[1]  # Gets the environment name
    file_name = save_dir + files_list[index]  # Final files directory
    return env_name, file_name


def plot_reward(index, runs, delays, n):
    env_name, file_name = get_file(index)
    plt.figure()
    if index == -1:
        plt.title('DQN')
        save_file = os.getcwd() + '/Plots/v{}/rewards_normal.pdf'.format(ver)
    if index == -2:
        plt.title('DQN+IS')
        save_file = os.getcwd() + '/Plots/v{}/rewards_IS.pdf'.format(ver)
    if index == -3:
        plt.title('delay-DQN')
        save_file = os.getcwd() + '/Plots/v{}/rewards_delay.pdf'.format(ver)
    for delay in delays:
        episodes = 10000
        X_axis = np.arange(episodes)
        rewards_plot = np.zeros([runs, episodes])
        for run in range(runs):
            if delay == 'stochastic':
                rewards = np.load(file_name[index] + '/rewards_delay_20_sd_run_{}.npy'.format(run), allow_pickle=True)[
                    ()]
            else:
                rewards = np.load(file_name + '/rewards_delay_{}_run_{}.npy'.format(delay, run), allow_pickle=True)[()]
            # plt.plot(running_mean(rewards, n), alpha=0.25, linestyle='-.', color='blue')
            rewards_plot[run] = rewards[0:episodes]
        rewards_mean = np.mean(rewards_plot, axis=0)
        rewards_deviation = np.std(rewards_plot, axis=0) / np.sqrt(runs)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        # plt.ylim(0, 210)
        plt.plot(running_mean(rewards_mean, n), label='delay={}'.format(delay))
        # plt.fill_between(X_axis, rewards_mean+rewards_deviation, rewards_mean-rewards_deviation, alpha=1.5)
    plt.legend(title='Delays', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(save_file, bbox_inches="tight")
    plt.savefig(file_name + '/rewards.pdf', bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def compare_learning_curves(indices, label, ver, runs, delay, n=1000):
    env_name = []
    file_name = []
    for index in indices:
        env, file = get_file(index, ver)
        env_name.append(env)
        file_name.append(file)
    # if not env_name.count(env_name[0]) == len(env_name):  # Check if all the environments are same
    #     raise Exception('Environments are different')
    plt.figure()
    plt.title(env_name[0])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    colors = ['blue', 'tab:orange', 'green']
    for index in range(len(indices)):
        for run in range(runs):
            if delay == 'stochastic':
                rewards = \
                    np.load(file_name[index] + '/rewards_delay_10_sd_run_{}.npy'.format(run), allow_pickle=True)[()]
            else:
                rewards = \
                    np.load(file_name[index] + '/rewards_delay_{}_run_{}.npy'.format(delay, run), allow_pickle=True)[()]
            plt.plot(running_mean(rewards, n), label=label[index] if run == 0 else '', color=colors[index], alpha=0.5)
    plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_dir = os.getcwd() + '/Plots/v{}/'.format(ver)
    try:
        plt.savefig(save_dir + '/rewards_curves.pdf', bbox_inches="tight")
    except FileNotFoundError:
        os.makedirs(os.getcwd() + '/Plots/v{}'.format(ver))
        plt.savefig(save_dir + '/rewards_curves.pdf', bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_losses(index, runs, n):
    env_name, file_name = get_file(index)
    plt.figure()
    plt.title(env_name)
    losses = np.load(file_name + '/loss.npy', allow_pickle=True)[()]
    episodes = len(losses[0])
    X_axis = np.arange(episodes)
    losses_plot = np.zeros([runs, episodes])
    for run in range(runs):
        losses_plot[run] = losses[run]
    losses_mean = np.mean(losses_plot, axis=0)
    losses_deviation = np.std(losses_plot, axis=0) / np.sqrt(runs)
    plt.xlabel('Episodes')
    plt.ylabel('Losses')
    plt.plot(running_mean(losses_mean, n))
    plt.savefig(file_name + '/losses.pdf')
    plt.show()


def compare_algorithms(indices, label, runs, delays, ver, colors):
    env_name = []
    file_name = []
    file_name_sd = []
    for index in indices:
        env, file = get_file(index, ver)
        env_name.append(env)
        file_name.append(file)
    for index in indices:
        _, file_sd = get_file(index, ver)
        file_name_sd.append(file_sd)
    # if not env_name.count(env_name[0]) == len(env_name):  # Check if all the environments are same
    #     raise Exception('Environments are different')
    plt.figure()
    plt.title(env_name[0], fontsize=20)
    plt.xlabel('Delay')
    plt.ylabel('Rewards')
    for index in range(len(indices)):
        count = 0
        X_axis = list(map(str, delays))
        r_mean = np.zeros(len(delays))
        r_std = np.zeros(len(delays))
        episodes = 10000
        # rewards_plot = np.zeros([runs, episodes])
        rewards_plot = np.zeros(runs)
        for delay in delays:
            for run in range(runs):
                if delay == 'stochastic':
                    rewards = \
                        np.load(file_name_sd[index] + '/rewards_delay_10_sd_run_{}.npy'.format(run), allow_pickle=True)[
                            ()]
                    rewards_plot[run] = np.mean(rewards)
                    print('Algorithm: {} Delay: Stochastic Run: {} Reward: {}'.format(index, run, rewards_plot[run]))
                else:
                    rewards = \
                        np.load(file_name[index] + '/rewards_delay_{}_run_{}.npy'.format(delay, run),
                                allow_pickle=True)[()]
                    rewards_plot[run] = np.mean(rewards)
                    print('Algorithm: {} Delay: {} Run: {} Reward: {}'.format(index, delay, run, rewards_plot[run]))
            rewards_mean = rewards_plot
            rewards_mean = np.mean(rewards_plot, axis=0)
            rewards_deviation = np.std(rewards_plot, axis=0) / np.sqrt(runs)
            r_mean[count] = np.mean(rewards_mean)
            r_std[count] = np.mean(rewards_deviation)
            count += 1
        if label[index] == 'DQN+IS':
            alg = 'DRDQN'
        else:
            alg = label[index]
        # plt.plot(X_axis, r_mean, marker='o', label=alg, color=colors[label[index]])
        plt.errorbar(X_axis, r_mean, marker='o', yerr=r_std, label=alg, color=colors[label[index]], uplims=True,
                     lolims=True)
    # plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.xlabel('Delays', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('Rewards', fontsize=16)
    plt.yticks(fontsize=16)
    save_dir = os.getcwd() + '/Plots/v{}/'.format(ver)
    try:
        plt.savefig(save_dir + '/rewards_comparison.pdf', bbox_inches="tight")
    except FileNotFoundError:
        os.makedirs(os.getcwd() + '/Plots/v{}'.format(ver))
        plt.savefig(save_dir + '/rewards_comparison.pdf', bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_time(indices, labels, delays, ver):
    plt.figure()
    env_name = []
    file_name = []
    file_name_sd = []
    for index in indices:
        env, file = get_file(index, ver)
        env_name.append(env)
        file_name.append(file)
    for index in indices:
        _, file_sd = get_file(index, ver)
        file_name_sd.append(file_sd)
    for index in range(len(indices)):
        time = np.zeros([len(delays)])
        X_axis = list(map(str, delays))
        for delay in range(len(delays)):
            if delays[delay] == 'stochastic':
                time[delay] = np.mean(np.load(file_name_sd[index] + '/time_delay_6_sd.npy'))
            else:
                time[delay] = np.mean(np.load(file_name[index] + '/time_delay_{}.npy'.format(delays[delay])))
        plt.plot(X_axis, time / 3600, label=labels[index], marker='o')
    plt.title(env_name[0])
    plt.xlabel('Delays')
    plt.ylabel('Average Hours per run')
    plt.legend()
    save_dir = os.getcwd() + '/Plots/v{}/'.format(ver)
    try:
        plt.savefig(save_dir + '/time_comparison.pdf', bbox_inches="tight")
    except FileNotFoundError:
        os.makedirs(os.getcwd() + '/Plots/v{}'.format(ver))
        plt.savefig(save_dir + '/time_comparison.pdf', bbox_inches="tight")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    runs = 10
    delays = [2, 4, 6, 8, 10]  # , 'stochastic']
    ver = '5.5'
    compare_indices = [-2, -1, -3]
    labels = ['DQN+IS', 'DQN', 'delay-DQN']
    colors = {'DQN+IS': u'#1f77b4', 'delay-DQN': 'red', 'DQN': u'#2ca02c'}
    compare_algorithms(compare_indices, labels, runs, delays, ver, colors)
    plot_time(compare_indices, labels, delays, ver)

    delay = 'stochastic'
    compare_learning_curves(compare_indices, labels, ver, runs, delay)
