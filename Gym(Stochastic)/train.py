import datetime
import os
import argparse
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Suppress Tensorflow Messages
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set CPU/GPU
import numpy as np
from agent import *
from env_stochasticdelay import Environment

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", help="algorithm")
parser.add_argument("--stochastic", help="use stochastic delays")
parser.add_argument("--delay", help="environment delay")
parser.add_argument("--verbose", help="log files")
args = parser.parse_args()
algorithm = args.algorithm
delay = int(args.delay)

if args.verbose == 'True':
    verbose = True
else:
    verbose = False

if args.stochastic == 'True':
    use_stochastic_delay = True
else:
    use_stochastic_delay = False
'''Log directory'''
# verbose = False
if verbose:
    save_dir = os.getcwd() + '/Results/Results-1.0/Results-'+algorithm
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if use_stochastic_delay:
        log_file_name = save_dir + '/log_sd.txt'
    else:
        log_file_name = save_dir + '/log.txt'
    reward_file_name = save_dir + '/rewards'
    loss_file_name = save_dir + '/loss'
else:
    log_file_name = ""

'''Environment Parameters'''
game = 'CartPole-v0'
seed = 0  # Seed for Env, TF, Numpy
num_frames = 1e6  # Million Frames
num_episodes = 10000
logs = {'log_interval': 100,  # Number of Episodes after which to print output/save batch output
        'log_file_name': log_file_name
        }

'''Parameters of Algorithm'''
algorithm_params = {'algorithm': algorithm,  # normal, delay, IS
                    'batch_size': 32,
                    'gamma': 0.99,
                    'learning_rate': 1e-3,
                    'start_epsilon': 1.0,
                    'stop_epsilon': 1e-3,
                    'epsilon_decay': 1e-4,
                    'use_stochastic_delay': use_stochastic_delay,
                    'delay': delay,
                    'min_delay': 0,
                    'seed': seed
                    }
model_params = {'hidden_units': [200],  # model architecture
                'max_buffer_size': 1000,
                'min_buffer_size': 100,
                'copy_step': 25,  # 1 means no target network
                'max_dimension': 10
                }

'''Runs'''
runs = 10
rewards = {}
losses = {}
time_taken = np.zeros(runs)
model_params['max_dimension'] = min(delay, model_params['max_dimension'])
if verbose:
    if use_stochastic_delay:
        time_file_name = save_dir + '/time_delay_{}_sd'.format(delay)
        if algorithm == 'IS':
            delay_avg = (delay + algorithm_params['min_delay'])/2
            # algorithm_params['gamma'] = np.power(algorithm_params['gamma'], 1/delay_avg)
    else:
        time_file_name = save_dir + '/time_delay_{}'.format(delay)
for run in range(runs):
    if verbose:
        if use_stochastic_delay:
            reward_file_name_cur = reward_file_name + '_delay_{}_sd_run_{}'.format(delay, run)
            loss_file_name_cur = loss_file_name + '_delay_{}_sd_run_{}'.format(delay, run)
        else:
            reward_file_name_cur = reward_file_name + '_delay_{}_run_{}'.format(delay, run)
            loss_file_name_cur = loss_file_name + '_delay_{}_run_{}'.format(delay, run)
    '''Set seed'''
    seed = run
    algorithm_params['seed'] = seed
    algorithm_params['delay'] = delay
    '''Write Parameters to log_file'''
    if verbose:
        with open(log_file_name, "a") as f:
            f.write('Environment: {}, Frames: {}\n'.format(game, num_frames))
            f.write('Algorithm Parameters: {} \n'.format(algorithm_params))
            f.write('Model Parameters: {} \n'.format(model_params))
            f.write('Run: {} \n'.format(run))
            f.flush()
    '''Initialize Environment & Model'''
    env = Environment(seed, game, algorithm_params['gamma'],
                      algorithm_params['use_stochastic_delay'], algorithm_params['delay'], algorithm_params['min_delay'])
    '''Train the Agent'''
    start_time = time.time()
    reward_history, loss_history = train_agent(env, num_frames, model_params, algorithm_params, logs, verbose)
    end_time = time.time()
    time_taken[run] = end_time - start_time
    if verbose:
        with open(log_file_name, "a") as f:
            f.write('Time taken: {}\n'.format(time_taken))
            f.flush()
    '''Store the results'''
    rewards = reward_history
    losses = loss_history
    '''Save Rewards and Losses'''
    if verbose:
        np.save(reward_file_name_cur, rewards)
        np.save(loss_file_name_cur, losses)
        np.save(time_file_name, time_taken)
