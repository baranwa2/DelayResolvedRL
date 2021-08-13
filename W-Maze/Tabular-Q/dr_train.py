import numpy as np
import matplotlib.pyplot as plt
import datetime
# from tqdm import tqdm
from pathlib import Path
from env import Environment
from dr_agent import Agent

'''Augmented State Implementation of Tabular-Q'''
algorithms = ['SARSA', 'Q']
for algorithm in algorithms:
    delays = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    runs = 10
    for delay in delays:
        for run in range(runs):
            seed = np.random.seed(run)
            env = Environment(seed, delay)  # Initialize Environment
            agent = Agent(env.state_space, env.num_actions, delay)  # Initialize Q-learning agent
            episodes = int(1e5)
            alpha = 0.1
            gamma = 0.995
            lambda_trace = 0.0  # lambda value for eligibility traces
            all_rewards = []
            file_dir = 'Results-v1.0-cumulative/' + 'maze_' + algorithm + '_lambda_' + str(lambda_trace) + '_' + str(delay)
            Path(file_dir).mkdir(parents=True, exist_ok=True)
            filename = file_dir + '/' + str(run)
            for episode in range(episodes):
                rewards = 0
                state = env.state
                agent.fill_up_buffer(state)
                action, agent_action = agent.choose_action(state)
                if episode % 1000 == 0:
                    agent.update_epsilon(0.995*agent.epsilon)
                for ep_step in range(env.turn_limit):
                    next_action_history = tuple(agent.actions_in_buffer)
                    action_history = tuple(agent.actions_in_buffer_prev)
                    next_state, reward, done = env.step(state, action)
                    next_action, agent_next_action = agent.choose_action(next_state)
                    if algorithm == 'Q':
                        # agent.E[state[0], state[1], agent_action] += 1  # accumulating traces
                        """Q-Learning"""
                        td_error = reward + gamma*np.max(agent.Q_values[(next_state[0], next_state[1]) + next_action_history]) \
                                   - agent.Q_values[(state[0], state[1]) + action_history + (agent_action,)]
                    elif algorithm == 'SARSA':
                        # agent.E[(state[0], state[1]) + agent_action] += 1  # accumulating traces
                        """SARSA"""
                        td_error = reward + gamma*agent.Q_values[(next_state[0], next_state[1]) + next_action_history +
                                                           (agent_next_action,)] - agent.Q_values[(state[0], state[1]) +
                                                                                                  action_history + (
                                                                                                  agent_action,)]
                    else:
                        raise Exception('Algorithm Undefined')
                    agent.Q_values[(state[0], state[1]) + action_history + (agent_action,)] += alpha * td_error
                    '''Trace Calculation'''
                    # for s_x in range(env.breadth):
                    #     for s_y in range(env.length):
                    #         for a in range(env.num_actions):
                    #             agent.Q_values[s_x, s_y, a] += \
                    #                 alpha * td_error * agent.E[s_x, s_y, a]
                    #             agent.E[s_x, s_y, a] = \
                    #                 gamma * lambda_trace * agent.E[s_x, s_y, a]
                    state = next_state
                    action = next_action
                    agent_action = agent_next_action
                    rewards += reward
                    if done:
                        '''Verbose'''
                        # print('\nEpisode: {}, Rewards: {} \r'.format(episode, rewards))
                        break
                all_rewards.append(rewards)
                np.save(filename, all_rewards)
