import numpy as np
from collections import deque
import copy
import random


class Environment:
    """Initialize Environment"""
    def __init__(self, seed, gamma, use_stochastic_delay, delay, min_delay):
        np.random.seed(seed)
        random.seed(seed)
        self.call = 0
        self.breadth = 7
        self.length = 11
        self.state_space = np.empty([self.breadth, self.length], dtype='<U1')
        self.state_space[:] = 'E'
        self.state_space[0] = 'X'
        self.state_space[1:4, self.length // 2 - 2] = 'X'
        self.state_space[1:4, self.length // 2 + 2] = 'X'
        self.state_space[0, self.length // 2 - 1:self.length // 2 + 2] = 'G'
        self.state_space[self.breadth - 1, 0] = 'P'
        self.actions = [1, 2, 3, 4]  # UP, DOWN, LEFT, RIGHT
        self.no_action = 0
        self.index = 0
        self.number_of_actions = len(self.actions)
        self.turn_limit = 300
        self.min_delay = min_delay
        self.delay = self.min_delay
        self.max_delay = delay
        self.use_stochastic_delay = use_stochastic_delay
        self.state_buffer = deque(maxlen=self.max_delay+2)
        self.reward_buffer = deque(maxlen=self.max_delay+2)
        self.done_buffer = deque(maxlen=self.max_delay+2)
        self.state = self.reset()
        self.update_delay()
        self.train = True
        self.step_count = 0
        self.delayed_action = 0
        self.gamma = gamma

    def reset(self):
        x = random.randint(0, self.breadth - 1)
        y = 0
        starting_state = [x, y]
        self.state_space[x, y] = 'P'
        self.step_count = 0
        return starting_state

    def update_delay(self):
        if self.use_stochastic_delay:
            self.delay = random.randint(self.min_delay, self.max_delay)
        else:
            self.delay = self.max_delay

    def step(self, state, action):
        if self.max_delay != 0:
            self.train = True
            if True not in self.done_buffer:
                next_state, rewards, done = self.env_step(action)
            else:
                next_state = state
                rewards = 0
                done = True
            if len(self.state_buffer) < self.delay:  # delay is greater than the number of unobserved states
                self.state_buffer.append(next_state)
                self.reward_buffer.append(rewards)
                self.done_buffer.append(done)
                self.train = False
                return state, 0, False
            elif len(self.state_buffer) > self.delay:  # delay is smaller than the number of unobserved states
                self.state_buffer.append(next_state)
                self.reward_buffer.append(rewards)
                self.done_buffer.append(done)
                rewards = 0
                no_observed_states = len(self.state_buffer) - self.delay
                for i in range(no_observed_states):
                    next_state = self.state_buffer.popleft()
                    gamma = np.power(self.gamma, no_observed_states - (i + 1))
                    rewards += gamma*self.reward_buffer.popleft()  # add all unobserved rewards
                    done = self.done_buffer.popleft()
                self.update_delay()
                if done:
                    self.state_buffer.clear()
                    self.reward_buffer.clear()
                    self.done_buffer.clear()
                return next_state, rewards, done
            else:
                self.state_buffer.append(next_state)
                self.reward_buffer.append(rewards)
                self.done_buffer.append(done)
                delayed_next_state = self.state_buffer.popleft()
                delayed_rewards = self.reward_buffer.popleft()
                delayed_done = self.done_buffer.popleft()
                self.update_delay()
                if delayed_done:
                    self.state_buffer.clear()
                    self.reward_buffer.clear()
                    self.done_buffer.clear()
                return delayed_next_state, delayed_rewards, delayed_done
        else:
            return self.env_step(action)

    def env_step(self, action):
        action += 1  # (0,3) -> (1,4)
        self.step_count += 1
        done = False
        player_position = self.state
        reward = -1
        """UP"""
        if action == 1:
            if player_position[0] - 1 >= 0 and self.state_space[player_position[0] - 1, player_position[1]] != 'X':
                self.state_space[player_position[0], player_position[1]] = 'E'
                if self.state_space[player_position[0] - 1, player_position[1]] == 'G':
                    done = True
                    self.state = self.reset()
                    reward = 11
                else:
                    self.state = [player_position[0] - 1, player_position[1]]
                    self.state_space[player_position[0] - 1, player_position[1]] = 'P'
        """DOWN"""
        if action == 2:
            if player_position[0] + 1 < self.breadth \
                    and self.state_space[player_position[0] + 1, player_position[1]] != 'X':
                self.state_space[player_position[0], player_position[1]] = 'E'
                if self.state_space[player_position[0] + 1, player_position[1]] == 'G':
                    done = True
                    self.state = self.reset()
                    reward = 11
                else:
                    self.state = [player_position[0] + 1, player_position[1]]
                    self.state_space[player_position[0] + 1, player_position[1]] = 'P'
        """LEFT"""
        if action == 3:
            if player_position[1] - 1 >= 0 and self.state_space[player_position[0], player_position[1] - 1] != 'X':
                self.state_space[player_position[0], player_position[1]] = 'E'
                if self.state_space[player_position[0], player_position[1] - 1] == 'G':
                    done = True
                    self.state = self.reset()
                    reward = 11
                else:
                    self.state = [player_position[0], player_position[1] - 1]
                    self.state_space[player_position[0], player_position[1] - 1] = 'P'
        """RIGHT"""
        if action == 4:
            if player_position[1] + 1 < self.length \
                    and self.state_space[player_position[0], player_position[1] + 1] != 'X':
                self.state_space[player_position[0], player_position[1]] = 'E'
                if self.state_space[player_position[0], player_position[1] + 1] == 'G':
                    done = True
                    self.state = self.reset()
                    reward = 11
                else:
                    self.state = [player_position[0], player_position[1] + 1]
                    self.state_space[player_position[0], player_position[1] + 1] = 'P'
        if self.step_count == self.turn_limit:
            done = True
        return self.state, reward, done
