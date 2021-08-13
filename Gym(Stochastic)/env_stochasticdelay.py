import gym
# import gym_minigrid
import numpy as np
import random
from collections import deque
import copy


class Environment:
    def __init__(self, seed, game_name, gamma, use_stochastic_delay, delay, min_delay):
        """Initialize Environment"""
        self.game_name = game_name
        self.env = gym.make(self.game_name)
        self.env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.number_of_actions = self.env.action_space.n
        self.delay = delay
        if 'MiniGrid' in self.game_name:
            self.state_space = self.env.observation_space['image']
        else:
            self.state_space = self.env.observation_space
        self.use_stochastic_delay = use_stochastic_delay
        self.no_action = 0
        self.index = 0
        self.use_stochastic_delay = use_stochastic_delay
        if self.use_stochastic_delay:
            self.min_delay = min_delay
            self.delay = self.min_delay
            self.max_delay = delay
        else:
            self.min_delay = delay
            self.delay = delay
            self.max_delay = delay
        self.state_buffer = deque(maxlen=self.max_delay + 2)
        self.reward_buffer = deque(maxlen=self.max_delay + 2)
        self.done_buffer = deque(maxlen=self.max_delay + 2)
        self.turn_limit = 200
        self.state = self.reset()
        self.update_delay()
        self.train = True
        self.step_count = 0
        self.delayed_action = 0
        self.gamma = gamma

    def process_state(self, observation):
        """Pre-process state if required"""
        if 'MiniGrid' in self.game_name:
            return np.array(observation['image'], dtype='float32')  # Using only image as state (7x7x3)
        else:
            return observation

    def reset(self):
        state = self.env.reset()
        if 'MiniGrid' in self.game_name:
            return self.process_state(state)
        else:
            return state

    def update_delay(self):
        if self.use_stochastic_delay:
            self.delay = random.randint(self.min_delay, self.max_delay)
        else:
            self.delay = self.max_delay

    def step(self, state, action):
        if self.max_delay != 0:
            self.train = True
            if True not in self.done_buffer:
                next_state, rewards, done, _ = self.env.step(action)
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
            elif len(self.state_buffer) > self.delay:  # delay is less than the number of unobserved states
                self.state_buffer.append(next_state)
                self.reward_buffer.append(rewards)
                self.done_buffer.append(done)
                rewards = 0
                no_observed_states = len(self.state_buffer) - self.delay
                for i in range(no_observed_states):
                    next_state = self.state_buffer.popleft()
                    gamma = np.power(self.gamma, no_observed_states-(i+1))
                    rewards += gamma * self.reward_buffer.popleft()  # add all unobserved rewards
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
            next_state, rewards, done, _ = self.env.step(action)
            return next_state, rewards, done

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
