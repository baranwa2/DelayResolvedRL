import gym
# import gym_minigrid
import numpy as np
from collections import deque


class Environment:
    def __init__(self, game_name, delay, seed):
        """Initialize Environment"""
        self.game_name = game_name
        self.env = gym.make(self.game_name)
        self.env.seed(seed)
        np.random.seed(seed)
        self.number_of_actions = self.env.action_space.n
        self.delay = delay
        if 'MiniGrid' in self.game_name:
            self.state_space = self.env.observation_space['image']
        else:
            self.state_space = self.env.observation_space
        self.actions_in_buffer = deque(maxlen=self.delay)
        self.fill_up_buffer()
        self.delayed_action = 0

    def process_state(self, observation):
        """Pre-process state if required"""
        if 'MiniGrid' in self.game_name:
            return np.array(observation['image'], dtype='float32')  # Using only image as state (7x7x3)
        else:
            return observation

    def fill_up_buffer(self):
        for _ in range(self.delay):
            action = np.random.choice(self.number_of_actions)
            self.actions_in_buffer.append(action)

    def reset(self):
        state = self.env.reset()
        self.fill_up_buffer()
        if 'MiniGrid' in self.game_name:
            return self.process_state(state)
        else:
            return state

    def step(self, action):
        if self.delay != 0:
            chosen_action = action
            self.delayed_action = self.actions_in_buffer.popleft()
            self.actions_in_buffer.append(chosen_action)
        else:
            self.delayed_action = action
        if 'MiniGrid' in self.game_name:
            next_state, reward, done, info = self.env.step(self.delayed_action)
            return self.process_state(next_state), reward, done, info
        else:
            return self.env.step(self.delayed_action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
