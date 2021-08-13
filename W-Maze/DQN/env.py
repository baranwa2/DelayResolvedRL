import numpy as np
from collections import deque
import random


class Environment:
    """Initialize Environment"""
    def __init__(self, seed, delay):
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
        self.number_of_actions = len(self.actions)
        self.turn_limit = 300
        self.delay = delay
        self.actions_in_buffer = deque(maxlen=self.delay)
        self.fill_up_buffer()
        self.delayed_action = 0
        self.state = self.reset()
        self.step_count = 0

    def reset(self):
        x = random.randint(0, self.breadth-1)
        y = 0
        starting_state = [x, y]
        self.state_space[x, y] = 'P'
        self.fill_up_buffer()
        self.step_count = 0
        return starting_state

    def fill_up_buffer(self):
        for _ in range(self.delay):
            action = 0
            self.actions_in_buffer.append(action)

    def step(self, state, action):
        self.step_count += 1
        if self.delay != 0:
            chosen_action = action
            self.delayed_action = self.actions_in_buffer.popleft()  # get delayed action from buffer
            self.actions_in_buffer.append(chosen_action)  # append undelayed action to buffer
            action = self.delayed_action
        else:
            self.delayed_action = action
        done = False
        player_position = state
        self.state = player_position
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
                    self.state = player_position[0] + 1, player_position[1]
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
                    self.state = player_position[0], player_position[1] - 1
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
