import numpy as np
from collections import deque


class Environment:
    """Initialize Environment"""
    def __init__(self, seed, delay):
        np.random.seed(seed)
        self.breadth = 7
        self.length = 11
        self.state_space = np.empty([self.breadth, self.length], dtype='<U1')
        '''Environment Configuration'''
        self.state_space[:] = 'E'
        self.state_space[0] = 'X'
        self.state_space[1:4, self.length // 2 - 2] = 'X'
        self.state_space[1:4, self.length // 2 + 2] = 'X'
        self.state_space[0, self.length // 2 - 1:self.length // 2 + 2] = 'G'
        self.state_space[self.breadth - 1, 0] = 'P'
        '''Actions'''
        self.actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
        self.num_actions = len(self.actions)
        self.turn_limit = 300
        self.delay = delay
        self.actions_in_buffer = deque(maxlen=self.delay)
        self.fill_up_buffer()
        self.delayed_action = 0
        self.state = self.reset()

    def reset(self):
        x = np.random.randint(self.breadth)
        y = 0
        starting_state = [x, y]
        self.state_space[x, y] = 'P'
        self.fill_up_buffer()
        return starting_state

    def fill_up_buffer(self):
        for _ in range(self.delay):
            action = np.random.choice(self.num_actions)
            self.actions_in_buffer.append(action)

    def step(self, state, action):
        done = False
        player_position = state
        reward = -1
        """UP"""
        if action == 0:
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
        if action == 1:
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
        if action == 2:
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
        if action == 3:
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
        return self.state, reward, done
